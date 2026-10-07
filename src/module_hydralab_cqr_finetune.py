#!/usr/bin/env python
# coding: utf-8
"""
Fine-tuning (weight-update) extension for the Hydralab CQR heads.

This module *adds to* `module_hydralab_cqr` (it does not modify it). It provides
a warm-start continued-training path so that a set of already-trained CQR heads
(`head_NNNN.pt` + `metadata.json`, as written by `train_all_heads`) can be
further fine-tuned on NEW data, re-calibrated with a fresh conformal offset, and
re-saved in the identical checkpoint schema that `predict_with_head` /
`predict_all_heads` expect.

It also provides:
  * `hashing_embedding_index` — an OFFLINE, deterministic surrogate for
    `build_embedding_index`, so the whole pipeline can be tested without a
    GPU or a HuggingFace download. Real runs should use the Clinical ModernBERT
    `build_embedding_index` exactly as in the reference notebook; this surrogate
    is only for environments without the model.
  * `interval_score` / `evaluate_heads_on_holdout` — the Winkler interval score
    (Gneiting & Raftery 2007) plus coverage and width, for measuring whether the
    weight update improved the calibrated prediction intervals.

References
----------
Romano, Patterson, Candès (2019). Conformalized Quantile Regression. NeurIPS.
Gneiting & Raftery (2007). Strictly Proper Scoring Rules, Prediction, and
    Estimation. JASA 102(477):359-378  (interval / Winkler score, eq. 43).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import module_hydralab_cqr as mh


# =============================================================================
# Offline surrogate embedding (for feasibility test only; real runs use Clinical ModernBERT)
# =============================================================================
def hashing_embedding_index(long_df: pd.DataFrame,
                            id_col: str = "id",
                            text_col: str = "text",
                            question_col: Optional[str] = "question",
                            n_features: int = 256,
                            ngram_range: tuple = (1, 2),
                            seed: int = 0) -> dict:
    """
    Deterministic offline stand-in for `build_embedding_index`.

    Deduplicates by `id_col` (one vector per patient, exactly like the real
    Stage-1 embedder), then hashes whitespace/词 n-grams of the text (optionally
    prepended with the id's first question) into a fixed-width L2-normalised
    vector. Because the synthetic notes literally contain the phenotype tokens
    (e.g. "diabetes", "alcohol", BMI values), this surrogate carries real
    predictive signal — enough to train and fine-tune heads and to demonstrate
    interval-score changes — WITHOUT a GPU or network.

    Parameters
    ----------
    long_df : pandas.DataFrame
        Long-form table (one row per id, code, value); text constant per id.
    id_col, text_col, question_col : str
        Column names. `question_col=None` disables question prepending.
    n_features : int
        Output embedding dimension (hash space size). Default 256.
    ngram_range : tuple[int, int]
        Word n-gram range for the hashing vectorizer. Default (1, 2).
    seed : int
        Unused placeholder for API symmetry (hashing is already deterministic).

    Returns
    -------
    dict
        Same keys/shape contract as `build_embedding_index`:
        {'embeddings': (n_ids, n_features) float32,
         'selection_embeddings': None,
         'id_to_row': {id -> row index},
         'id_order': [ids in row order]}.
    """
    from sklearn.feature_extraction.text import HashingVectorizer

    text_df = long_df.drop_duplicates(subset=[id_col]).reset_index(drop=True)
    ids = text_df[id_col].tolist()
    texts = text_df[text_col].astype(str)
    if question_col is not None and question_col in text_df.columns:
        texts = (text_df[question_col].astype(str) + " [SEP] " + texts)
    vec = HashingVectorizer(n_features=n_features, alternate_sign=False,
                            norm="l2", ngram_range=ngram_range)
    X = vec.transform(texts.tolist()).astype(np.float32).toarray()
    id_to_row = {id_val: i for i, id_val in enumerate(ids)}
    return {"embeddings": X, "selection_embeddings": None,
            "id_to_row": id_to_row, "id_order": ids}


# =============================================================================
# Warm-start fine-tuning of one head
# =============================================================================
def finetune_single_head(model: "mh.FFNNQuantileRegressor",
                         X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         taus: tuple, device: str,
                         y_mean: float, y_std: float,
                         lr: float = 1e-4, weight_decay: float = 1e-5,
                         batch_size: int = 256, max_epochs: int = 200,
                         patience: int = 10, min_delta: float = 1e-5,
                         freeze_hidden: bool = False):
    """
    Continue training an EXISTING FFNNQuantileRegressor on new data (warm start).

    Unlike `mh.train_single_head`, this does NOT re-initialise the network and
    does NOT recompute the target standardisation: it reuses the head's stored
    `y_mean`/`y_std` so the pre-trained weights stay meaningful and the model's
    standardized output space is unchanged. Only the weights move, nudged toward
    the new data by a small learning rate.

    Parameters
    ----------
    model : mh.FFNNQuantileRegressor
        Pre-trained head with weights already loaded (the object is updated
        in place and also returned).
    X_train, y_train, X_val, y_val : numpy.ndarray
        New-data feature/target splits (raw y in original units).
    taus : tuple[float, ...]
        Quantile levels, matching the head's output columns (e.g. (0.025,0.5,0.975)).
    device : str
        Torch device.
    y_mean, y_std : float
        Standardisation stats from the pre-trained checkpoint (kept fixed).
    lr : float
        Fine-tuning learning rate (default 1e-4 — an order of magnitude below the
        1e-3 used for from-scratch training, to avoid catastrophic forgetting).
    weight_decay, batch_size, max_epochs, patience, min_delta : ...
        Standard optimisation controls; early stopping on val pinball loss.
    freeze_hidden : bool
        If True, freeze all layers except the final Linear (calibrate only the
        output head). Useful when the new data is small. Default False.

    Returns
    -------
    model : mh.FFNNQuantileRegressor
        The fine-tuned head (best-val-loss state reloaded).
    history : dict
        {'train_loss': [...], 'val_loss': [...]} per epoch (standardized space).
    best_val_loss : float
        Best validation pinball loss in standardized y-space.

    Notes
    -----
    Loss is the multi-quantile pinball loss `mh.pinball_loss`; targets are
    standardized with the fixed (y_mean, y_std) before the loss, exactly as in
    the original trainer, so val losses are directly comparable pre/post update.
    """
    taus_t = torch.tensor(taus, dtype=torch.float32, device=device)
    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    Xt = torch.from_numpy(np.asarray(X_train)).float()
    yt = torch.from_numpy(y_train_n).float()
    Xv = torch.from_numpy(np.asarray(X_val)).float().to(device)
    yv = torch.from_numpy(y_val_n).float().to(device)

    if freeze_hidden:
        last_linear = [m for m in model.net if hasattr(m, "weight")][-1]
        for p in model.parameters():
            p.requires_grad = False
        for p in last_linear.parameters():
            p.requires_grad = True

    model = model.to(device)
    train_loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size,
                              shuffle=True, drop_last=False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    best_val = float("inf"); best_state = None; no_improve = 0
    history = {"train_loss": [], "val_loss": []}
    for _ in range(1, max_epochs + 1):
        model.train(); running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            loss = mh.pinball_loss(model(xb), yb, taus_t)
            loss.backward(); opt.step()
            running += loss.item() * xb.size(0)
        tr = running / max(len(Xt), 1)
        model.eval()
        with torch.no_grad():
            vl = mh.pinball_loss(model(Xv), yv, taus_t).item()
        history["train_loss"].append(tr); history["val_loss"].append(vl)
        if vl < best_val - min_delta:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val


# =============================================================================
# Warm-start fine-tuning of every head in a directory
# =============================================================================
def _load_quantile_model(ckpt: dict, device: str) -> "mh.FFNNQuantileRegressor":
    """Rebuild an FFNNQuantileRegressor from a checkpoint and load its weights."""
    model = mh.FFNNQuantileRegressor(
        in_dim=int(ckpt["quantile_in_dim"]),
        n_quantiles=int(ckpt["n_quantiles"]),
        hidden=int(ckpt.get("quantile_hidden", 256)),
        depth=int(ckpt.get("quantile_depth", 2)),
    ).to(device)
    model.load_state_dict(ckpt["quantile_state_dict"])
    return model


def finetune_all_heads(initial_heads_dir: Path,
                       updated_heads_dir: Path,
                       new_long_df: pd.DataFrame,
                       new_embeddings: np.ndarray,
                       new_id_to_row: dict,
                       id_col: str = "id", code_col: str = "code",
                       value_col: str = "value",
                       alpha: float = 0.05,
                       val_size: float = 0.15, calib_size: float = 0.15,
                       min_calib_samples: int = 16,
                       min_samples_per_head: int = 8,
                       lr: float = 1e-4, max_epochs: int = 200,
                       base_patience: int = 10, batch_size: int = 256,
                       freeze_hidden: bool = False,
                       recalibrate: bool = True,
                       train_new_heads: bool = True,
                       min_samples_new_head: int = 32,
                       new_head_depth: Optional[int] = None,
                       new_head_hidden: Optional[int] = None,
                       new_head_lr: float = 1e-3,
                       new_head_max_epochs: int = 1000,
                       random_state: int = 42,
                       device: Optional[str] = None) -> list:
    """
    Warm-start fine-tune every trained head in `initial_heads_dir` on NEW data and
    write the updated heads to `updated_heads_dir` in the same on-disk format.

    For each head:
      1. Load its checkpoint and rebuild the quantile MLP with its stored weights.
      2. Gather the new-data rows for that target (via `long_to_per_target`),
         three-way split them (train/val/calib) reusing the base pipeline's splitter.
      3. Continue training with `finetune_single_head` (low LR, early stopping),
         keeping the checkpoint's (y_mean, y_std) fixed.
      4. Re-fit the CQR conformal offset Q_hat on the fresh calibration split so the
         (1 - alpha) coverage guarantee tracks the updated model and new data.
      5. Update the plausible-support bounds y_min/y_max to the union of the old
         and new observed ranges, and re-save in the identical schema.
    Heads whose target has too few (or no) new observations are COPIED THROUGH
    unchanged, so the updated directory is a complete, self-sufficient head set.

    NEW TARGETS: any code present in `new_long_df` that has no existing head is,
    when `train_new_heads=True`, trained from scratch (same three-way-split →
    pinball → CQR procedure and identical checkpoint schema as
    `train_all_heads`) and appended with a fresh index continuing after the
    existing heads. New heads reuse the directory's existing depth/width so the
    updated set stays architecturally consistent (override via `new_head_depth`/
    `new_head_hidden`). This lets a fine-tune pass simultaneously (a) update the
    weights of existing heads and (b) grow the head set to cover newly observed
    targets.

    Parameters
    ----------
    initial_heads_dir : pathlib.Path
        Directory of the initially trained heads (head_NNNN.pt + metadata.json).
    updated_heads_dir : pathlib.Path
        Destination directory for the fine-tuned heads (created/overwritten).
    new_long_df : pandas.DataFrame
        New long-form data (id, code, value[, text, question]).
    new_embeddings : numpy.ndarray
        (n_new_ids, hidden) embeddings of the new data, hidden dim == training dim.
    new_id_to_row : dict
        id -> row index into `new_embeddings`.
    id_col, code_col, value_col : str
        Column names in `new_long_df`.
    alpha : float
        Miscoverage rate; must match the heads being updated (e.g. 0.05 -> 95% PI).
    val_size, calib_size : float
        New-data validation / calibration fractions per head.
    min_calib_samples : int
        Below this many new calib rows, Q_hat is left at the checkpoint's value
        (existing heads) or the new head is saved uncalibrated (new heads).
    min_samples_per_head : int
        Below this many new observations, an EXISTING head is copied through unchanged.
    lr, max_epochs, base_patience, batch_size, freeze_hidden : ...
        Passed to `finetune_single_head` (existing-head warm start).
    recalibrate : bool
        If True (default) re-fit Q_hat on the new calib split; else keep the old Q_hat.
    train_new_heads : bool
        If True (default), train brand-new heads for codes in `new_long_df` that
        have no existing head. If False, such codes are ignored.
    min_samples_new_head : int
        Minimum new observations required to train a NEW head from scratch
        (below this the new code is recorded as skipped). Default 32, matching
        `train_all_heads`'s `min_samples_per_head`.
    new_head_depth, new_head_hidden : int or None
        Architecture for new heads. None (default) reuses the existing heads'
        depth/width found in `initial_heads_dir`.
    new_head_lr : float
        Learning rate for from-scratch new-head training (default 1e-3, the
        module's normal training LR — higher than the 1e-4 warm-start LR).
    new_head_max_epochs : int
        Max epochs for new-head training (early stopping usually fires first).
    random_state : int
        Seed for the per-head new-data split.
    device : str or None
        Torch device (auto: cuda if available).

    Returns
    -------
    list[dict]
        Updated metadata (also written to `updated_heads_dir/metadata.json`).
        Existing-head entries gain fine-tuning diagnostics ('finetuned',
        'n_finetune_obs', 'epochs_run_finetune', 'val_pinball_before'/'after',
        'Q_hat_before'/'after', refreshed calib metrics). New-head entries carry
        'newly_trained': True, 'status', 'n_new_obs', and the same training/
        calibration fields `train_all_heads` records.

    Side effects
    ------------
    Writes head_NNNN.pt for every head and a metadata.json to updated_heads_dir.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    initial_heads_dir = Path(initial_heads_dir)
    updated_heads_dir = Path(updated_heads_dir)
    updated_heads_dir.mkdir(parents=True, exist_ok=True)

    taus = mh._quantile_levels_from_alpha(alpha)
    meta = json.loads((initial_heads_dir / "metadata.json").read_text())

    # New-data per-target features/targets keyed by code
    per_target = mh.long_to_per_target(
        new_long_df, embeddings=new_embeddings, id_to_row=new_id_to_row,
        id_col=id_col, code_col=code_col, value_col=value_col,
        include_selection=False)
    new_by_code = {name: (X, y) for (name, X, y) in per_target}

    out_meta = []
    for entry in meta:
        idx = entry.get("idx")
        src_ckpt_path = initial_heads_dir / f"head_{idx:04d}.pt"
        # Non-trained or missing checkpoints: pass metadata through untouched.
        if entry.get("status") != "trained" or not src_ckpt_path.exists():
            out_meta.append({**entry, "finetuned": False})
            continue

        ckpt = torch.load(src_ckpt_path, map_location=device, weights_only=False)
        name = entry["name"]
        dst_ckpt_path = updated_heads_dir / f"head_{idx:04d}.pt"

        # Evaluate pre-update val pinball on the new val split (for reporting),
        # then decide whether we have enough new data to fine-tune.
        has_new = name in new_by_code and len(new_by_code[name][1]) >= min_samples_per_head
        if not has_new:
            # copy weights through unchanged
            torch.save(ckpt, dst_ckpt_path)
            out_meta.append({**entry, "finetuned": False,
                             "n_finetune_obs": int(len(new_by_code[name][1]) if name in new_by_code else 0)})
            continue

        X_new, y_new = new_by_code[name]
        y_new = np.asarray(y_new, dtype=np.float64)
        finite = np.isfinite(y_new)
        X_new, y_new = X_new[finite], y_new[finite]

        X_tr, y_tr, X_val, y_val, X_cal, y_cal = mh._random_three_way_split(
            X_new, y_new, val_size=val_size, calib_size=calib_size,
            random_state=random_state)

        y_mean = float(ckpt["y_mean"]); y_std = float(ckpt["y_std"])
        model = _load_quantile_model(ckpt, device)

        # pre-update val pinball (standardized) for delta reporting
        with torch.no_grad():
            taus_t = torch.tensor(taus, dtype=torch.float32, device=device)
            yv_n = torch.from_numpy((y_val - y_mean) / y_std).float().to(device)
            Xv_t = torch.from_numpy(np.asarray(X_val)).float().to(device)
            val_before = float(mh.pinball_loss(model(Xv_t), yv_n, taus_t).item())

        model, hist, val_after = finetune_single_head(
            model, X_tr, y_tr, X_val, y_val, taus=taus, device=device,
            y_mean=y_mean, y_std=y_std, lr=lr, max_epochs=max_epochs,
            patience=base_patience, batch_size=batch_size, freeze_hidden=freeze_hidden)

        # Re-fit conformal offset on the fresh calib split
        Q_before = ckpt.get("Q_hat")
        Q_after, calib_metrics, calib_status = Q_before, None, "kept_previous"
        if recalibrate and len(y_cal) >= min_calib_samples:
            Q_after, calib_metrics = mh.fit_cqr_offset(
                model, X_cal, y_cal, y_mean=y_mean, y_std=y_std,
                device=device, alpha=alpha)
            calib_status = "refit"

        # Update plausible-support bounds to union(old, new observed)
        y_min = ckpt.get("y_min"); y_max = ckpt.get("y_max")
        y_min = float(min(y_min, y_new.min())) if y_min is not None else float(y_new.min())
        y_max = float(max(y_max, y_new.max())) if y_max is not None else float(y_new.max())

        new_ckpt = dict(ckpt)  # preserve every field predict_with_head reads
        new_ckpt["quantile_state_dict"] = model.state_dict()
        new_ckpt["Q_hat"] = Q_after
        new_ckpt["alpha"] = float(alpha) if Q_after is not None else None
        new_ckpt["y_min"] = y_min; new_ckpt["y_max"] = y_max
        torch.save(new_ckpt, dst_ckpt_path)

        upd = {**entry, "finetuned": True,
               "n_finetune_obs": int(len(y_new)),
               "epochs_run_finetune": len(hist["val_loss"]),
               "val_pinball_before": val_before,
               "val_pinball_after": float(val_after),
               "Q_hat_before": (float(Q_before) if Q_before is not None else None),
               "Q_hat_after": (float(Q_after) if Q_after is not None else None),
               "finetune_calib_status": calib_status,
               "y_train_min": y_min, "y_train_max": y_max}
        if calib_metrics is not None:
            upd.update({f"finetune_{k}": v for k, v in calib_metrics.items()})
        out_meta.append(upd)

    # ---------------------------------------------------------------------
    # NEW heads: codes present in the new data but with no existing head are
    # trained from scratch (same procedure/schema as train_all_heads) and
    # appended with fresh indices continuing after the existing heads.
    # ---------------------------------------------------------------------
    existing_names = {e["name"] for e in meta if "name" in e}
    new_codes = [c for c in sorted(new_by_code) if c not in existing_names]

    if train_new_heads and new_codes:
        # Resolve the architecture for new heads: reuse the directory's existing
        # depth/width (all heads in one dir share an architecture) unless overridden.
        depth_r, hidden_r = new_head_depth, new_head_hidden
        if depth_r is None or hidden_r is None:
            probe = next((e for e in meta if e.get("status") == "trained"
                          and (initial_heads_dir / f"head_{e['idx']:04d}.pt").exists()), None)
            if probe is not None:
                pck = torch.load(initial_heads_dir / f"head_{probe['idx']:04d}.pt",
                                 map_location=device, weights_only=False)
                depth_r = int(pck.get("quantile_depth", 2)) if depth_r is None else depth_r
                hidden_r = int(pck.get("quantile_hidden", 256)) if hidden_r is None else hidden_r
            else:
                depth_r = 2 if depth_r is None else depth_r
                hidden_r = 256 if hidden_r is None else hidden_r

        next_idx = max((e["idx"] for e in meta if "idx" in e), default=-1) + 1
        for code in new_codes:
            X_c, y_c = new_by_code[code]
            y_c = np.asarray(y_c, dtype=np.float64)
            finite = np.isfinite(y_c)
            X_c, y_c = X_c[finite], y_c[finite]

            skip = None
            if len(y_c) < min_samples_new_head:
                skip = "skipped_too_few_samples"
            elif np.std(y_c) < 1e-12:
                skip = "skipped_no_variance"
            if skip is not None:
                out_meta.append({"idx": next_idx, "name": code, "status": skip,
                                 "newly_trained": True, "finetuned": False,
                                 "n_new_obs": int(len(y_c))})
                next_idx += 1
                continue

            result = mh._train_head_vanilla_cqr(
                X_obs=X_c, y=y_c, taus=taus, calibrate=recalibrate,
                val_size=val_size, calib_size=calib_size, random_state=random_state,
                alpha=alpha, min_calib_samples=min_calib_samples,
                head_patience=base_patience, device=device, depth=depth_r,
                hidden=hidden_r, max_epochs=new_head_max_epochs, lr=new_head_lr,
                batch_size=batch_size)

            if result.get("status") != "trained":
                out_meta.append({"idx": next_idx, "name": code,
                                 "status": result.get("status", "skipped"),
                                 "newly_trained": True, "finetuned": False,
                                 "n_new_obs": int(len(y_c))})
                next_idx += 1
                continue

            # Save checkpoint in the identical schema train_all_heads writes.
            q_model = result["quantile_model"]
            ckpt = {
                "quantile_state_dict": q_model.state_dict(),
                "quantile_in_dim": int(q_model.net[0].in_features),
                "quantile_hidden": int(q_model.hidden),
                "quantile_depth": int(q_model.depth),
                "n_quantiles": len(taus),
                "quantile_levels": list(taus),
                "target_name": code,
                "y_mean": float(result["y_mean"]),
                "y_std": float(result["y_std"]),
                "y_min": (float(result["y_min"]) if result.get("y_min") is not None else None),
                "y_max": (float(result["y_max"]) if result.get("y_max") is not None else None),
                "Q_hat": result["Q_hat"],
                "alpha": float(alpha) if result["Q_hat"] is not None else None,
                "selection_used": False,
            }
            torch.save(ckpt, updated_heads_dir / f"head_{next_idx:04d}.pt")

            entry = {
                "idx": next_idx, "name": code, "status": "trained",
                "selection_used": False, "outcome_depth": int(depth_r),
                "epochs_run_quantile": len(result["history_quantile"]["val_loss"]),
                "best_val_pinball_standardized": result["best_val_pinball_standardized"],
                "y_train_mean": float(result["y_mean"]),
                "y_train_std": float(result["y_std"]),
                "y_train_min": (float(result["y_min"]) if result.get("y_min") is not None else None),
                "y_train_max": (float(result["y_max"]) if result.get("y_max") is not None else None),
                "quantile_levels": list(taus),
                "calibration_status": result["calib_status"],
                "n_train": result["n_train"], "n_val": result["n_val"],
                "newly_trained": True, "finetuned": False,
                "n_new_obs": int(len(y_c)),
            }
            if result.get("calib_metrics") is not None:
                entry.update(result["calib_metrics"])
            out_meta.append(entry)
            next_idx += 1

    (updated_heads_dir / "metadata.json").write_text(json.dumps(out_meta, indent=2, default=str))
    return out_meta


# =============================================================================
# Interval-score evaluation on a holdout set
# =============================================================================
def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                   alpha: float) -> np.ndarray:
    """
    Per-observation Winkler / interval score for central (1 - alpha) intervals.

    Formula (Gneiting & Raftery 2007, eq. 43):
        S = (u - l)
            + (2/alpha) * (l - y) * 1[y < l]
            + (2/alpha) * (y - u) * 1[y > u]
    Lower is better: the score rewards narrow intervals and penalises misses in
    proportion to how far outside the interval the observation falls, scaled by
    2/alpha. It is a strictly proper scoring rule for the interval.

    Parameters
    ----------
    y : numpy.ndarray
        True observed values, shape (n,).
    lower, upper : numpy.ndarray
        Calibrated interval bounds, shape (n,).
    alpha : float
        Miscoverage rate (e.g. 0.05 for a 95% interval).

    Returns
    -------
    numpy.ndarray
        Per-observation interval scores, shape (n,). Take `.mean()` for the
        mean interval score.
    """
    y = np.asarray(y, float); lower = np.asarray(lower, float); upper = np.asarray(upper, float)
    width = upper - lower
    below = (lower - y) * (y < lower)
    above = (y - upper) * (y > upper)
    return width + (2.0 / alpha) * (below + above)


def evaluate_heads_on_holdout(heads_dir: Path, holdout_long_df: pd.DataFrame,
                              holdout_embeddings: np.ndarray, holdout_id_order: list,
                              alpha: float = 0.05,
                              id_col: str = "id", code_col: str = "code",
                              value_col: str = "value") -> pd.DataFrame:
    """
    Score a directory of heads on a holdout set: coverage, width, interval score.

    Runs `predict_all_heads` to get per-target lower/upper/pred, melts to long,
    joins the true holdout values, and computes per-(row) coverage, interval
    width, and the Winkler interval score.

    Parameters
    ----------
    heads_dir : pathlib.Path
        Directory of heads to evaluate.
    holdout_long_df : pandas.DataFrame
        Holdout long-form data (id, code, value).
    holdout_embeddings : numpy.ndarray
        Embeddings for the holdout ids (row order == holdout_id_order).
    holdout_id_order : list
        Ids in embedding-row order (from the embedder's id_order).
    alpha : float
        Miscoverage rate used for the interval score.
    id_col, code_col, value_col : str
        Column names in `holdout_long_df`.

    Returns
    -------
    pandas.DataFrame
        Long-form per-observation evaluation with columns:
        id, code, value, pred, lower, upper, covered (0/1), width, interval_score.
        Only (id, code) pairs that were actually observed in the holdout are kept.
    """
    preds = mh.predict_all_heads(Path(heads_dir), holdout_embeddings)
    preds.insert(0, id_col, list(holdout_id_order))

    def _melt(suffix, value_name):
        cols = [id_col] + [c for c in preds.columns if c.endswith(suffix)]
        m = preds[cols].melt(id_vars=[id_col], var_name=code_col, value_name=value_name)
        m[code_col] = m[code_col].str.replace(suffix, "", regex=False)
        return m

    pm = _melt("__pred", "pred")
    lm = _melt("__lower", "lower")
    um = _melt("__upper", "upper")
    wide = pm.merge(lm, on=[id_col, code_col]).merge(um, on=[id_col, code_col])

    truth = holdout_long_df[[id_col, code_col, value_col]].rename(columns={value_col: "value"})
    ev = truth.merge(wide, on=[id_col, code_col], how="inner").dropna(subset=["lower", "upper"])
    ev["covered"] = ((ev["value"] >= ev["lower"]) & (ev["value"] <= ev["upper"])).astype(int)
    ev["width"] = ev["upper"] - ev["lower"]
    ev["interval_score"] = interval_score(ev["value"].to_numpy(),
                                          ev["lower"].to_numpy(),
                                          ev["upper"].to_numpy(), alpha=alpha)
    return ev
