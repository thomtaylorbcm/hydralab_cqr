#!/usr/bin/env python
# coding: utf-8

"""
Language Model -> 500 CQR heads with optional Heckman selection
(deeper networks + heteroscedastic selection via reparameterization
+ separate selection text for true exclusion restriction).


  (1) `--selection-text-column` / `selection_text_col`
      A separate text column embedded by Clinical ModernBERT for the
      SELECTION equation only. The outcome equation still uses the
      `text_col` embedding. This gives a genuine EXCLUSION RESTRICTION:
      a feature space that affects ordering but not outcome value.
      Without this, Heckman identification rested entirely on the
      nonlinearity of the IMR — which a flexible MLP can replicate
      without explicit IMR.

  (2) `--depth` and `--selection-depth`
      Configurable depth for the outcome quantile MLP and the selection
      MLP independently. depth=2 (default) reproduces the v1
      architecture. Larger values stack more hidden layers at constant
      width.

  (3) `--reparam-selection`
      Replaces the standard probit selection model with a HETEROSCEDASTIC
      probit:
          z = mu(x) + sigma(x) * eta,   eta ~ N(0, 1)
      where both mu and sigma are learned functions of x, and eta is a
      fixed standard-normal sample (the "reparameterization trick" —
      gradients flow through mu and sigma during training).

      Klein & Vella (2010) showed that when sigma(x)
      varies with x in a way distinguishable from the functional form
      of the outcome equation, Heckman identification is achievable
      without an exclusion restriction — i.e. identification by
      heteroscedasticity. This is a useful complement to (1): even
      without separate selection text, heteroscedastic noise in the
      latent index can rescue identification.

      Reference: Klein, R., & Vella, F. (2010). Estimating a class of
      triangular simultaneous equations models without exclusion
      restrictions. Journal of Econometrics, 154(2), 154-164.

================================================================
CLI ARGUMENTS (also documented per-argument in argparse --help)
================================================================

  DATA & INPUT
    --data PATH                   (required) Path to long-form CSV with
                                  one row per (id, code, value) observation.
    --id-column NAME              Patient / note identifier. text and
                                  question must be constant within an id.
                                  [default: id]
    --text-column NAME            Clinical text column (constant per id).
                                  [default: text]
    --code-column NAME            Column naming which target this row is
                                  for (e.g. ICD code, lab name).
                                  [default: code]
    --value-column NAME           Observed continuous value for (id, code).
                                  NaN rows are dropped.
                                  [default: value]

  EMBEDDING / BERT
    --model-name NAME             HuggingFace model id for the encoder.
                                  [default: Simonlee711/Clinical_ModernBERT]
    --batch-size N                Batch size for the BERT forward pass.
                                  [default: 16]
    --max-length N                Max token length per chunk.
                                  [default: 512]
    --question-column NAME        (Optional) Column whose value is
                                  prepended to every chunk during
                                  tokenization, format:
                                      [CLS] question [SEP] text [SEP]
                                  Must be constant per id, non-empty when
                                  set.
                                  [default: None]
    --chunk-stride N              Token overlap between adjacent chunks
                                  of the same text. Only used when
                                  --question-column is set.
                                  [default: 0]
    --chunk-aggregation MODE      How to combine per-chunk embeddings:
                                  'max' or 'mean'. Only used when
                                  --question-column is set.
                                  [default: max]

  OUTCOME (QUANTILE) MLP
    --hidden N                    Width of every hidden layer in the
                                  outcome MLP.
                                  [default: 256]
    --depth N                     Number of hidden layers in the outcome
                                  MLP. depth=2 reproduces the v1
                                  architecture.
                                  [default: 2]
    --max-epochs N                Hard cap on training epochs per head;
                                  early stopping usually fires first.
                                  [default: 200]
    --base-patience N             Epochs without val-loss improvement
                                  before early-stopping.
                                  [default: 10]
    --val-size F                  Fraction of labeled rows held out per
                                  head for validation / early stopping.
                                  [default: 0.15]
    --min-samples-per-head N      Heads with fewer observed values than
                                  this are skipped (recorded in
                                  metadata.json).
                                  [default: 32]

  CQR CONFORMAL CALIBRATION
    --alpha F                     Miscoverage rate. alpha=0.1 -> 90%
                                  prediction intervals. Also sets the
                                  quantile levels (alpha/2, 0.5,
                                  1-alpha/2) trained against.
                                  [default: 0.1]
    --calib-size F                Fraction of labeled rows held out per
                                  head for the conformal calibration
                                  step.
                                  [default: 0.15]
    --min-calib-samples N         Heads with fewer calibration samples
                                  save Q_hat=None and produce NaN
                                  interval bounds.
                                  [default: 16]
    --no-calibrate                Disable the CQR conformal step. Point
                                  predictions still produced; lower /
                                  upper bounds are NaN.

  HECKMAN SELECTION CORRECTION (all require --selection)
    --selection                   Toggle the Heckman two-step path on.
                                  Without this flag the pipeline is
                                  vanilla CQR (same as the previous file).
    --selection-text-column NAME  (Optional) Separate text column
                                  embedded for the SELECTION equation
                                  only. Gives a true EXCLUSION
                                  RESTRICTION. Must be constant per id
                                  and non-empty for at least most ids.
                                  Ignored if --selection is not set.
                                  [default: None]
    --selection-hidden N          Width of every hidden layer in the
                                  selection MLP.
                                  [default: 128]
    --selection-depth N           Number of hidden layers in the
                                  selection MLP.
                                  [default: 2]
    --selection-max-epochs N      Hard cap on epochs for the selection
                                  model. Patience reuses --base-patience.
                                  [default: 100]
    --reparam-selection           Use the heteroscedastic probit
                                  selection model
                                      z = mu(x) + sigma(x) * eta,
                                      eta ~ N(0, 1)
                                  sampled via the reparameterization
                                  trick. Enables Klein-Vella (2010)
                                  identification through
                                  heteroscedasticity.
    --min-sigma F                 Floor on sigma(x) for the reparam
                                  selection model. Keeps gradients
                                  well-conditioned for very negative
                                  raw sigma outputs.
                                  [default: 0.1]

  CACHING & OUTPUT
    --cache-dir PATH              Directory for embeddings.npy and
                                  id_order.json (and
                                  selection_embeddings.npy when
                                  --selection-text-column is used).
                                  [default: cache]
    --heads-dir PATH              Directory for the per-head .pt files
                                  and metadata.json.
                                  [default: heads_cqr_heckman]
    --skip-embedding              Skip Stage 1 if cached embeddings
                                  already exist in --cache-dir. Useful
                                  for iterating on Stage 2 hyperparams
                                  without re-running BERT.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


# =============================================================================
# Stage 1 — Embedding extraction
# =============================================================================

def masked_max_pool(last_hidden_state: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Max-pool token embeddings over the sequence dimension, ignoring padding.

    Standard max-pooling over the time axis would let padding-token activations
    (which BERT-family models produce arbitrary values for) win the max for
    short inputs. We replace those positions with the dtype's minimum value
    before reducing, so only real tokens contribute.

    Args:
        last_hidden_state: Token-level encoder outputs, shape (B, T, H) where
                           B is batch size, T is sequence length, H is hidden dim.
        attention_mask:    Mask of real-vs-padding tokens, shape (B, T), with
                           1 for real tokens and 0 for padding.

    Returns:
        Pooled embeddings, shape (B, H). The j-th row is the elementwise max
        over all real tokens of the j-th input sequence.
    """
    mask = attention_mask.unsqueeze(-1).bool()
    very_neg = torch.finfo(last_hidden_state.dtype).min
    masked = last_hidden_state.masked_fill(~mask, very_neg)
    pooled, _ = masked.max(dim=1)
    return pooled


def masked_mean_pool(last_hidden_state: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool token embeddings over the sequence dimension, ignoring padding.

    Averages the hidden states of the real (non-padding) tokens. Padding
    positions are zeroed before summation and excluded from the denominator,
    so a short sequence is not diluted by its padding. This is the pooling
    used by Sentence-BERT and is a robust default for both encoder models and
    for decoder models whose attention has been made bidirectional over the
    input (e.g. an all-prefix PrefixLM forward, or LLM2Vec-style conversion).

    Equation (per sequence, m_t in {0, 1} the attention mask, h_t the token
    hidden state):

        e = (sum_t  m_t * h_t) / max(1, sum_t m_t)

    Args:
        last_hidden_state: Token-level model outputs, shape (B, T, H) where
                           B is batch size, T is sequence length, H is hidden
                           dim.
        attention_mask:    Mask of real-vs-padding tokens, shape (B, T), with
                           1 for real tokens and 0 for padding.

    Returns:
        Pooled embeddings, shape (B, H). The b-th row is the average of the
        real-token hidden states of the b-th input sequence.

    References:
        Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence
        Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP.
        arXiv:1908.10084. (Mean-pooling of token embeddings.)
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # (B, T, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                    # (B, H)
    counts = mask.sum(dim=1).clamp(min=1.0)                           # (B, 1)
    return summed / counts


def masked_last_token_pool(last_hidden_state: torch.Tensor,
                           attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Last-real-token pooling — the decoder/causal-LM analogue of
    `masked_max_pool`.

    In a decoder-only (generative) model with causal attention, the hidden
    state at position t has only attended to tokens <= t. Only the LAST real
    token has therefore "seen" the entire input, which makes its hidden state
    the natural whole-sequence summary. This is the pooling strategy used by
    OpenAI's cpt-text and by E5-Mistral; it is the principled counterpart to
    max-pooling (which suits bidirectional encoders, where every token already
    has full context).

    This implementation locates the last real token from the attention mask
    rather than assuming a padding side, so it is correct for BOTH right-
    padded and left-padded batches:

        tau = max { t : m_t = 1 }          (index of the last real token)
        e   = h_tau

    Args:
        last_hidden_state: Token-level model outputs, shape (B, T, H) where
                           B is batch size, T is sequence length, H is hidden
                           dim.
        attention_mask:    Mask of real-vs-padding tokens, shape (B, T), with
                           1 for real tokens and 0 for padding. Must contain at
                           least one real token per row.

    Returns:
        Pooled embeddings, shape (B, H). The b-th row is the hidden state of
        the last non-padding token of the b-th input sequence.

    Raises:
        ValueError: If any row of `attention_mask` is entirely padding (no
                    real token to pool).

    References:
        Neelakantan, A. et al. (2022). Text and Code Embeddings by Contrastive
        Pre-Training (cpt-text). arXiv:2201.10005. (Uses the hidden state of
        the special last token of a GPT-style decoder as the embedding.)
        Wang, L. et al. (2024). Improving Text Embeddings with Large Language
        Models (E5-Mistral). arXiv:2401.00368. (Last-token pooling on a
        decoder-only LM.)
    """
    B, T, _ = last_hidden_state.shape
    if (attention_mask.sum(dim=1) == 0).any():
        raise ValueError("masked_last_token_pool: at least one sequence is all "
                         "padding (no real token to pool).")
    # Index of the last position whose mask is 1, robust to padding side.
    pos = torch.arange(T, device=attention_mask.device).unsqueeze(0)      # (1, T)
    masked_pos = pos.masked_fill(attention_mask == 0, -1)                 # (B, T)
    last_idx = masked_pos.max(dim=1).values                              # (B,)
    batch_idx = torch.arange(B, device=last_hidden_state.device)
    return last_hidden_state[batch_idx, last_idx]                         # (B, H)


# Registry mapping a pooling-strategy name to its function. Lets callers
# (and the CLI) pick a pooling strategy by string. 'max' and 'mean' are
# appropriate for bidirectional encoders (or PrefixLM-bidirectional decoder
# forwards); 'last' is appropriate for plain causal decoder forwards.
POOLING_FUNCTIONS = {
    "max": masked_max_pool,
    "mean": masked_mean_pool,
    "last": masked_last_token_pool,
}


def pool_token_embeddings(last_hidden_state: torch.Tensor,
                          attention_mask: torch.Tensor,
                          strategy: str = "max") -> torch.Tensor:
    """
    Dispatch to a masked pooling function by name.

    Thin wrapper over `POOLING_FUNCTIONS` so that encoder and generative code
    paths can share a single, string-configurable pooling call.

    Args:
        last_hidden_state: Token-level model outputs, shape (B, T, H).
        attention_mask:    Real-vs-padding mask, shape (B, T) (1 = real token).
        strategy:          One of 'max', 'mean', or 'last':
                             - 'max'  : elementwise max over real tokens
                                        (default; encoder-style, see
                                        `masked_max_pool`).
                             - 'mean' : average over real tokens
                                        (`masked_mean_pool`).
                             - 'last' : hidden state of the last real token
                                        (`masked_last_token_pool`; the natural
                                        choice for causal decoder models).

    Returns:
        Pooled embeddings, shape (B, H).

    Raises:
        ValueError: If `strategy` is not a key of `POOLING_FUNCTIONS`.
    """
    if strategy not in POOLING_FUNCTIONS:
        raise ValueError(f"Unknown pooling strategy {strategy!r}; "
                         f"expected one of {sorted(POOLING_FUNCTIONS)}.")
    return POOLING_FUNCTIONS[strategy](last_hidden_state, attention_mask)


@torch.no_grad() ##no tracking of gradients; saves memory
def embed_texts(texts: list[str], model_name: str,
                batch_size: int = 16, max_length: int = 512,
                device: Optional[str] = None) -> np.ndarray:
    """
    Run a HuggingFace encoder over a list of texts and return max-pooled
    document embeddings.

    Used by build_embedding_index when no `question_col` is supplied. Texts
    are tokenized in batches with truncation (no overflow handling), so any
    text longer than `max_length` tokens is silently cut off — use
    `embed_with_question` if you need chunked / overlapping behavior.

    Args:
        texts:      List of N strings to embed.
        model_name: HuggingFace model identifier (e.g.
                    'Simonlee711/Clinical_ModernBERT'). Loaded via AutoModel
                    + AutoTokenizer.
        batch_size: How many texts to forward through the encoder at once.
                    Memory/throughput trade-off; 16 is a safe CPU default,
                    32-64 typical on a single GPU.
        max_length: Maximum token length per input (longer texts truncated).
        device:     Torch device string ('cuda' / 'cpu'). Default: auto-pick
                    cuda if available else cpu.

    Returns:
        (N, hidden_size) float32 NumPy array of pooled embeddings, in the
        same row order as the input `texts`.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[embed] device={device}  model={model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, 
                                      attn_implementation="eager").to(device).eval()
    chunks = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        out = model(**enc)
        pooled = masked_max_pool(out.last_hidden_state, enc.attention_mask)
        chunks.append(pooled.float().cpu().numpy())
    embeddings = np.concatenate(chunks, axis=0)
    print(f"[embed] done. shape={embeddings.shape}")
    return embeddings


@torch.no_grad()
def embed_with_question(questions: list[str], texts: list[str],
                        model_name: str, batch_size: int = 16,
                        max_length: int = 512, stride: int = 0,
                        chunk_aggregation: str = "max",
                        device: Optional[str] = None) -> np.ndarray:
    """
    Question-conditioned, chunked document embeddings.

    For each (question, text) pair, tokenize as a SEQUENCE PAIR — format
    is roughly `[CLS] question [SEP] text_chunk [SEP]` — and use
    `truncation="only_second"` so the question is never truncated. If the
    text exceeds `max_length` after the question, it is split into multiple
    chunks (overlapping by `stride` tokens), with the question prepended
    to each chunk. The pooled per-chunk embeddings are then combined per
    document via `chunk_aggregation`.

    Use this instead of `embed_texts` whenever you want the BERT encoder
    to see the question alongside the document, OR when documents are long
    enough that you need real chunking rather than silent truncation.

    Args:
        questions:         List of N question strings. Each question must
                           be non-empty and constant across chunks of its
                           associated text.
        texts:             List of N document strings, same length as
                           `questions`.
        model_name:        HuggingFace model identifier.
        batch_size:        Chunks forwarded through the encoder per batch.
                           Note: chunks (not documents) are batched, so a
                           document that splits into 4 chunks contributes
                           4 forward-pass rows.
        max_length:        Maximum token length per chunk (question + text
                           chunk + special tokens, total).
        stride:            Token overlap between adjacent chunks of the
                           same text. 0 = no overlap; higher = chunks
                           share context, useful when relevant content may
                           straddle a chunk boundary.
        chunk_aggregation: How to combine chunk-level embeddings into one
                           per-document embedding: 'max' (elementwise max
                           across chunks; composes naturally with the
                           within-chunk max pool) or 'mean' (smoother,
                           less sensitive to a single salient chunk).
        device:            Torch device.

    Returns:
        (N, hidden_size) float32 NumPy array of per-document embeddings,
        aligned with the input row order.

    Raises:
        ValueError: If `len(questions) != len(texts)` or `chunk_aggregation`
                    is neither 'max' nor 'mean'.
    """
    if len(questions) != len(texts):
        raise ValueError("questions and texts must have the same length")
    if chunk_aggregation not in {"max", "mean"}:
        raise ValueError("chunk_aggregation must be 'max' or 'mean'")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, 
                                      attn_implementation="eager").to(device).eval()

    all_input_ids, all_attention_mask, doc_id_per_chunk = [], [], []
    for doc_idx, (q, t) in enumerate(zip(questions, texts)):
        enc = tokenizer(q, t, max_length=max_length, truncation="only_second",
                        return_overflowing_tokens=True, stride=stride,
                        padding="max_length", return_tensors="pt")
        n_chunks = enc["input_ids"].size(0)
        all_input_ids.append(enc["input_ids"])
        all_attention_mask.append(enc["attention_mask"])
        doc_id_per_chunk.extend([doc_idx] * n_chunks)

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_mask, dim=0)
    chunk_to_doc = np.asarray(doc_id_per_chunk)

    chunk_embs = []
    for start in tqdm(range(0, len(input_ids), batch_size), desc="Embedding chunks"):
        batch_ids = input_ids[start:start + batch_size].to(device)
        batch_mask = attention_mask[start:start + batch_size].to(device)
        out = model(input_ids=batch_ids, attention_mask=batch_mask)
        pooled = masked_max_pool(out.last_hidden_state, batch_mask)
        chunk_embs.append(pooled.float().cpu().numpy())
    chunk_embs = np.concatenate(chunk_embs, axis=0)

    hidden = chunk_embs.shape[1]
    doc_embs = np.zeros((len(questions), hidden), dtype=np.float32)
    for doc_idx in range(len(questions)):
        sel = chunk_to_doc == doc_idx
        if not sel.any():
            continue
        doc_chunks = chunk_embs[sel]
        if chunk_aggregation == "max":
            doc_embs[doc_idx] = doc_chunks.max(axis=0)
        else:
            doc_embs[doc_idx] = doc_chunks.mean(axis=0)
    print(f"[embed_q] done. shape={doc_embs.shape}")
    return doc_embs


# -----------------------------------------------------------------------------
# Generative (decoder-only) model embedding path
# -----------------------------------------------------------------------------

def _resolve_torch_dtype(torch_dtype: Optional[Union[str, "torch.dtype"]],
                         device: str) -> "torch.dtype":
    """
    Resolve a user-facing dtype spec into a concrete torch.dtype.

    Large generative checkpoints (e.g. HRM-Text-1B) ship in bfloat16, so the
    sensible default is bfloat16 on CUDA and float32 on CPU (bf16 matmul
    support on CPU is patchy and slow).

    Args:
        torch_dtype: One of None, 'auto', a string ('float32', 'float16',
                     'bfloat16'), or an actual torch.dtype. None / 'auto'
                     selects bfloat16 on CUDA else float32.
        device:      Resolved device string ('cuda' / 'cpu').

    Returns:
        A concrete torch.dtype.

    Raises:
        ValueError: If `torch_dtype` is an unrecognized string.
    """
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if torch_dtype in (None, "auto"):
        return torch.bfloat16 if device == "cuda" else torch.float32
    name_to_dtype = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    if torch_dtype not in name_to_dtype:
        raise ValueError(f"Unrecognized torch_dtype {torch_dtype!r}; expected one "
                         f"of {sorted(name_to_dtype)} or a torch.dtype.")
    return name_to_dtype[torch_dtype]


def _extract_last_hidden_state(model_output) -> torch.Tensor:
    """
    Pull the final-layer token hidden states out of a HuggingFace model output,
    tolerating the several shapes that custom / generative models use.

    A bidirectional encoder (AutoModel) exposes `.last_hidden_state` directly.
    A causal LM called with `output_hidden_states=True` exposes
    `.hidden_states`, a tuple of (num_layers + 1) tensors whose LAST element is
    the final-layer output. Some custom architectures (HRM-Text is recurrent
    and ships as `trust_remote_code`) may expose only one of these, so we probe
    in priority order.

    Args:
        model_output: The object returned by a HuggingFace model forward pass.

    Returns:
        Tensor of shape (B, T, H): final-layer per-token hidden states.

    Raises:
        AttributeError: If neither `last_hidden_state` nor a non-empty
                        `hidden_states` is present. The message suggests how to
                        adapt for an unusual custom model.
    """
    hs = getattr(model_output, "hidden_states", None)
    if hs is not None and len(hs) > 0:
        return hs[-1]
    last = getattr(model_output, "last_hidden_state", None)
    if last is not None:
        return last
    raise AttributeError(
        "Could not find token hidden states on the model output (no "
        "`hidden_states` from output_hidden_states=True, and no "
        "`last_hidden_state`). For an unusual custom architecture you may need "
        "to load the base module via AutoModel and read its hidden states "
        "directly, or register a forward hook on the final norm layer.")


@torch.no_grad()
def embed_texts_generative(texts: list[str], model_name: str,
                           batch_size: int = 8, max_length: int = 512,
                           pooling: str = "last",
                           trust_remote_code: bool = True,
                           prefix_lm_bidirectional: bool = False,
                           torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
                           device: Optional[str] = None) -> np.ndarray:
    """
    Run an open-weight DECODER-ONLY / generative pretrained model over a list
    of texts and return pooled document embeddings.

    This is the generative-model counterpart to `embed_texts` (which targets
    bidirectional BERT-family encoders). It loads the model with
    `AutoModelForCausalLM`, runs a single forward pass with
    `output_hidden_states=True`, takes the final-layer per-token hidden states,
    and pools them into one vector per document.

    Pooling choice matters more than it does for encoders:
      * 'last' (default) returns the hidden state of the last real token, which
        is the only position that has attended over the whole input under
        causal attention. This is the standard for decoder-LM embeddings
        (cpt-text, E5-Mistral).
      * 'mean' / 'max' average / max over all real tokens. These are only
        well-motivated when attention is bidirectional over the input -- e.g.
        a PrefixLM model run with the entire input marked as the prefix
        (see `prefix_lm_bidirectional`), or an LLM2Vec-style conversion.

    Example (HRM-Text-1B, https://huggingface.co/sapientinc/HRM-Text-1B):
        emb = embed_texts_generative(
            texts, model_name="sapientinc/HRM-Text-1B",
            pooling="last", trust_remote_code=True,
            prefix_lm_bidirectional=True)   # whole input as a bidirectional prefix

    Args:
        texts:                   List of N strings to embed.
        model_name:              HuggingFace model identifier for a generative
                                 (causal-LM) checkpoint, e.g.
                                 'sapientinc/HRM-Text-1B'. Loaded via
                                 AutoModelForCausalLM + AutoTokenizer.
        batch_size:              Texts forwarded per batch. Generative models
                                 are large, so the default (8) is lower than the
                                 encoder default; tune to GPU memory.
        max_length:              Maximum token length per input (longer texts
                                 are truncated).
        pooling:                 'last' (default), 'mean', or 'max'. See above
                                 and `pool_token_embeddings`.
        trust_remote_code:       Passed to from_pretrained. Required True for
                                 custom architectures such as HRM-Text that ship
                                 their modeling code on the Hub.
        prefix_lm_bidirectional: If True, set token_type_ids = 1 on every input
                                 token. For a PrefixLM checkpoint (HRM-Text) this
                                 marks the whole input as a bidirectional prefix
                                 -- matching its training-time prefix forward and
                                 letting every token attend to every other,
                                 which makes 'mean'/'max' pooling meaningful.
                                 Ignored by models that do not consume
                                 token_type_ids.
        torch_dtype:             None/'auto' (bf16 on CUDA, fp32 on CPU),
                                 'float32'/'float16'/'bfloat16', or a torch.dtype.
        device:                  'cuda' / 'cpu'. Default: cuda if available.

    Returns:
        (N, hidden_size) float32 NumPy array of pooled embeddings, in the same
        row order as `texts`. `hidden_size` is the model's residual-stream
        width (typically larger than a BERT encoder's, so downstream MLP input
        dims differ from the Clinical ModernBERT path).

    Raises:
        ValueError:     If `pooling` is not a valid strategy.
        AttributeError: If the model output exposes no usable hidden states
                        (see `_extract_last_hidden_state`).

    Notes / caveats:
        * HRM-Text is a recurrent, custom architecture. This function assumes it
          honours `output_hidden_states=True` (the standard PreTrainedModel
          contract). If a future revision does not, read its base module's
          hidden states directly or hook the final norm; the pooling functions
          here still apply unchanged.
        * If the tokenizer has no pad token (common for decoder LMs), the EOS
          token is used for padding; padded positions still get attention_mask
          0, so pooling correctly ignores them.

    References:
        Neelakantan, A. et al. (2022). Text and Code Embeddings by Contrastive
        Pre-Training. arXiv:2201.10005.
        Wang, L. et al. (2024). Improving Text Embeddings with Large Language
        Models. arXiv:2401.00368.
        BehnamGhader, P. et al. (2024). LLM2Vec: Large Language Models Are
        Secretly Powerful Text Encoders. arXiv:2404.05961.
        Sapient Intelligence (2026). HRM-Text-1B model card.
        https://huggingface.co/sapientinc/HRM-Text-1B
    """
    if pooling not in POOLING_FUNCTIONS:
        raise ValueError(f"Unknown pooling strategy {pooling!r}; "
                         f"expected one of {sorted(POOLING_FUNCTIONS)}.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _resolve_torch_dtype(torch_dtype, device)
    print(f"[embed_gen] device={device}  dtype={dtype}  model={model_name}  "
          f"pooling={pooling}  prefix_lm_bidirectional={prefix_lm_bidirectional}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        # Decoder LMs frequently lack a pad token; reuse EOS for padding.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=trust_remote_code,
        attn_implementation="eager",
        torch_dtype=dtype).to(device).eval()

    chunks = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding (gen)"):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt").to(device)
        if prefix_lm_bidirectional:
            # Mark every input token as part of the bidirectional prefix.
            enc["token_type_ids"] = torch.ones_like(enc["input_ids"])
        out = model(**enc, output_hidden_states=True)
        last_hidden = _extract_last_hidden_state(out)
        pooled = pool_token_embeddings(last_hidden, enc["attention_mask"],
                                       strategy=pooling)
        chunks.append(pooled.float().cpu().numpy())
    embeddings = np.concatenate(chunks, axis=0)
    print(f"[embed_gen] done. shape={embeddings.shape}")
    return embeddings


def embed_documents(texts: list[str], model_name: str,
                    model_kind: str = "encoder",
                    questions: Optional[list[str]] = None,
                    batch_size: int = 16, max_length: int = 512,
                    pooling: Optional[str] = None,
                    stride: int = 0, chunk_aggregation: str = "max",
                    trust_remote_code: bool = False,
                    prefix_lm_bidirectional: bool = False,
                    torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
                    device: Optional[str] = None) -> np.ndarray:
    """
    Unified embedding entry point that routes to the encoder or generative path.

    Lets the rest of the pipeline embed text without caring whether the backbone
    is a BERT-family encoder or a decoder-only generative model. The Clinical
    ModernBERT behaviour is preserved exactly when `model_kind='encoder'`.

    Routing:
      * model_kind='encoder'  (default)
            - with `questions`    -> `embed_with_question` (chunked, sequence-pair)
            - without `questions` -> `embed_texts`
        Pooling is the encoder's built-in masked-max pool; `pooling` is ignored
        here except that question-mode honours `chunk_aggregation`.
      * model_kind='generative'
            -> `embed_texts_generative`. If `questions` is given, each question
               is prepended to its text ("question\\n\\ntext"), since decoder
               models have no [CLS]/[SEP] sequence-pair convention.
        Pooling defaults to 'last' (override via `pooling`).

    Args:
        texts:                   List of N document strings.
        model_name:              HuggingFace model id.
        model_kind:              'encoder' (BERT-family) or 'generative'
                                 (decoder-only causal LM).
        questions:               Optional list of N question strings for
                                 question-conditioned embedding. Must match
                                 `texts` in length when provided.
        batch_size:              Forward-pass batch size.
        max_length:              Max token length per input/chunk.
        pooling:                 Pooling strategy. None -> 'max' for encoder,
                                 'last' for generative. 'max'/'mean'/'last'.
        stride:                  (encoder + questions only) chunk token overlap.
        chunk_aggregation:       (encoder + questions only) 'max' or 'mean'
                                 across chunks.
        trust_remote_code:       (generative) allow Hub-provided modeling code.
        prefix_lm_bidirectional: (generative) mark whole input as a bidirectional
                                 PrefixLM prefix (e.g. for HRM-Text).
        torch_dtype:             (generative) model dtype spec.
        device:                  Torch device. Default: cuda if available.

    Returns:
        (N, hidden_size) float32 NumPy array, aligned with `texts`.

    Raises:
        ValueError: If `model_kind` is not 'encoder' or 'generative', or if
                    `questions` is given but its length differs from `texts`.
    """
    if model_kind not in ("encoder", "generative"):
        raise ValueError(f"model_kind must be 'encoder' or 'generative', "
                         f"got {model_kind!r}.")
    if questions is not None and len(questions) != len(texts):
        raise ValueError("questions and texts must have the same length.")

    if model_kind == "encoder":
        if questions is not None:
            return embed_with_question(
                questions=questions, texts=texts, model_name=model_name,
                batch_size=batch_size, max_length=max_length, stride=stride,
                chunk_aggregation=chunk_aggregation, device=device)
        return embed_texts(texts, model_name=model_name, batch_size=batch_size,
                           max_length=max_length, device=device)

    # generative
    gen_pooling = pooling or "last"
    gen_texts = texts
    if questions is not None:
        gen_texts = [f"{q}\n\n{t}" for q, t in zip(questions, texts)]
    return embed_texts_generative(
        gen_texts, model_name=model_name, batch_size=batch_size,
        max_length=max_length, pooling=gen_pooling,
        trust_remote_code=trust_remote_code,
        prefix_lm_bidirectional=prefix_lm_bidirectional,
        torch_dtype=torch_dtype, device=device)


# =============================================================================
# Long-form -> per-target data  (with optional selection_text_col)
# =============================================================================

def build_embedding_index(long_df: pd.DataFrame,
                          id_col: str,
                          text_col: str,
                          model_name: str,
                          selection_text_col: Optional[str] = None,
                          question_col: Optional[str] = None,
                          batch_size: int = 16,
                          max_length: int = 512,
                          stride: int = 0,
                          chunk_aggregation: str = "max",
                          model_kind: str = "encoder",
                          pooling: Optional[str] = None,
                          trust_remote_code: bool = False,
                          prefix_lm_bidirectional: bool = False,
                          torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
                          device: Optional[str] = None) -> dict:
    """
    Deduplicate `long_df` by `id_col`, embed each unique text exactly once,
    and (optionally) embed a separate selection-equation text column too.

    This is Stage 1 of the pipeline. The output is consumed by
    `long_to_per_target` to build per-head training data.

    Args:
        long_df:             Long-form DataFrame; one row per
                             (id, code, value) observation. Texts must be
                             constant within an id.
        id_col:              Patient/note identifier column. Used for
                             deduplication and per-head selection masks.
        text_col:            Column containing the clinical text whose
                             embedding feeds the OUTCOME quantile model.
        model_name:          HuggingFace encoder identifier (e.g.
                             'Simonlee711/Clinical_ModernBERT').
        selection_text_col:  (Optional) Column containing process /
                             scheduling text embedded SEPARATELY for the
                             selection equation. Provides a true
                             exclusion restriction in Heckman. If None
                             (default), the selection equation reuses
                             the outcome embeddings. Required to be
                             constant per id and non-empty for at least
                             most ids when supplied.
        question_col:        (Optional) Column whose value is prepended
                             to every chunk during tokenization (see
                             `embed_with_question` for details).
        batch_size:          Forward-pass batch size (chunks per batch
                             when question conditioning is on).
        max_length:          Maximum token length per chunk.
        stride:              Token overlap between adjacent chunks of
                             the same text (only used with `question_col`).
        chunk_aggregation:   'max' or 'mean' — how to pool chunk
                             embeddings into one per-id embedding.
        model_kind:          'encoder' (default; BERT-family, e.g. Clinical
                             ModernBERT) or 'generative' (decoder-only causal
                             LM, e.g. HRM-Text-1B). Routes outcome AND
                             selection text through the matching backbone.
        pooling:             Token-pooling strategy. None -> 'max' for encoder,
                             'last' for generative. One of 'max'/'mean'/'last'.
        trust_remote_code:   (generative) allow Hub-provided modeling code
                             (required for custom architectures like HRM-Text).
        prefix_lm_bidirectional: (generative) mark the whole input as a
                             bidirectional PrefixLM prefix; see
                             `embed_texts_generative`.
        torch_dtype:         (generative) model dtype spec (None/'auto',
                             'float32'/'float16'/'bfloat16', or a torch.dtype).
        device:              Torch device. Defaults to cuda if available.

    Returns:
        Dict with four keys:
            'embeddings':           (n_unique_ids, hidden) float32 outcome
                                    embeddings, one row per unique id.
            'selection_embeddings': (n_unique_ids, hidden) float32 array
                                    aligned with `embeddings` if
                                    `selection_text_col` was supplied;
                                    None otherwise.
            'id_to_row':            dict mapping id value -> row index
                                    in `embeddings`. Use this to look
                                    up an id's embedding row.
            'id_order':             list of ids in the order they appear
                                    in `embeddings`.

    Raises:
        ValueError: If `selection_text_col` is named but absent from
                    `long_df`, or if the question column is empty for
                    any id.

    Warnings:
        Prints to stdout (not raised) when the same id has multiple
        distinct text values, selection texts, or questions. In those
        cases the first encountered is used. Make ids more granular
        upstream (e.g. include encounter id) if every variant should
        be embedded.
    """
    # --- consistency checks on text_col ---
    text_per_id = long_df.groupby(id_col)[text_col].nunique()
    inconsistent = text_per_id[text_per_id > 1]
    if len(inconsistent) > 0:
        print(f"[warn] {len(inconsistent)} ids have multiple distinct text_col values; "
              f"using the first per id.")

    if selection_text_col is not None:
        if selection_text_col not in long_df.columns:
            raise ValueError(f"selection_text_col {selection_text_col!r} not in DataFrame")
        sel_per_id = long_df.groupby(id_col)[selection_text_col].nunique(dropna=False)
        bad_sel = sel_per_id[sel_per_id > 1]
        if len(bad_sel) > 0:
            print(f"[warn] {len(bad_sel)} ids have multiple distinct selection_text_col "
                  f"values; using the first per id.")

    if question_col is not None:
        if question_col not in long_df.columns:
            raise ValueError(f"question_col {question_col!r} not in DataFrame")
        q_per_id = long_df.groupby(id_col)[question_col].nunique(dropna=False)
        inconsistent_q = q_per_id[q_per_id > 1]
        if len(inconsistent_q) > 0:
            print(f"[warn] {len(inconsistent_q)} ids have multiple distinct questions; "
                  f"using the first per id.")

    text_df = long_df.drop_duplicates(subset=[id_col]).reset_index(drop=True)
    ids = text_df[id_col].tolist()
    texts = text_df[text_col].astype(str).tolist()

    # --- embed outcome text ---
    if question_col is not None:
        questions_raw = text_df[question_col]
        bad = questions_raw.isna() | (questions_raw.astype(str).str.strip() == "")
        if bad.any():
            bad_ids = text_df.loc[bad, id_col].head(5).tolist()
            raise ValueError(f"{int(bad.sum())} ids have empty/NaN question (e.g. {bad_ids}).")
        questions = questions_raw.astype(str).tolist()
        print(f"[embed] outcome ({model_kind}): {len(long_df)} long-form rows -> "
              f"{len(texts)} unique ids to embed WITH question conditioning")
    else:
        questions = None
        print(f"[embed] outcome ({model_kind}): {len(long_df)} long-form rows -> "
              f"{len(texts)} unique ids to embed")
    embeddings = embed_documents(
        texts, model_name=model_name, model_kind=model_kind,
        questions=questions, batch_size=batch_size, max_length=max_length,
        pooling=pooling, stride=stride, chunk_aggregation=chunk_aggregation,
        trust_remote_code=trust_remote_code,
        prefix_lm_bidirectional=prefix_lm_bidirectional,
        torch_dtype=torch_dtype, device=device)

    # --- embed selection text if requested ---
    selection_embeddings = None
    if selection_text_col is not None:
        sel_texts = text_df[selection_text_col].astype(str).fillna("").tolist()
        # Validate that not every selection text is empty
        n_nonempty = sum(1 for t in sel_texts if t.strip())
        if n_nonempty == 0:
            raise ValueError(f"All values in selection_text_col {selection_text_col!r} are "
                             f"empty. With --selection-text-column the selection text must "
                             f"contain actual content for at least most ids.")
        elif n_nonempty < 0.5 * len(sel_texts):
            print(f"[warn] selection_text_col has {n_nonempty} non-empty / {len(sel_texts)} "
                  f"total. Empty selection texts will all embed to the same vector.")
        print(f"[embed] selection ({model_kind}): embedding {len(sel_texts)} "
              f"selection texts separately")
        selection_embeddings = embed_documents(
            sel_texts, model_name=model_name, model_kind=model_kind,
            questions=None, batch_size=batch_size, max_length=max_length,
            pooling=pooling, trust_remote_code=trust_remote_code,
            prefix_lm_bidirectional=prefix_lm_bidirectional,
            torch_dtype=torch_dtype, device=device)

    id_to_row = {id_val: i for i, id_val in enumerate(ids)}
    return {
        "embeddings": embeddings,
        "selection_embeddings": selection_embeddings,
        "id_to_row": id_to_row,
        "id_order": ids,
    }


def long_to_per_target(long_df: pd.DataFrame,
                       embeddings: np.ndarray,
                       id_to_row: dict,
                       id_col: str,
                       code_col: str,
                       value_col: str,
                       include_selection: bool = False,
                       selection_embeddings: Optional[np.ndarray] = None
                       ) -> Union[
                           list[tuple[str, np.ndarray, np.ndarray]],
                           list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                       ]:
    """
    Group `long_df` by `code_col` and produce the per-target training data
    tuples consumed by `train_all_heads`.

    The tuple shape changes based on `include_selection`:

      include_selection=False  →  3-tuples (code, X, y)
      include_selection=True   →  5-tuples (code, X_outcome, X_selection, y, S)

    See "Returns" below for details on each shape.

    Args:
        long_df:              Long-form observations DataFrame (one row per
                              (id, code, value)). NaN values in `value_col`
                              are dropped automatically.
        embeddings:           (n_unique_ids, hidden) outcome embeddings,
                              typically from `build_embedding_index`.
        id_to_row:            Dict mapping id value -> row index in
                              `embeddings`. Rows whose id is not in this
                              dict are dropped with a warning.
        id_col:               ID column name in `long_df`.
        code_col:             Column naming the target for each row.
                              Groups are produced by sorting these values
                              for deterministic head numbering.
        value_col:            Column with the observed numeric value.
        include_selection:    If True, return 5-tuples with a per-id
                              selection mask spanning the FULL embedded
                              population (required for the Heckman path).
                              If False, return 3-tuples with only the
                              observed rows for each code (vanilla CQR).
        selection_embeddings: (Optional) (n_unique_ids, hidden) array
                              aligned with `embeddings`. When non-None
                              AND `include_selection=True`, this is what
                              the selection model is fed for each id —
                              providing the EXCLUSION RESTRICTION. When
                              None, `embeddings` is used for both
                              equations.

    Returns:
        A list with one entry per unique code (sorted), each entry shape
        depending on `include_selection`:

        If include_selection=False:
            (name: str,
             X: (n_obs, hidden) float32 — outcome embeddings for the rows
                                          where this code was observed,
             y: (n_obs,) — observed numeric values).

        If include_selection=True:
            (name: str,
             X_outcome:   (n_unique_ids, hidden) outcome embeddings for
                          ALL embedded ids (same array reused across codes
                          — no memory duplication),
             X_selection: (n_unique_ids, hidden) selection embeddings;
                          equal-by-reference to X_outcome when
                          `selection_embeddings` is None, otherwise the
                          supplied selection array,
             y_universe:  (n_unique_ids,) observed value where this code's
                          row exists, NaN otherwise,
             S:           (n_unique_ids,) float32 selection indicator —
                          1.0 where the lab was observed, 0.0 otherwise).

    Raises:
        ValueError: If `selection_embeddings` is supplied with a row count
                    that doesn't match `embeddings.shape[0]`.
    """
    df = long_df[[id_col, code_col, value_col]].copy()
    df["_emb_idx"] = df[id_col].map(id_to_row)
    n_before = len(df)
    df = df.dropna(subset=["_emb_idx"])
    if len(df) < n_before:
        print(f"[long->targets] dropped {n_before - len(df)} rows with ids missing from index")
    df["_emb_idx"] = df["_emb_idx"].astype(int)
    df = df.dropna(subset=[value_col])

    n_universe = embeddings.shape[0]
    X_selection = selection_embeddings if selection_embeddings is not None else embeddings
    if include_selection and X_selection.shape[0] != n_universe:
        raise ValueError(
            f"selection_embeddings has {X_selection.shape[0]} rows but outcome "
            f"embeddings have {n_universe}. They must share id ordering.")

    per_target: list = []
    for code, group in df.groupby(code_col, sort=True):
        emb_idx = group["_emb_idx"].to_numpy()
        y_observed = group[value_col].to_numpy()
        if include_selection:
            y_full = np.full(n_universe, np.nan, dtype=np.float64)
            S = np.zeros(n_universe, dtype=np.float32)
            y_full[emb_idx] = y_observed
            S[emb_idx] = 1.0
            per_target.append((str(code), embeddings, X_selection, y_full, S))
        else:
            per_target.append((str(code), embeddings[emb_idx], y_observed))

    msg = ("with selection indicators (separate selection embeddings)"
           if include_selection and selection_embeddings is not None
           else ("with selection indicators" if include_selection else ""))
    print(f"[long->targets] {len(per_target)} unique codes ready for training {msg}")
    return per_target


# =============================================================================
# Stage 2 — Models, losses, and selection helpers
# =============================================================================

def _quantile_levels_from_alpha(alpha: float) -> tuple[float, float, float]:
    """
    Map a target miscoverage rate `alpha` to the three quantile levels used
    by the CQR heads: lower, median, upper.

    Args:
        alpha: Miscoverage rate in (0, 1). e.g. 0.1 -> 90% prediction interval.

    Returns:
        (alpha/2, 0.5, 1 - alpha/2) — the lower-tail, median, and upper-tail
        quantile levels the outcome MLP is trained to predict simultaneously
        via pinball loss.

    Raises:
        ValueError: If alpha is not strictly in (0, 1).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    return (alpha / 2.0, 0.5, 1.0 - alpha / 2.0)


def _build_mlp(in_dim: int, out_dim: int, hidden: int, depth: int,
               dropout: float) -> nn.Sequential:
    """
    Build a constant-width MLP:
        Linear(in_dim, hidden) -> GELU -> Dropout
        [Linear(hidden, hidden) -> GELU -> Dropout] x (depth - 1)
        Linear(hidden, out_dim)

    Used by FFNNQuantileRegressor, SelectionModel, and (for its trunk)
    SelectionModelReparam.

    Args:
        in_dim:  Input dimension.
        out_dim: Output dimension.
        hidden:  Width of every hidden layer.
        depth:   Number of hidden layers (each Linear -> GELU -> Dropout).
                 Must be >= 1.
        dropout: Dropout probability applied after each GELU.

    Returns:
        An `nn.Sequential` module with the described layer pattern.

    Raises:
        ValueError: If depth < 1.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1; got {depth}")
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(depth):
        layers += [nn.Linear(prev, hidden), nn.GELU(), nn.Dropout(dropout)]
        prev = hidden
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class FFNNQuantileRegressor(nn.Module):
    """
    Per-target outcome model: predicts `n_quantiles` quantile estimates from
    one input vector, trained with pinball loss.

    For CQR with alpha=0.1 (90% PI), `n_quantiles=3` produces predictions
    at levels (0.05, 0.5, 0.95) simultaneously — lower, median, upper.

    Predictions are emitted in STANDARDIZED y-space; the calling code stores
    the per-head y_mean, y_std and de-standardizes at inference. Outputs
    are also unsorted; the calling code applies `np.sort(..., axis=1)`
    after de-standardization (the "rearrangement" post-hoc fix for
    quantile crossing).
    """
    def __init__(self, in_dim: int = 768, n_quantiles: int = 3,
                 hidden: int = 256, depth: int = 2, dropout: float = 0.1):
        """
        Args:
            in_dim:      Input dimension. For the Heckman path this is
                         hidden_size + 1 (BERT embedding plus IMR feature);
                         for plain CQR it is hidden_size.
            n_quantiles: Number of quantile levels to predict.
            hidden:      Width of every hidden layer.
            depth:       Number of hidden layers.
            dropout:     Dropout probability.
        """
        super().__init__()
        self.n_quantiles = n_quantiles
        self.hidden = hidden
        self.depth = depth
        self.net = _build_mlp(in_dim, n_quantiles, hidden, depth, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, in_dim).

        Returns:
            Tensor of shape (B, n_quantiles) of standardized quantile
            predictions, one column per level configured at construction.
        """
        return self.net(x)


class SelectionModel(nn.Module):
    """
    Plain (homoscedastic) probit selection model.

    Outputs the probit INDEX z = x'gamma; convert to a selection probability
    via Phi(z) (standard normal CDF) at the call site if needed. Trained
    with the probit negative log-likelihood `probit_nll`. The inverse Mills
    ratio at z is what gets added as a feature to the outcome equation in
    the Heckman two-step.

    Use this when you want classical Heckman behavior. For the
    heteroscedastic variant — which provides Klein-Vella identification
    through learned noise scale — use `SelectionModelReparam`.
    """
    def __init__(self, in_dim: int = 768, hidden: int = 128,
                 depth: int = 2, dropout: float = 0.1):
        """
        Args:
            in_dim:  Input dimension (size of the embedding fed to the
                     selection equation; this may differ from the outcome
                     model's in_dim if a separate selection text column
                     was used).
            hidden:  Width of every hidden layer.
            depth:   Number of hidden layers.
            dropout: Dropout probability.
        """
        super().__init__()
        self.hidden = hidden
        self.depth = depth
        self.net = _build_mlp(in_dim, 1, hidden, depth, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, in_dim).

        Returns:
            Tensor of shape (B,) — the probit index z for each row.
            Selection probability is Phi(z); IMR is phi(z)/Phi(z).
        """
        return self.net(x).squeeze(-1)


class SelectionModelReparam(nn.Module):
    """
    HETEROSCEDASTIC probit selection model using the reparameterization trick.

    Architecture:
        x -> shared trunk -> { mu_head -> mu(x),  sigma_head -> sigma(x) }
        At training:  z = mu(x) + sigma(x) * eta,   eta ~ N(0, 1)
        At inference: returns mu(x) (no stochasticity).

    Sigma is parameterized as softplus(raw) + min_sigma so it stays strictly
    positive. The min_sigma floor keeps gradients well-conditioned when the
    raw output is very negative.

    Why this helps identification:
        Klein & Vella (2010) show that when sigma(x) varies with x in a way
        that's distinguishable from the functional form of the outcome
        equation, the Heckman model is identified WITHOUT an exclusion
        restriction. Identification comes from the heteroscedasticity itself.

    The reparameterization trick (sampling eta as a fixed standard-normal
    and pushing gradients through mu and sigma) is what makes this
    stochastic latent index trainable via standard backprop.
    """
    def __init__(self, in_dim: int = 768, hidden: int = 128,
                 depth: int = 2, dropout: float = 0.1, min_sigma: float = 0.1):
        """
        Args:
            in_dim:    Input dimension.
            hidden:    Width of every hidden layer in the shared trunk.
            depth:     Number of hidden layers in the trunk (must be >= 1).
            dropout:   Dropout probability.
            min_sigma: Floor on the learned sigma(x). Prevents sigma from
                       collapsing to zero (which would make the probit
                       degenerate and gradients ill-conditioned). Set
                       lower (e.g. 0.01) for more flexibility if training
                       is well-behaved.
        """
        super().__init__()
        self.hidden = hidden
        self.depth = depth
        self.min_sigma = min_sigma
        if depth < 1:
            raise ValueError(f"depth must be >= 1; got {depth}")
        trunk_layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(depth):
            trunk_layers += [nn.Linear(prev, hidden), nn.GELU(), nn.Dropout(dropout)]
            prev = hidden
        self.trunk = nn.Sequential(*trunk_layers)
        self.mu_head = nn.Linear(prev, 1)
        self.raw_sigma_head = nn.Linear(prev, 1)

    def get_mu_sigma(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the latent mean and stddev of the probit index given x.

        Args:
            x: Input tensor of shape (B, in_dim).

        Returns:
            (mu, sigma):
                mu:    (B,) — center of the probit index distribution.
                sigma: (B,) — heteroscedastic stddev (>= min_sigma).
        """
        h = self.trunk(x)
        mu = self.mu_head(h).squeeze(-1)
        sigma = F.softplus(self.raw_sigma_head(h).squeeze(-1)) + self.min_sigma
        return mu, sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with branching by train/eval mode.

        Args:
            x: Input tensor of shape (B, in_dim).

        Returns:
            (B,) probit index z.
                - In training mode: z = mu(x) + sigma(x) * eta with
                  eta ~ N(0,1) sampled fresh per call. Gradients flow
                  through mu and sigma via the reparameterization trick.
                - In eval mode: z = mu(x) deterministically (no eta
                  sampling), so val-set probit NLL is well-defined.

        For an explicit non-stochastic mu(x) regardless of mode, call
        `deterministic_index(x)` instead.
        """
        mu, sigma = self.get_mu_sigma(x)
        if self.training:
            eta = torch.randn_like(mu)
            return mu + sigma * eta
        else:
            return mu

    def deterministic_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return mu(x) only — used at inference time to compute the IMR
        feature deterministically regardless of model mode.

        Args:
            x: Input tensor of shape (B, in_dim).

        Returns:
            (B,) tensor — the deterministic probit index mu(x).
        """
        mu, _ = self.get_mu_sigma(x)
        return mu


def probit_nll(z: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """
    Probit negative log-likelihood for binary selection.

    For each row,
        - if S=1, the contribution is -log Phi(z)
        - if S=0, the contribution is -log Phi(-z) = -log(1 - Phi(z))
    Uses `torch.special.log_ndtr` for numerical stability in the tails
    (where naive log(Phi(z)) underflows).

    Args:
        z: Probit index, shape (B,). For SelectionModelReparam this is
           the sampled z = mu + sigma * eta during training, mu at eval.
        S: Binary selection indicator, shape (B,), values in {0.0, 1.0}.

    Returns:
        Scalar tensor — mean NLL across the batch.
    """
    log_p1 = torch.special.log_ndtr(z)
    log_p0 = torch.special.log_ndtr(-z)
    return -(S * log_p1 + (1.0 - S) * log_p0).mean()


def inverse_mills_ratio(z: Union[torch.Tensor, np.ndarray]
                        ) -> Union[torch.Tensor, np.ndarray]:
    """
    Inverse Mills ratio for SELECTED observations: lambda(z) = phi(z) / Phi(z).

    Used as an additional feature in the Heckman outcome equation. For
    very negative z (selection probability near 0), the denominator Phi(z)
    underflows; we clip the log_cdf below at log(1e-30) to keep the
    return finite.

    Args:
        z: Probit index. Accepts either torch.Tensor (no_grad context
           recommended) or np.ndarray. Output type matches input type.

    Returns:
        IMR values, same shape and type as input.
    """
    if isinstance(z, torch.Tensor):
        log_phi = -0.5 * z * z - 0.5 * math.log(2.0 * math.pi)
        log_Phi = torch.special.log_ndtr(z).clamp_min(math.log(1e-30))
        return torch.exp(log_phi - log_Phi)
    else:
        z_np = np.asarray(z, dtype=np.float64)
        log_phi = -0.5 * z_np * z_np - 0.5 * np.log(2.0 * np.pi)
        from math import erf, sqrt
        Phi = 0.5 * (1.0 + np.vectorize(erf)(z_np / sqrt(2.0)))
        Phi = np.clip(Phi, 1e-30, None)
        return np.exp(log_phi - np.log(Phi))


def pinball_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                 taus: torch.Tensor) -> torch.Tensor:
    """
    Multi-quantile pinball (a.k.a. quantile / check) loss — the standard
    objective for quantile regression.

    For one observation with target y and a quantile prediction q at level
    tau, the per-quantile pinball loss is
        L(q, y, tau) = max( tau*(y - q),  (tau - 1)*(y - q) )
    which is asymmetric: under-prediction is penalized at rate tau,
    over-prediction at rate (1-tau). Averaging across batch and quantile
    levels yields the multi-quantile loss returned here.

    Args:
        y_pred: Predicted quantiles, shape (B, n_quantiles).
        y_true: Target values, shape (B,).
        taus:   Quantile levels, shape (n_quantiles,), each in (0, 1).
                Order must match the column order of y_pred.

    Returns:
        Scalar tensor — mean pinball loss across batch and quantile levels.
    """
    diff = y_true.unsqueeze(-1) - y_pred
    return torch.max(taus * diff, (taus - 1.0) * diff).mean()


# -----------------------------------------------------------------------------
# Training: quantile regressor (with optional selection)
# -----------------------------------------------------------------------------

def train_single_head(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      taus: tuple[float, ...], device: str,
                      hidden: int = 256, depth: int = 2, dropout: float = 0.1,
                      lr: float = 1e-3, weight_decay: float = 1e-5,
                      batch_size: int = 256, max_epochs: int = 200,
                      patience: int = 10, min_delta: float = 1e-5
                      ) -> tuple[FFNNQuantileRegressor, dict, float, float, float]:
    """
    Train one FFNNQuantileRegressor with multi-quantile pinball loss,
    standardized targets, and early stopping.

    The target is standardized (z-scored) using train-set statistics before
    the loss is computed; the returned y_mean and y_std must be applied at
    inference to recover the original scale. After training, the model is
    reloaded with the best-val-loss checkpoint (not the final epoch's
    weights).

    Args:
        X_train, y_train: Training features (n_train, in_dim) and targets
                          (n_train,).
        X_val, y_val:     Validation features and targets; used for
                          early stopping and to report best_val_loss.
        taus:             Tuple of quantile levels (e.g. (0.05, 0.5, 0.95)).
                          Determines the number of output columns.
        device:           Torch device string.
        hidden:           Hidden width of the FFNNQuantileRegressor.
        depth:            Number of hidden layers.
        
        dropout:          Dropout probability.
        lr:               Adam learning rate.
        weight_decay:     Adam L2 regularization weight.
        batch_size:       Mini-batch size for training.
        max_epochs:       Hard cap on training epochs (early stopping
                          usually fires first).
        patience:         Epochs of no improvement before early stop.
        min_delta:        Smallest val_loss improvement counted as progress.

    Returns:
        5-tuple:
            model:         Trained FFNNQuantileRegressor (best-val state
                           reloaded).
            history:       Dict with two lists: 'train_loss' and 'val_loss',
                           one entry per epoch actually run.
            best_val_loss: Float — best validation pinball loss seen, in
                           STANDARDIZED y-space (not the original scale).
            y_mean:        Float — train-set mean of y (use to
                           de-standardize predictions at inference).
            y_std:         Float — train-set std of y (use to
                           de-standardize predictions at inference).
    """
    n_quantiles = len(taus)
    taus_t = torch.tensor(taus, dtype=torch.float32, device=device)

    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train) + 1e-8)
    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    Xt = torch.from_numpy(X_train).float()
    yt = torch.from_numpy(y_train_n).float()
    Xv = torch.from_numpy(X_val).float().to(device)
    yv = torch.from_numpy(y_val_n).float().to(device)

    train_loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size,
                              shuffle=True, drop_last=False)
    model = FFNNQuantileRegressor(in_dim=X_train.shape[1], n_quantiles=n_quantiles,
                                   hidden=hidden, depth=depth, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf"); best_state = None; epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = pinball_loss(model(xb), yb, taus_t)
            loss.backward(); opt.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(Xt)
        model.eval()
        with torch.no_grad():
            val_loss = pinball_loss(model(Xv), yv, taus_t).item()
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val_loss, y_mean, y_std


def train_selection_model(X_train: np.ndarray, S_train: np.ndarray,
                          X_val: np.ndarray, S_val: np.ndarray,
                          device: str,
                          hidden: int = 128, depth: int = 2,
                          reparam: bool = False, min_sigma: float = 0.1,
                          dropout: float = 0.1,
                          lr: float = 1e-3, weight_decay: float = 1e-5,
                          batch_size: int = 256, max_epochs: int = 100,
                          patience: int = 10, min_delta: float = 1e-5):
    """
    Train the probit selection model (the first stage of the Heckman two-step).

    Trained against `probit_nll` on a (X, S) dataset spanning the FULL
    population (both selected and unselected rows). After training, the
    model's IMR feature gets concatenated to the outcome embeddings of
    SELECTED rows for stage 2.

    Args:
        X_train, S_train: Training features (n_train, in_dim) and binary
                          selection labels (n_train,) in {0.0, 1.0}.
        X_val, S_val:     Validation features and labels.
        device:           Torch device.
        hidden:           Hidden width.
        depth:            Number of hidden layers.
        reparam:          If False (default), use plain SelectionModel
                          (homoscedastic probit). If True, use the
                          heteroscedastic SelectionModelReparam
                          (sigma(x) learned via reparam trick).
        min_sigma:        Floor on sigma(x); only used when reparam=True.
        dropout:          Dropout probability.
        lr:               Adam learning rate.
        weight_decay:     Adam L2 regularization weight.
        batch_size:       Mini-batch size.
        max_epochs:       Hard cap on epochs.
        patience:         Early-stopping patience.
        min_delta:        Minimum val_loss decrease counted as improvement.

    Returns:
        3-tuple:
            model:         Trained SelectionModel or SelectionModelReparam
                           (best-val state reloaded). Caller can detect
                           the variant via `isinstance`.
            history:       Dict with 'train_loss' and 'val_loss' (probit
                           NLL, lower is better) lists, one entry per
                           epoch actually run.
            best_val_loss: Best validation probit NLL achieved.
    """
    Xt = torch.from_numpy(X_train).float()
    St = torch.from_numpy(S_train).float()
    Xv = torch.from_numpy(X_val).float().to(device)
    Sv = torch.from_numpy(S_val).float().to(device)

    train_loader = DataLoader(TensorDataset(Xt, St), batch_size=batch_size,
                              shuffle=True, drop_last=False)
    if reparam:
        model = SelectionModelReparam(in_dim=X_train.shape[1], hidden=hidden,
                                       depth=depth, dropout=dropout,
                                       min_sigma=min_sigma).to(device)
    else:
        model = SelectionModel(in_dim=X_train.shape[1], hidden=hidden,
                                depth=depth, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf"); best_state = None; epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for xb, sb in train_loader:
            xb = xb.to(device, non_blocking=True); sb = sb.to(device, non_blocking=True)
            opt.zero_grad()
            z = model(xb)  # reparam mode: stochastic z; plain mode: deterministic
            loss = probit_nll(z, sb)
            loss.backward(); opt.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(Xt)
        model.eval()
        with torch.no_grad():
            # For reparam model in eval mode, forward() returns mu (no noise)
            val_loss = probit_nll(model(Xv), Sv).item()
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val_loss


@torch.no_grad()
def _compute_imr(selection_model: nn.Module,
                 X: np.ndarray, device: str) -> np.ndarray:
    """
    Run the selection model forward and return the inverse Mills ratio per row.

    For SelectionModelReparam, uses `deterministic_index(x) = mu(x)`
    rather than the stochastic forward (no sampling at inference). For
    plain SelectionModel, uses the standard forward.

    Args:
        selection_model: A trained SelectionModel or SelectionModelReparam.
        X:               (n_rows, in_dim) NumPy array of features.
        device:          Torch device string.

    Returns:
        (n_rows,) float32 NumPy array — IMR value lambda(z) = phi(z) / Phi(z)
        per row, suitable for concatenation as an additional outcome
        feature.
    """
    selection_model.eval()
    x = torch.from_numpy(X).float().to(device)
    if isinstance(selection_model, SelectionModelReparam):
        z = selection_model.deterministic_index(x)
    else:
        z = selection_model(x)
    return inverse_mills_ratio(z).cpu().numpy().astype(np.float32)


# -----------------------------------------------------------------------------
# data splitting internal helper functions
# -----------------------------------------------------------------------------

def _random_split(X, y, val_size, random_state):
    """
    Simple two-way (train, val) random split via sklearn's train_test_split.
    Used in the vanilla CQR path when conformal calibration is disabled.

    Args:
        X:            Feature matrix (n_rows, n_features).
        y:            Target array (n_rows,).
        val_size:     Fraction in (0, 1) used for the validation set.
        random_state: RNG seed for reproducible splits.

    Returns:
        4-tuple (X_train, X_val, y_train, y_val) — sklearn's native return
        ordering, NOT (X_train, y_train, X_val, y_val).
    """
    return train_test_split(X, y, test_size=val_size, random_state=random_state)


def _random_three_way_split(X, y, val_size, calib_size, random_state):
    """
    Three-way random split into train / val / calib sets.

    Used in the vanilla CQR path when conformal calibration is enabled.
    The calib set is held out FIRST, then val is carved out of the
    remainder so the requested val/calib fractions are over the full
    n_rows (not relative-to-remainder fractions).

    Args:
        X:            Feature matrix (n_rows, n_features).
        y:            Target array (n_rows,).
        val_size:     Fraction in (0, 1) for validation.
        calib_size:   Fraction in (0, 1) for conformal calibration.
                      Combined val_size + calib_size must be < 1.0.
        random_state: RNG seed.

    Returns:
        6-tuple (X_train, y_train, X_val, y_val, X_calib, y_calib).

    Raises:
        ValueError: If val_size + calib_size >= 1.0.
    """
    if val_size + calib_size >= 1.0:
        raise ValueError(f"val_size + calib_size must be < 1")
    X_rem, X_calib, y_rem, y_calib = train_test_split(
        X, y, test_size=calib_size, random_state=random_state)
    val_size_adj = val_size / (1.0 - calib_size)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_rem, y_rem, test_size=val_size_adj, random_state=random_state)
    return X_tr, y_tr, X_val, y_val, X_calib, y_calib


def _three_way_split_indices(n: int, val_size: float, calib_size: float,
                             random_state: int):
    """
    Three-way split that returns INDICES rather than sliced arrays.

    Used by the Heckman path, where the SAME row ordering must be applied
    to multiple arrays (outcome embeddings, selection embeddings, y, S)
    so they stay aligned per id.

    Args:
        n:            Total number of rows to split over.
        val_size:     Fraction in (0, 1) for the val split.
        calib_size:   Fraction in (0, 1) for the calib split. Use 0.0 to
                      get an empty calib_idx when calibration is disabled.
                      val_size + calib_size must be < 1.0.
        random_state: RNG seed for the permutation.

    Returns:
        3-tuple (train_idx, val_idx, calib_idx) — each a 1-D int NumPy
        array, disjoint, together covering 0..n-1. Use these to index any
        arrays of length n.

    Raises:
        ValueError: If val_size + calib_size >= 1.0.
    """
    if val_size + calib_size >= 1.0:
        raise ValueError(f"val_size + calib_size must be < 1")
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    n_calib = int(round(n * calib_size))
    n_val = int(round(n * val_size))
    return perm[n_calib + n_val:], perm[n_calib:n_calib + n_val], perm[:n_calib]


# -----------------------------------------------------------------------------
# Conformalized Quantile Regression (CQR) conformal calibration
# -----------------------------------------------------------------------------

def _predict_quantiles_destandardized(model, X, y_mean, y_std, device):
    """
    Run the quantile model forward, de-standardize, and sort to fix any
    quantile-crossing.

    Args:
        model:  Trained FFNNQuantileRegressor (predictions in standardized
                y-space).
        X:      (n_rows, in_dim) feature matrix.
        y_mean: Train-set mean of y (from train_single_head's return).
        y_std:  Train-set std of y.
        device: Torch device.

    Returns:
        (n_rows, n_quantiles) float NumPy array of predictions in the
        ORIGINAL y-scale, with each row sorted ascending. Sorting is the
        post-hoc "rearrangement" fix for quantile crossing — without it,
        a poorly-fit head can output q_lo > q_hi for some rows.
    """
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X).float().to(device)
        q_std = model(x).cpu().numpy()
    q = q_std * y_std + y_mean
    return np.sort(q, axis=1)


def fit_cqr_offset(model, X_calib, y_calib, y_mean, y_std, device, alpha=0.1):
    """
    Compute the Conformalized Quantile Regression (CQR) offset Q_hat on a
    held-out calibration set.

    For each calibration row i, compute the conformity score
        E_i = max( q_lo(x_i) - y_i,  y_i - q_hi(x_i) )
    where q_lo and q_hi are the lower and upper de-standardized quantile
    predictions. The CQR offset Q_hat is the (1-alpha)*(n+1)/n quantile of
    the {E_i} (clipped to <= 1.0). Final prediction intervals are then
    [q_lo - Q_hat, q_hi + Q_hat], which provably achieve marginal
    coverage >= 1-alpha under exchangeability (Romano, Patterson, Candes 2019).

    Q_hat may be NEGATIVE: if the raw quantile model already over-covers,
    CQR shrinks the interval. This is what distinguishes CQR from plain
    split conformal regression (where the offset is non-negative).

    Args:
        model:    Trained FFNNQuantileRegressor.
        X_calib:  (n_calib, in_dim) calibration features.
        y_calib:  (n_calib,) calibration targets.
        y_mean:   Train-set mean of y (used to de-standardize).
        y_std:    Train-set std of y (used to de-standardize).
        device:   Torch device.
        alpha:    Target miscoverage rate; coverage will be >= 1 - alpha.

    Returns:
        2-tuple:
            Q_hat:        Float — additive offset to apply to quantile
                          predictions. Save this in the checkpoint; at
                          inference, intervals are
                          [q_lo - Q_hat, q_hi + Q_hat].
            calib_metrics: Dict of calibration diagnostics:
                'Q_hat':                          (same as above)
                'alpha':                          alpha
                'target_coverage':                1 - alpha
                'quantile_level_used':            actual quantile level
                                                  (ceil((n+1)(1-alpha))/n)
                'empirical_coverage_calib':       coverage AFTER applying
                                                  Q_hat on the calib set
                'mean_interval_half_width_calib': interval half-width AFTER
                                                  Q_hat
                'raw_coverage_pre_cqr':           coverage BEFORE Q_hat
                                                  (just the quantile model)
                'raw_interval_half_width_pre_cqr': half-width BEFORE Q_hat
                'Q_hat_sign':                     'positive' (interval
                                                  expanded) / 'negative'
                                                  (interval shrunk) / 'zero'
                'n_calib':                        n_calib

    Raises:
        ValueError: If alpha is not strictly in (0, 1).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1)")
    q_pred = _predict_quantiles_destandardized(model, X_calib, y_mean, y_std, device)
    q_lo, q_hi = q_pred[:, 0], q_pred[:, -1]
    E = np.maximum(q_lo - y_calib, y_calib - q_hi)
    n = len(y_calib)
    q_level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    Q_hat = float(np.quantile(E, q_level, method="higher"))
    lower, upper = q_lo - Q_hat, q_hi + Q_hat
    return Q_hat, {
        "Q_hat": Q_hat, "alpha": alpha, "target_coverage": 1.0 - alpha,
        "quantile_level_used": float(q_level),
        "empirical_coverage_calib": float(((y_calib >= lower) & (y_calib <= upper)).mean()),
        "mean_interval_half_width_calib": float(((upper - lower) / 2.0).mean()),
        "raw_coverage_pre_cqr": float(((y_calib >= q_lo) & (y_calib <= q_hi)).mean()),
        "raw_interval_half_width_pre_cqr": float(((q_hi - q_lo) / 2.0).mean()),
        "Q_hat_sign": ("negative (interval shrunk)" if Q_hat < 0
                       else ("positive (interval expanded)" if Q_hat > 0
                             else "zero (no adjustment)")),
        "n_calib": int(n),
    }


# -----------------------------------------------------------------------------
# Per-head training: vanilla CQR vs Heckman
# -----------------------------------------------------------------------------

def _train_head_vanilla_cqr(X_obs, y, taus, calibrate, val_size, calib_size,
                            random_state, alpha, min_calib_samples,
                            head_patience, device, depth, **train_kwargs):
    """
    Train a single CQR head with no selection correction (vanilla path).

    Pipeline per head:
        1. Split observed rows into train/val (and optionally calib).
        2. Train an FFNNQuantileRegressor on (train, val).
        3. If calibration is enabled and the calib set is large enough,
           fit the CQR offset Q_hat on the held-out calib set.

    Args:
        X_obs:              (n_obs, in_dim) feature matrix for rows where
                            this target was OBSERVED.
        y:                  (n_obs,) target values.
        taus:               Quantile levels tuple.
        calibrate:          If True, hold out a calib set and fit Q_hat.
                            If False, only train/val split is used and
                            Q_hat is returned as None.
        val_size:           Val fraction.
        calib_size:         Calib fraction (ignored when calibrate=False).
        random_state:       RNG seed for the splits.
        alpha:              Target miscoverage rate (passed to fit_cqr_offset).
        min_calib_samples:  Minimum calib-set size; below this, Q_hat is
                            left None and calib_status reports the skip.
        head_patience:      Early-stopping patience for the quantile model.
        device:             Torch device.
        depth:              Outcome MLP depth.
        **train_kwargs:     Passed through to train_single_head (hidden,
                            max_epochs, etc.).

    Returns:
        Dict with fixed-shape result fields:
            status:                          'trained' on success;
                                             'skipped_split_failed' if the
                                             split produced unusable sizes.
            selection_used:                  Always False on this code path.
            quantile_model:                  The trained FFNNQuantileRegressor.
            selection_model:                 Always None on this path.
            history_quantile, history_selection: Loss histories
                                             (selection history is None here).
            best_val_pinball_standardized:   Best val pinball in z-score space.
            y_mean, y_std:                   Train-set standardization.
            n_train, n_val:                  Split sizes.
            Q_hat, calib_status, calib_metrics:
                                             From fit_cqr_offset; Q_hat is
                                             None and metrics is None when
                                             calibration was disabled or
                                             skipped.
    """
    if calibrate:
        X_tr, y_tr, X_val, y_val, X_calib, y_calib = _random_three_way_split(
            X_obs, y, val_size=val_size, calib_size=calib_size, random_state=random_state)
    else:
        X_tr, X_val, y_tr, y_val = _random_split(X_obs, y, val_size=val_size,
                                                  random_state=random_state)
        X_calib, y_calib = None, None
    if len(y_val) < 1 or len(y_tr) < 2:
        return {"status": "skipped_split_failed"}
    model, history, best_val_loss, y_mean, y_std = train_single_head(
        X_tr, y_tr, X_val, y_val, taus=taus, device=device,
        patience=head_patience, depth=depth, **train_kwargs)
    Q_hat, calib_metrics, calib_status = None, None, "disabled"
    if calibrate and y_calib is not None:
        if len(y_calib) >= min_calib_samples:
            Q_hat, calib_metrics = fit_cqr_offset(
                model, X_calib, y_calib, y_mean=y_mean, y_std=y_std,
                device=device, alpha=alpha)
            calib_status = "fitted"
        else:
            calib_status = "skipped_too_few_calib_samples"
    return {
        "status": "trained", "selection_used": False,
        "quantile_model": model, "selection_model": None,
        "history_quantile": history, "history_selection": None,
        "best_val_pinball_standardized": float(best_val_loss),
        "y_mean": y_mean, "y_std": y_std,
        "y_min": float(np.min(y)), "y_max": float(np.max(y)),
        "n_train": int(len(y_tr)), "n_val": int(len(y_val)),
        "Q_hat": Q_hat, "calib_status": calib_status, "calib_metrics": calib_metrics,
    }


def _train_head_heckman(X_outcome_universe: np.ndarray,
                        X_selection_universe: np.ndarray,
                        y_universe: np.ndarray, S: np.ndarray,
                        taus, calibrate, val_size, calib_size, random_state,
                        alpha, min_calib_samples, head_patience, device,
                        outcome_depth: int, selection_depth: int,
                        selection_hidden: int, selection_max_epochs: int,
                        reparam_selection: bool, min_sigma: float,
                        **train_kwargs):
    """
    Heckman two-step training for one head, with optional separate
    selection embeddings and optional heteroscedastic-probit reparameterization.

    Pipeline per head:
      Step 0: Three-way split of ids (train/val/calib) using indices so
              every aligned array uses the same row partition.
      Step 1: Fit selection probit on (X_selection_universe, S) over the
              FULL population (selected + unselected) in train and val splits.
              Uses SelectionModelReparam when reparam_selection=True,
              else plain SelectionModel.
      Step 2: For selected rows in each split, run the trained selection
              model forward to compute the IMR per row. Concatenate the
              IMR as an extra feature onto X_outcome.
      Step 3: Train the FFNNQuantileRegressor on the augmented outcome
              features over selected rows only.
      Step 4: If calibration is enabled, fit CQR offset Q_hat on the
              augmented-features calib set.
      Step 5: Compute aggregated diagnostics (mean/std of sigma(x) for
              the reparam model, IMR stats, selection rate, etc.).

    Args:
        X_outcome_universe:   (n_universe, hidden) outcome embeddings,
                              one row per id in the embedded population.
        X_selection_universe: (n_universe, hidden) selection embeddings.
                              May be the same object as X_outcome_universe
                              (no separate selection text) or a distinct
                              array (true exclusion restriction).
        y_universe:           (n_universe,) observed value where S=1,
                              NaN where S=0.
        S:                    (n_universe,) float32 selection indicator
                              in {0.0, 1.0}.
        taus:                 Quantile levels tuple for the outcome model.
        calibrate, val_size, calib_size, random_state, alpha,
        min_calib_samples, head_patience:
                              Same semantics as in _train_head_vanilla_cqr.
        device:               Torch device.
        outcome_depth:        Depth of the FFNNQuantileRegressor.
        selection_depth:      Depth of the selection model.
        selection_hidden:     Hidden width of the selection model.
        selection_max_epochs: Max epochs for the selection model.
        reparam_selection:    If True, use SelectionModelReparam
                              (heteroscedastic probit via reparam trick).
                              If False, use plain SelectionModel.
        min_sigma:            Floor on sigma(x) (reparam only).
        **train_kwargs:       Passed through to train_single_head for the
                              outcome model.

    Returns:
        Dict with fields:
            status:                          'trained' on success;
                                             'skipped_selection_one_class'
                                             if S has only one class in train;
                                             'skipped_too_few_selected' if
                                             too few S=1 rows for the outcome
                                             model.
            selection_used:                  Always True on this code path.
            quantile_model:                  Trained FFNNQuantileRegressor.
                                             in_dim is hidden + 1 (IMR
                                             concatenated as last feature).
            selection_model:                 Trained SelectionModel or
                                             SelectionModelReparam.
            history_quantile, history_selection:
                                             Per-epoch loss dicts for both
                                             models.
            best_val_pinball_standardized:   Outcome model's best val pinball.
            best_val_probit_nll:             Selection model's best val NLL.
            y_mean, y_std:                   Outcome target standardization.
            n_train_selection:               Selection-stage train size
                                             (full population).
            n_train_outcome, n_val_outcome:  Outcome-stage train/val sizes
                                             (selected rows only).
            selection_rate_train:            Empirical P(S=1) in train.
            mean_predicted_selection_prob:   Phi(z) averaged over train.
            mean_imr_train, std_imr_train:   IMR distribution stats.
            reparam_selection:               Echo of the input flag.
            mean_selection_sigma, std_selection_sigma:
                                             sigma(x) stats (None when not
                                             using reparam). std > 0 is the
                                             empirical signature of
                                             Klein-Vella identification
                                             being active.
            separate_selection_embeddings:   True iff X_outcome_universe
                                             is not X_selection_universe.
            Q_hat, calib_status, calib_metrics:
                                             From fit_cqr_offset on the
                                             augmented features.
    """
    n_universe = X_outcome_universe.shape[0]
    train_idx, val_idx, calib_idx = _three_way_split_indices(
        n_universe, val_size=val_size,
        calib_size=calib_size if calibrate else 0.0,
        random_state=random_state)

    # Step 1: selection model on X_selection_universe (full population in each split)
    X_tr_sel_all = X_selection_universe[train_idx]
    X_val_sel_all = X_selection_universe[val_idx]
    S_tr = S[train_idx]; S_val = S[val_idx]
    if S_tr.sum() < 2 or (1 - S_tr).sum() < 2:
        return {"status": "skipped_selection_one_class"}

    sel_model, sel_history, sel_val_nll = train_selection_model(
        X_tr_sel_all, S_tr, X_val_sel_all, S_val, device=device,
        hidden=selection_hidden, depth=selection_depth,
        reparam=reparam_selection, min_sigma=min_sigma,
        max_epochs=selection_max_epochs, patience=head_patience)

    # Step 2: compute IMR on the SELECTION embeddings for selected rows in each split.
    # IMR is then concatenated to the OUTCOME embeddings as the additional feature.
    train_sel_mask = S[train_idx] == 1
    val_sel_mask = S[val_idx] == 1
    calib_sel_mask = S[calib_idx] == 1 if calibrate else np.array([], dtype=bool)

    if train_sel_mask.sum() < 2 or val_sel_mask.sum() < 1:
        return {"status": "skipped_too_few_selected"}

    # Selection embeddings used to compute IMR
    X_sel_tr_selected = X_selection_universe[train_idx][train_sel_mask]
    X_sel_val_selected = X_selection_universe[val_idx][val_sel_mask]
    imr_tr = _compute_imr(sel_model, X_sel_tr_selected, device=device)
    imr_val = _compute_imr(sel_model, X_sel_val_selected, device=device)

    # Outcome embeddings + IMR for the outcome network
    X_out_tr_selected = X_outcome_universe[train_idx][train_sel_mask]
    X_out_val_selected = X_outcome_universe[val_idx][val_sel_mask]
    y_tr_sel = y_universe[train_idx][train_sel_mask]
    y_val_sel = y_universe[val_idx][val_sel_mask]
    X_tr_aug = np.concatenate([X_out_tr_selected, imr_tr[:, None]], axis=1).astype(np.float32)
    X_val_aug = np.concatenate([X_out_val_selected, imr_val[:, None]], axis=1).astype(np.float32)

    if calibrate:
        X_sel_calib_selected = X_selection_universe[calib_idx][calib_sel_mask]
        X_out_calib_selected = X_outcome_universe[calib_idx][calib_sel_mask]
        imr_calib = _compute_imr(sel_model, X_sel_calib_selected, device=device)
        y_calib_sel = y_universe[calib_idx][calib_sel_mask]
        X_calib_aug = np.concatenate(
            [X_out_calib_selected, imr_calib[:, None]], axis=1).astype(np.float32)
    else:
        X_calib_aug, y_calib_sel = None, None

    # Step 3: quantile model on augmented features
    q_model, q_history, best_val_loss, y_mean, y_std = train_single_head(
        X_tr_aug, y_tr_sel, X_val_aug, y_val_sel,
        taus=taus, device=device, patience=head_patience,
        depth=outcome_depth, **train_kwargs)

    # Step 4: CQR calibration
    Q_hat, calib_metrics, calib_status = None, None, "disabled"
    if calibrate and y_calib_sel is not None:
        if len(y_calib_sel) >= min_calib_samples:
            Q_hat, calib_metrics = fit_cqr_offset(
                q_model, X_calib_aug, y_calib_sel,
                y_mean=y_mean, y_std=y_std, device=device, alpha=alpha)
            calib_status = "fitted"
        else:
            calib_status = "skipped_too_few_calib_samples"

    # Diagnostics
    with torch.no_grad():
        x_all_sel = torch.from_numpy(X_selection_universe[train_idx]).float().to(device)
        if reparam_selection:
            z_all = sel_model.deterministic_index(x_all_sel)
            _, sigma_all = sel_model.get_mu_sigma(x_all_sel)
            mean_sigma = float(sigma_all.mean().item())
            std_sigma = float(sigma_all.std().item())
        else:
            z_all = sel_model(x_all_sel)
            mean_sigma, std_sigma = None, None
        prob_sel_mean = float(torch.special.ndtr(z_all).mean().item())

    # Plausible target support = min/max over ALL observed (selected) outcome
    # labels this head was developed on (train + val + calib selected rows).
    _y_obs_parts = [y_tr_sel, y_val_sel]
    if calibrate and y_calib_sel is not None:
        _y_obs_parts.append(y_calib_sel)
    _y_observed = np.concatenate(_y_obs_parts)

    return {
        "status": "trained", "selection_used": True,
        "quantile_model": q_model, "selection_model": sel_model,
        "history_quantile": q_history, "history_selection": sel_history,
        "best_val_pinball_standardized": float(best_val_loss),
        "best_val_probit_nll": float(sel_val_nll),
        "y_mean": y_mean, "y_std": y_std,
        "y_min": float(np.min(_y_observed)), "y_max": float(np.max(_y_observed)),
        "n_train_selection": int(len(train_idx)),
        "n_train_outcome": int(train_sel_mask.sum()),
        "n_val_outcome": int(val_sel_mask.sum()),
        "selection_rate_train": float(S_tr.mean()),
        "mean_predicted_selection_prob": prob_sel_mean,
        "mean_imr_train": float(np.mean(imr_tr)),
        "std_imr_train": float(np.std(imr_tr)),
        "reparam_selection": bool(reparam_selection),
        "mean_selection_sigma": mean_sigma,
        "std_selection_sigma": std_sigma,
        "separate_selection_embeddings": bool(X_outcome_universe is not X_selection_universe),
        "Q_hat": Q_hat, "calib_status": calib_status, "calib_metrics": calib_metrics,
    }


# -----------------------------------------------------------------------------
# Top-level training driver
# -----------------------------------------------------------------------------

def train_all_heads(per_target_data: Iterable[tuple],
                    out_dir: Path,
                    val_size: float = 0.15, random_state: int = 42,
                    min_samples_per_head: int = 32,
                    base_patience: int = 10,
                    calibrate: bool = True, calib_size: float = 0.15,
                    min_calib_samples: int = 16,
                    alpha: float = 0.1,
                    selection: bool = False,
                    selection_hidden: int = 128,
                    selection_depth: int = 2,
                    selection_max_epochs: int = 100,
                    reparam_selection: bool = False,
                    min_sigma: float = 0.1,
                    depth: int = 2,
                    device: Optional[str] = None,
                    **train_kwargs) -> list[dict]:
    """
    Train one CQR head per target, with optional Heckman selection
    correction. This is the orchestrator over `_train_head_vanilla_cqr`
    and `_train_head_heckman`, and is responsible for:

        - Iterating over per-target tuples produced by `long_to_per_target`,
        - Skipping targets with too few samples / zero variance / non-numeric
          values (recorded as `status='skipped_*'` in metadata),
        - Saving one .pt checkpoint per trained head (`head_NNNN.pt`),
        - Writing a metadata.json index with per-head diagnostics.

    Args:
        per_target_data:        Iterable of per-target tuples from
                                `long_to_per_target`. Tuple shape depends
                                on the `selection` flag:
                                    selection=False -> 3-tuples
                                        (name, X, y)
                                    selection=True  -> 5-tuples
                                        (name, X_outcome, X_selection, y, S)
        out_dir:                Output directory for .pt files and
                                metadata.json. Created if it doesn't
                                exist.
        val_size:               Per-head validation fraction.
        random_state:           Seed for per-head splits.
        min_samples_per_head:   Heads with fewer observed values (or fewer
                                selected rows in the Heckman path) are
                                skipped with a 'skipped_*' status.
        base_patience:          Early-stopping patience for BOTH the
                                outcome and selection models.
        calibrate:              Whether to run CQR conformal calibration
                                per head.
        calib_size:             Per-head calibration fraction.
        min_calib_samples:      Heads with smaller calib sets save
                                Q_hat=None and produce NaN intervals.
        alpha:                  Target miscoverage rate (1-alpha coverage).
        selection:              If True, run Heckman two-step path.
                                Tuples must be 5-tuples.
        selection_hidden:       Hidden width of the selection MLP.
        selection_depth:        Number of hidden layers in selection MLP.
        selection_max_epochs:   Max epochs for the selection model.
        reparam_selection:      If True, selection model is
                                SelectionModelReparam (heteroscedastic
                                probit). Only used when selection=True.
        min_sigma:              Floor on sigma(x) (reparam only).
        depth:                  Depth of the outcome (quantile) MLP.
        device:                 Torch device. Auto-picks cuda if available.
        **train_kwargs:         Passed through to train_single_head
                                (e.g. `hidden`, `max_epochs`, `lr`,
                                `dropout`, `batch_size`).

    Returns:
        List of dicts, one per target (length = len(per_target_data)).
        Each entry includes at minimum `idx`, `name`, and `status`. For
        `status='trained'` entries, additional fields summarize the
        training outcome — see per-head trainer docs for the full set.

    Side effects:
        Writes one .pt file per trained head to `out_dir/head_NNNN.pt`,
        and a single `out_dir/metadata.json` containing the full metadata
        list (JSON-serialized).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    taus = _quantile_levels_from_alpha(alpha)

    per_target_data = list(per_target_data)
    n_targets = len(per_target_data)
    mode_bits = []
    if selection:
        mode_bits.append("HECKMAN")
        if reparam_selection:
            mode_bits.append("reparam (heteroscedastic probit)")
    else:
        mode_bits.append("vanilla CQR")
    mode = " + ".join(mode_bits)
    cmsg = (f"  alpha={alpha}  calib_size={calib_size}" if calibrate else "")
    print(f"[train] device={device}  n_targets={n_targets}  mode={mode}  "
          f"depth_outcome={depth}  depth_selection={selection_depth}  "
          f"calibrate={calibrate}{cmsg}")

    metadata: list[dict] = []

    for j, item in enumerate(tqdm(per_target_data, desc="Heads")):
        if selection:
            if len(item) != 5:
                raise ValueError(
                    f"selection=True expects 5-tuples (name, X_out, X_sel, y, S); "
                    f"got tuple of length {len(item)}. Call "
                    f"long_to_per_target(..., include_selection=True).")
            col, X_outcome, X_selection, y_universe, S = item
        else:
            if len(item) == 5:
                col, X_outcome, _, y_universe, S = item
                sel_mask = (S == 1) & np.isfinite(y_universe)
                X_obs_local = X_outcome[sel_mask]
                y_local = y_universe[sel_mask]
            else:
                col, X_obs_local, y_local = item

        if selection:
            try:
                y_clean = np.asarray(y_universe, dtype=np.float64)
            except (TypeError, ValueError):
                metadata.append({"idx": j, "name": col, "status": "skipped_non_numeric"})
                continue
            n_selected = int(S.sum())
            if n_selected < min_samples_per_head:
                metadata.append({"idx": j, "name": col, "status": "skipped_too_few_selected",
                                 "n_selected": n_selected})
                continue
            if not np.isfinite(y_clean[S == 1]).all():
                bad = (S == 1) & ~np.isfinite(y_clean)
                S = S.copy(); S[bad] = 0.0
                if int(S.sum()) < min_samples_per_head:
                    metadata.append({"idx": j, "name": col, "status": "skipped_too_few_selected"})
                    continue
            if np.std(y_clean[S == 1]) < 1e-12:
                metadata.append({"idx": j, "name": col, "status": "skipped_no_variance"})
                continue

            result = _train_head_heckman(
                X_outcome_universe=X_outcome,
                X_selection_universe=X_selection,
                y_universe=y_clean, S=S.astype(np.float32),
                taus=taus, calibrate=calibrate,
                val_size=val_size, calib_size=calib_size,
                random_state=random_state, alpha=alpha,
                min_calib_samples=min_calib_samples,
                head_patience=base_patience, device=device,
                outcome_depth=depth, selection_depth=selection_depth,
                selection_hidden=selection_hidden,
                selection_max_epochs=selection_max_epochs,
                reparam_selection=reparam_selection,
                min_sigma=min_sigma, **train_kwargs)
        else:
            try:
                y_clean = np.asarray(y_local, dtype=np.float64)
            except (TypeError, ValueError):
                metadata.append({"idx": j, "name": col, "status": "skipped_non_numeric"})
                continue
            finite = np.isfinite(y_clean)
            y_clean = y_clean[finite]
            X_local_filt = X_obs_local[finite]
            if len(y_clean) < min_samples_per_head:
                metadata.append({"idx": j, "name": col, "status": "skipped_too_few_samples"})
                continue
            if np.std(y_clean) < 1e-12:
                metadata.append({"idx": j, "name": col, "status": "skipped_no_variance"})
                continue
            result = _train_head_vanilla_cqr(
                X_obs=X_local_filt, y=y_clean, taus=taus, calibrate=calibrate,
                val_size=val_size, calib_size=calib_size, random_state=random_state,
                alpha=alpha, min_calib_samples=min_calib_samples,
                head_patience=base_patience, device=device, depth=depth, **train_kwargs)

        if result["status"] != "trained":
            metadata.append({"idx": j, "name": col,
                             **{k: v for k, v in result.items()
                                if isinstance(v, (int, float, str, bool, type(None)))}})
            continue

        # --- save ---
        q_model = result["quantile_model"]
        ckpt: dict = {
            "quantile_state_dict": q_model.state_dict(),
            "quantile_in_dim": int(q_model.net[0].in_features),
            "quantile_hidden": int(q_model.hidden),
            "quantile_depth": int(q_model.depth),
            "n_quantiles": len(taus),
            "quantile_levels": list(taus),
            "target_name": col,
            "y_mean": float(result["y_mean"]),
            "y_std": float(result["y_std"]),
            "y_min": (float(result["y_min"]) if result.get("y_min") is not None else None),
            "y_max": (float(result["y_max"]) if result.get("y_max") is not None else None),
            "Q_hat": result["Q_hat"],
            "alpha": float(alpha) if result["Q_hat"] is not None else None,
            "selection_used": result["selection_used"],
        }
        if result["selection_used"]:
            sel_model = result["selection_model"]
            ckpt.update({
                "selection_state_dict": sel_model.state_dict(),
                "selection_in_dim": int(next(sel_model.parameters()).shape[-1])
                                    if reparam_selection
                                    else int(sel_model.net[0].in_features),
                "selection_hidden": int(sel_model.hidden),
                "selection_depth": int(sel_model.depth),
                "selection_is_reparam": bool(reparam_selection),
                "selection_min_sigma": float(min_sigma) if reparam_selection else None,
                "separate_selection_embeddings": bool(result.get("separate_selection_embeddings", False)),
            })
            # Better way to get selection in_dim from the model itself:
            if reparam_selection:
                ckpt["selection_in_dim"] = int(sel_model.trunk[0].in_features)
            else:
                ckpt["selection_in_dim"] = int(sel_model.net[0].in_features)
        torch.save(ckpt, out_dir / f"head_{j:04d}.pt")

        # --- metadata entry ---
        meta_entry = {
            "idx": j, "name": col, "status": "trained",
            "selection_used": result["selection_used"],
            "outcome_depth": depth,
            "epochs_run_quantile": len(result["history_quantile"]["val_loss"]),
            "best_val_pinball_standardized": result["best_val_pinball_standardized"],
            "y_train_mean": float(result["y_mean"]),
            "y_train_std": float(result["y_std"]),
            "y_train_min": (float(result["y_min"]) if result.get("y_min") is not None else None),
            "y_train_max": (float(result["y_max"]) if result.get("y_max") is not None else None),
            "quantile_levels": list(taus),
            "calibration_status": result["calib_status"],
        }
        if result["selection_used"]:
            meta_entry.update({
                "selection_depth": selection_depth,
                "selection_is_reparam": result["reparam_selection"],
                "separate_selection_embeddings": result["separate_selection_embeddings"],
                "epochs_run_selection": len(result["history_selection"]["val_loss"]),
                "best_val_probit_nll": result["best_val_probit_nll"],
                "n_train_selection_universe": result["n_train_selection"],
                "n_train_outcome_selected": result["n_train_outcome"],
                "n_val_outcome_selected": result["n_val_outcome"],
                "selection_rate_train": result["selection_rate_train"],
                "mean_predicted_selection_prob": result["mean_predicted_selection_prob"],
                "mean_imr_train": result["mean_imr_train"],
                "std_imr_train": result["std_imr_train"],
                "mean_selection_sigma": result["mean_selection_sigma"],
                "std_selection_sigma": result["std_selection_sigma"],
            })
        else:
            meta_entry.update({"n_train": result["n_train"], "n_val": result["n_val"]})
        if result["calib_metrics"] is not None:
            meta_entry.update(result["calib_metrics"])
        metadata.append(meta_entry)

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    n_trained = sum(m["status"] == "trained" for m in metadata)
    print(f"[train] wrote {n_trained} / {n_targets} heads to {out_dir}")
    return metadata


# =============================================================================
# Inference internal helper functions
# =============================================================================

#added this for logical bounding based on AoU observed data
def _clip_to_support(values: np.ndarray,
                     y_min: Optional[float],
                     y_max: Optional[float]) -> np.ndarray:
    """
    Clip predictions / interval bounds to a fixed plausible support [y_min, y_max].

    Purpose:
        Quantile-regression outputs and the additive CQR margin Q_hat are
        unconstrained in the original target scale, so on out-of-distribution
        rows they can fall outside any plausible range. This enforces domain
        plausibility as a cheap post-hoc step.

        Coverage preservation: if the true label is guaranteed to lie in
        [y_min, y_max], clipping a conformal interval [L, U] inward to that
        range does NOT reduce its marginal coverage. For any y in
        [y_min, y_max]:
            max(L, y_min) <= y   <=>   L <= y      (because y_min <= y)
            min(U, y_max) >= y   <=>   U >= y      (because y_max >= y)
        so {y in clipped interval} == {y in original interval} on that event.
        Coverage is lost only on the probability mass where y itself lies
        outside the bounds, i.e. it is bounded by P(Y not in [y_min, y_max]).
        Prefer true domain/physical limits over empirical train min/max when
        holdout labels may legitimately exceed the observed training range.

    Args:
        values: Float NumPy array (any shape) of predictions or interval
                bounds in the ORIGINAL target scale. NaNs are passed through
                unchanged (np.clip leaves NaN untouched).
        y_min:  Lower clip bound. If None, no clipping is applied (returns
                `values` unchanged).
        y_max:  Upper clip bound. If None, no clipping is applied.

    Returns:
        NumPy array, same shape as `values`, clipped elementwise into
        [y_min, y_max]; NaNs preserved.

    Reference:
        Romano, Patterson & Candes, "Conformalized Quantile Regression,"
        NeurIPS 2019 (arXiv:1905.03222) -- the (1 - alpha) marginal-coverage
        guarantee under exchangeability that this clip is designed to preserve.
    """
    if y_min is None or y_max is None:
        return values
    return np.clip(values, y_min, y_max)


def _flag_out_of_support(lower_preclip: np.ndarray,
                         upper_preclip: np.ndarray,
                         y_min: Optional[float],
                         y_max: Optional[float],
                         tol_frac: float = 0.5) -> dict:
    """
    Scale sanity check: flag rows whose PRE-CLIP interval bounds fall far
    outside the plausible support [y_min, y_max].

    Purpose:
        Hard clipping (`_clip_to_support`) silently rescues an interval that
        was wildly wrong (e.g. driven by embedding misalignment, a saturated
        Q_hat from a tiny calibration set, or genuine extrapolation on shifted
        holdout data). A clipped-but-wrong interval looks plausible while being
        untrustworthy. This check measures, per row, how far the raw bounds
        overshot the support BEFORE clipping, so those rows can be surfaced for
        review rather than trusted blindly.

    Definition:
        Let W = y_max - y_min be the support width. For each row:
            below_i = max(0, y_min - lower_i)   # how far the lower bound
                                                # dropped below the floor
            above_i = max(0, upper_i - y_max)   # how far the upper bound
                                                # rose above the ceiling
            severity_i = max(below_i, above_i) / W
        A row is flagged when severity_i > tol_frac, i.e. a bound overshot the
        support edge by more than `tol_frac` of the whole support width.
        Rows with NaN bounds (uncalibrated heads, Q_hat is None) get severity 0
        and are not flagged, since an undefined interval cannot be assessed.

    Args:
        lower_preclip: (n_rows,) lower interval bounds in the ORIGINAL scale,
                       BEFORE any clipping. NaNs allowed.
        upper_preclip: (n_rows,) upper interval bounds in the ORIGINAL scale,
                       BEFORE any clipping. NaNs allowed.
        y_min:         Lower edge of the plausible support. If None (no bounds
                       available), nothing is flagged.
        y_max:         Upper edge of the plausible support. If None, nothing is
                       flagged.
        tol_frac:      Non-negative tolerance as a fraction of the support
                       width W. Default 0.5 means "flag a row if a bound landed
                       more than half a support-width past the edge." Use a
                       smaller value for stricter flagging.

    Returns:
        Dict of per-row diagnostics (arrays are (n_rows,)):
            'flag':          bool array -- True where severity > tol_frac.
            'severity':      float array -- normalized overshoot
                             max(below, above)/W (0 if within support).
            'below':         float array -- absolute overshoot below y_min.
            'above':         float array -- absolute overshoot above y_max.
            'support_width': float W (NaN if bounds unavailable).
            'tol_frac':      the tolerance used.
            'n_flagged':     int -- number of rows flagged.
    """
    n = int(lower_preclip.shape[0])
    if y_min is None or y_max is None:
        z = np.zeros(n, dtype=float)
        return {"flag": np.zeros(n, dtype=bool), "severity": z,
                "below": z.copy(), "above": z.copy(),
                "support_width": float("nan"), "tol_frac": float(tol_frac),
                "n_flagged": 0}
    width = float(y_max) - float(y_min)
    denom = width if width > 0 else 1.0
    below = np.maximum(0.0, float(y_min) - lower_preclip)
    above = np.maximum(0.0, upper_preclip - float(y_max))
    # NaN bounds (uncalibrated heads) cannot be assessed -> treat as in-support.
    below = np.where(np.isnan(below), 0.0, below)
    above = np.where(np.isnan(above), 0.0, above)
    severity = np.maximum(below, above) / denom
    flag = severity > float(tol_frac)
    return {"flag": flag, "severity": severity, "below": below, "above": above,
            "support_width": width, "tol_frac": float(tol_frac),
            "n_flagged": int(flag.sum())}


def predict_with_head(head_path: Path,
                      embeddings: np.ndarray,
                      selection_embeddings: Optional[np.ndarray] = None,
                      device: Optional[str] = None,
                      clip_bounds: Optional[tuple[float, float]] = None,
                      flag_tol_frac: float = 0.5,
                      return_diagnostics: bool = True
                      ) -> tuple:
    """
    Load one saved head from disk and produce predictions with calibrated
    prediction intervals for a batch of new rows.

    The function inspects the checkpoint to decide what to do:
        - If the head was trained without selection, runs the quantile
          model directly on `embeddings`.
        - If trained with Heckman and the SAME embeddings for both
          equations, computes IMR on `embeddings`, augments, and runs
          the quantile model.
        - If trained with a separate selection text column, computes IMR
          on `selection_embeddings` (which is required, not optional in
          this case) and augments with `embeddings` for the quantile model.

    The Conformalized Quantile Regression offset Q_hat is applied to
    produce the final interval bounds:
        lower = q_lo - Q_hat
        upper = q_hi + Q_hat
    where q_lo and q_hi are the lower and upper quantile predictions
    after de-standardization and sorting. If the head was saved without
    calibration (Q_hat is None), the bounds are returned as NaN.

    Args:
        head_path:            Path to a .pt file written by `train_all_heads`.
        embeddings:           (n_rows, hidden) outcome embeddings for the
                              rows to predict on. Hidden size must match
                              the head's expected `quantile_in_dim - 1`
                              (Heckman) or `quantile_in_dim` (no selection).
        selection_embeddings: (n_rows, hidden) selection embeddings,
                              aligned with `embeddings` row-for-row.
                              REQUIRED when the saved head was trained
                              with `separate_selection_embeddings=True`;
                              otherwise ignored (the head reuses
                              `embeddings` internally).
        device:               Torch device. Auto-picks cuda if available.
        clip_bounds:          Optional (y_min, y_max) plausible support. When
                              provided, OVERRIDES the train-label bounds stored
                              in the checkpoint; pass true domain/physical
                              limits here (e.g. a non-negative lab value, a
                              probability in [0, 1]) when holdout labels may
                              legitimately exceed the observed training range.
                              When None, the checkpoint's saved y_min/y_max are
                              used; if those are also absent, no clipping or
                              flagging is performed.
        flag_tol_frac:        Tolerance for the scale sanity check, as a
                              fraction of the support width. A row is flagged
                              when a PRE-CLIP bound overshot the support edge by
                              more than this fraction (default 0.5). See
                              `_flag_out_of_support`.
        return_diagnostics:   If True (default), returns a 4-tuple whose last
                              element is the out-of-support diagnostics dict. If
                              False, returns the legacy 3-tuple
                              (pred, lower, upper) for backward compatibility.

    Returns:
        If return_diagnostics is True (default), a 4-tuple:
            pred:  (n_rows,) median point prediction, CLIPPED to support.
            lower: (n_rows,) lower bound of the calibrated interval, CLIPPED to
                   support (NaN if the head was uncalibrated, Q_hat is None).
            upper: (n_rows,) upper bound of the calibrated interval, CLIPPED to
                   support (NaN if uncalibrated).
            oob:   Dict from `_flag_out_of_support`, augmented with the
                   pre-clip arrays and the support actually used:
                   'flag', 'severity', 'below', 'above', 'support_width',
                   'tol_frac', 'n_flagged', plus 'lower_preclip',
                   'upper_preclip', 'pred_preclip', 'y_min', 'y_max'. The
                   pre-clip arrays let you inspect exactly how far a flagged
                   row overshot before clamping.
        If return_diagnostics is False, the legacy 3-tuple (pred, lower, upper)
        with the same clipping applied but no diagnostics dict.

    Raises:
        ValueError: If the head was trained with `separate_selection_embeddings`
                    but `selection_embeddings` was not supplied. This
                    prevents the silent failure mode where a caller
                    forgets to pass the second embedding matrix and
                    silently gets IMR computed from the wrong vectors.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(head_path, map_location=device, weights_only=False)
    selection_used = bool(ckpt.get("selection_used", False))

    # Quantile model
    q_in = int(ckpt["quantile_in_dim"])
    q_hidden = int(ckpt.get("quantile_hidden", 256))
    q_depth = int(ckpt.get("quantile_depth", 2))
    n_quantiles = int(ckpt["n_quantiles"])
    q_model = FFNNQuantileRegressor(in_dim=q_in, n_quantiles=n_quantiles,
                                     hidden=q_hidden, depth=q_depth).to(device).eval()
    q_model.load_state_dict(ckpt["quantile_state_dict"])

    if selection_used:
        sel_in = int(ckpt["selection_in_dim"])
        sel_hidden = int(ckpt["selection_hidden"])
        sel_depth = int(ckpt.get("selection_depth", 2))
        is_reparam = bool(ckpt.get("selection_is_reparam", False))
        separate = bool(ckpt.get("separate_selection_embeddings", False))

        if is_reparam:
            sel_model = SelectionModelReparam(
                in_dim=sel_in, hidden=sel_hidden, depth=sel_depth,
                min_sigma=float(ckpt.get("selection_min_sigma") or 0.1)
            ).to(device).eval()
        else:
            sel_model = SelectionModel(
                in_dim=sel_in, hidden=sel_hidden, depth=sel_depth
            ).to(device).eval()
        sel_model.load_state_dict(ckpt["selection_state_dict"])

        # Pick the right embeddings for the selection model
        if separate:
            if selection_embeddings is None:
                raise ValueError(
                    f"Head at {head_path} was trained with a separate selection "
                    f"text column, so predict_with_head requires `selection_embeddings`. "
                    f"Pass the selection-text embeddings (same id ordering as `embeddings`).")
            sel_input = selection_embeddings.astype(np.float32)
        else:
            sel_input = embeddings.astype(np.float32)

        imr = _compute_imr(sel_model, sel_input, device=device)
        X_aug = np.concatenate([embeddings.astype(np.float32), imr[:, None]], axis=1)
        q_pred = _predict_quantiles_destandardized(
            q_model, X_aug, y_mean=float(ckpt["y_mean"]),
            y_std=float(ckpt["y_std"]), device=device)
    else:
        q_pred = _predict_quantiles_destandardized(
            q_model, embeddings.astype(np.float32),
            y_mean=float(ckpt["y_mean"]), y_std=float(ckpt["y_std"]), device=device)

    median_col = n_quantiles // 2
    pred = q_pred[:, median_col]
    q_lo, q_hi = q_pred[:, 0], q_pred[:, -1]
    Q_hat = ckpt.get("Q_hat")
    if Q_hat is not None:
        Q = float(Q_hat)
        lower, upper = q_lo - Q, q_hi + Q
    else:
        lower = np.full_like(pred, np.nan); upper = np.full_like(pred, np.nan)

    # --- resolve plausible support: explicit clip_bounds override checkpoint estimate ---
    if clip_bounds is not None:
        y_min, y_max = float(clip_bounds[0]), float(clip_bounds[1])
    else:
        y_min, y_max = ckpt.get("y_min"), ckpt.get("y_max")

    # --- scale sanity check on the PRE-CLIP bounds (before any clamping) ---
    oob = _flag_out_of_support(lower, upper, y_min, y_max, tol_frac=flag_tol_frac)
    oob["lower_preclip"] = lower.copy()
    oob["upper_preclip"] = upper.copy()
    oob["pred_preclip"] = pred.copy()
    oob["y_min"] = (float(y_min) if y_min is not None else None)
    oob["y_max"] = (float(y_max) if y_max is not None else None)

    # --- bound point prediction and interval to the plausible support ---
    pred = _clip_to_support(pred, y_min, y_max)
    lower = _clip_to_support(lower, y_min, y_max)
    upper = _clip_to_support(upper, y_min, y_max)

    if return_diagnostics:
        return pred, lower, upper, oob
    return pred, lower, upper


def predict_all_heads(heads_dir: Path, embeddings: np.ndarray,
                      selection_embeddings: Optional[np.ndarray] = None,
                      device: Optional[str] = None,
                      clip_bounds: Optional[tuple[float, float]] = None,
                      flag_tol_frac: float = 0.5) -> pd.DataFrame:
    """
    Apply every saved head to a batch of rows and assemble a wide-format
    DataFrame of predictions.

    Reads `metadata.json` from `heads_dir`, iterates over each trained
    entry, loads the matching `head_NNNN.pt`, and runs `predict_with_head`.
    Output columns are named `{target}__pred`, `{target}__lower`,
    `{target}__upper`, `{target}__oob_flag`, `{target}__oob_severity` per target.

    Args:
        heads_dir:            Directory containing per-head .pt files and
                              metadata.json (produced by train_all_heads).
        embeddings:           (n_rows, hidden) outcome embeddings for the
                              rows to predict on. Same hidden dimension
                              as during training.
        selection_embeddings: (Optional) (n_rows, hidden) selection
                              embeddings. Required if ANY head in the
                              directory was trained with a separate
                              selection text column. Pass them whenever
                              available; heads that don't use them ignore
                              the argument cleanly.
        device:               Torch device.
        clip_bounds:          Optional (y_min, y_max) plausible support applied
                              to EVERY head, overriding each head's stored
                              train-label bounds. Use for a single global
                              domain/physical limit; leave None to use each
                              head's own saved y_min/y_max.
        flag_tol_frac:        Tolerance (fraction of support width) for the
                              per-row scale sanity check; passed through to
                              `predict_with_head`. Default 0.5.

    Returns:
        Wide-format DataFrame with one row per input row and five columns
        per target — `{target}__pred`, `{target}__lower`, `{target}__upper`
        (all clipped to support), plus `{target}__oob_flag` (bool: a pre-clip
        bound overshot support by > flag_tol_frac of its width) and
        `{target}__oob_severity` (float: normalized overshoot magnitude).
        Heads with status != 'trained' or with missing checkpoint files
        are silently skipped (no columns produced for them).

    Raises:
        FileNotFoundError: If `heads_dir/metadata.json` doesn't exist.
        ValueError:        If a saved head requires `selection_embeddings`
                           but the argument is None.
    """
    heads_dir = Path(heads_dir)
    meta_path = heads_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {heads_dir}")
    metadata = json.loads(meta_path.read_text())

    out: dict[str, np.ndarray] = {}
    for entry in tqdm(metadata, desc="Applying heads"):
        if entry.get("status") != "trained":
            continue
        head_path = heads_dir / f"head_{entry['idx']:04d}.pt"
        if not head_path.exists():
            continue
        name = entry["name"]
        pred, lower, upper, oob = predict_with_head(
            head_path, embeddings, selection_embeddings=selection_embeddings,
            device=device, clip_bounds=clip_bounds, flag_tol_frac=flag_tol_frac)
        out[f"{name}__pred"] = pred
        out[f"{name}__lower"] = lower
        out[f"{name}__upper"] = upper
        out[f"{name}__oob_flag"] = oob["flag"]
        out[f"{name}__oob_severity"] = oob["severity"]
    return pd.DataFrame(out)


# =============================================================================
# CLI
# =============================================================================

def main():
    """
    Command-line entrypoint orchestrating the full training pipeline end-to-end.

    =============================================================================
    PIPELINE EXPLANATION
    =============================================================================
    The pipeline runs in two stages, with optional behavior toggled by CLI flags.

    STAGE 1 — Embedding (build_embedding_index):
        Reads the long-form CSV at --data. Deduplicates by --id-column so each
        patient/note is embedded exactly once. Runs the Clinical ModernBERT
        encoder (or whatever --model-name points at) over the --text-column,
        max-pooling token outputs into one (hidden,) vector per id.

        Optional behaviors at this stage:
          * --question-column: prepends a question to each chunk (sequence-pair
            tokenization), with long texts split into overlapping chunks
            (controlled by --chunk-stride) and chunk embeddings combined per
            --chunk-aggregation (max or mean).
          * --selection-text-column: when --selection is also on, embeds a
            SEPARATE text column to be used by the selection equation. This
            provides a true exclusion restriction in the Heckman correction.
            The two embedding matrices are stored as
            cache/embeddings.npy and cache/selection_embeddings.npy.

        --skip-embedding lets you reuse cached embeddings, useful when
        iterating on Stage 2 hyperparameters.

    STAGE 2 — Per-target training (long_to_per_target + train_all_heads):
        The long DataFrame is grouped by --code-column. For each unique code,
        one head is trained. The pipeline takes one of two paths:

        PATH A — Vanilla CQR (default; no --selection flag):
            * Use only rows where this code was observed.
            * Three-way random split: train / val / calib.
            * Train an FFNNQuantileRegressor (depth set by --depth, width by
              --hidden) with pinball loss on (alpha/2, 0.5, 1-alpha/2) — three
              quantile levels simultaneously.
            * Fit the Conformalized Quantile Regression offset Q_hat on the
              calib set so the (1-alpha) marginal coverage guarantee holds.

        PATH B — Heckman two-step (when --selection is set):
            * Use ALL embedded rows (selected + unselected) for the selection
              equation.
            * Step 1: Train a probit selection model on (X_selection, S)
              where S = 1 if this code was observed for the id, else 0.
              When --reparam-selection is on, the selection model is
              heteroscedastic (z = mu(x) + sigma(x)*eta) trained via the
              reparameterization trick — enabling Klein-Vella identification
              through heteroscedasticity in addition to (or instead of) an
              exclusion restriction.
            * Step 2: For selected rows in each split, run the selection
              model forward and compute the inverse Mills ratio
              lambda(z) = phi(z)/Phi(z). Concatenate IMR as an extra column
              to the outcome embeddings.
            * Step 3: Train the quantile model on the augmented features
              over selected rows, with the same CQR conformal calibration
              as Path A.

        Heads that fail per-head sanity checks (too few samples, no
        variance, only one selection class, etc.) are skipped and
        recorded with a 'skipped_*' status.

    =============================================================================
    INPUTS
    =============================================================================
    Inputs come from the command line; see the module docstring for the
    full grouped CLI reference and `python ... --help` for per-flag detail.

    Critical required input:
        --data                Path to long-form CSV with one row per
                              (id, code, value) observation. Texts must
                              be constant per id; values are numeric.

    Common required-when-defaults-don't-fit:
        --id-column, --text-column, --code-column, --value-column
                              CSV column names if they differ from
                              ('id', 'text', 'code', 'value').

    Optional toggles by purpose:
        Embedding:    --model-name, --batch-size, --max-length,
                      --question-column, --chunk-stride,
                      --chunk-aggregation
        Outcome MLP:  --hidden, --depth, --max-epochs, --base-patience,
                      --val-size, --min-samples-per-head
        CQR calib:    --alpha, --calib-size, --min-calib-samples,
                      --no-calibrate
        Heckman:      --selection, --selection-text-column,
                      --selection-hidden, --selection-depth,
                      --selection-max-epochs, --reparam-selection,
                      --min-sigma
        Caching/IO:   --cache-dir, --heads-dir, --skip-embedding

    =============================================================================
    OUTPUTS (written to disk)
    =============================================================================
    All outputs are FILES written to two directories. main() itself
    returns None — its purpose is to write artifacts for later inference
    via `predict_with_head` / `predict_all_heads`.

    To --cache-dir (default: cache/):
        embeddings.npy            (n_unique_ids, hidden) float32 — outcome
                                  embeddings from the Clinical ModernBERT
                                  encoder, one row per unique id.
        selection_embeddings.npy  (n_unique_ids, hidden) float32 — written
                                  ONLY when --selection-text-column is set
                                  with --selection. Aligned row-for-row
                                  with embeddings.npy.
        id_order.json             List of ids in the order they appear in
                                  the embedding arrays. Use this to map
                                  patient ids -> embedding row indices.

    To --heads-dir (default: heads_cqr_heckman/):
        head_NNNN.pt              One PyTorch checkpoint per trained head
                                  (NNNN = zero-padded head index from
                                  metadata.json). Fields stored:
                                      quantile_state_dict, quantile_in_dim,
                                      quantile_hidden, quantile_depth,
                                      n_quantiles, quantile_levels,
                                      target_name,
                                      y_mean, y_std,
                                      Q_hat, alpha, selection_used.
                                  When selection_used=True, additionally:
                                      selection_state_dict, selection_in_dim,
                                      selection_hidden, selection_depth,
                                      selection_is_reparam,
                                      selection_min_sigma,
                                      separate_selection_embeddings.
        metadata.json             A JSON-serialized list with one entry
                                  per target, each containing:
                                      idx, name, status
                                  and, for status='trained' entries:
                                      selection_used, outcome_depth,
                                      epochs_run_quantile,
                                      best_val_pinball_standardized,
                                      y_train_mean, y_train_std,
                                      quantile_levels, calibration_status,
                                      and the calibration metrics dict
                                      from fit_cqr_offset
                                      (Q_hat, target_coverage,
                                       empirical_coverage_calib,
                                       raw_coverage_pre_cqr, etc).
                                  Heckman heads additionally include:
                                      selection_depth, selection_is_reparam,
                                      separate_selection_embeddings,
                                      epochs_run_selection,
                                      best_val_probit_nll,
                                      n_train_selection_universe,
                                      n_train_outcome_selected,
                                      n_val_outcome_selected,
                                      selection_rate_train,
                                      mean_predicted_selection_prob,
                                      mean_imr_train, std_imr_train,
                                      mean_selection_sigma,
                                      std_selection_sigma.
                                  Use this file to enumerate heads and
                                  read per-head training diagnostics
                                  without loading each .pt.

    =============================================================================
    USING THE OUTPUTS FOR INFERENCE
    =============================================================================
    After main() finishes, embed your new rows with `build_embedding_index`
    (using the same --model-name and --selection-text-column if applicable),
    then call `predict_all_heads(heads_dir, embeddings, selection_embeddings=...)`
    to produce the wide-format DataFrame of (pred, lower, upper) for every
    trained head. See those functions' docstrings for the details.
    """
    p = argparse.ArgumentParser(
        description="Clinical ModernBERT -> per-target CQR heads with optional "
                    "Heckman selection correction. See module docstring for the "
                    "full CLI reference grouped by purpose.")

    # --- Data & input ---
    p.add_argument("--data", required=True,
                   help="Path to long-form CSV with one row per (id, code, value) observation.")
    p.add_argument("--id-column", default="id",
                   help="Patient / note identifier column. Text and question must be "
                        "constant within an id. (default: id)")
    p.add_argument("--text-column", default="text",
                   help="Clinical text column (constant per id). (default: text)")
    p.add_argument("--code-column", default="code",
                   help="Column naming which target this row belongs to (e.g. ICD code, "
                        "lab name). (default: code)")
    p.add_argument("--value-column", default="value",
                   help="Observed continuous value for (id, code). NaN rows are dropped. "
                        "(default: value)")

    # --- Embedding / BERT ---
    p.add_argument("--model-name", default="Simonlee711/Clinical_ModernBERT",
                   help="HuggingFace model id for the encoder. "
                        "(default: Simonlee711/Clinical_ModernBERT)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size for the BERT forward pass. (default: 16)")
    p.add_argument("--max-length", type=int, default=512,
                   help="Maximum token length per chunk. (default: 512)")
    p.add_argument("--question-column", default=None,
                   help="Optional column whose value is prepended to every chunk during "
                        "tokenization, format: [CLS] question [SEP] text [SEP]. Must be "
                        "constant per id and non-empty when set. (default: off)")
    p.add_argument("--chunk-stride", type=int, default=0,
                   help="Token overlap between adjacent chunks of the same text. Only "
                        "used when --question-column is set. (default: 0)")
    p.add_argument("--chunk-aggregation", choices=["max", "mean"], default="max",
                   help="How to combine per-chunk embeddings into one per-id embedding. "
                        "Only used when --question-column is set. (default: max)")
    p.add_argument("--model-kind", choices=["encoder", "generative"], default="encoder",
                   help="Backbone family. 'encoder' = BERT-family (e.g. Clinical "
                        "ModernBERT, the default). 'generative' = decoder-only causal LM "
                        "(e.g. sapientinc/HRM-Text-1B). (default: encoder)")
    p.add_argument("--pooling", choices=["max", "mean", "last"], default=None,
                   help="Token-pooling strategy. Default depends on --model-kind: 'max' "
                        "for encoder, 'last' for generative. 'last' = last real token "
                        "(principled for causal models); 'max'/'mean' = over all real "
                        "tokens (suited to encoders or PrefixLM-bidirectional forwards). "
                        "(default: per model-kind)")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="Allow loading Hub-provided modeling code. Required for custom "
                        "generative architectures such as HRM-Text. (default: off)")
    p.add_argument("--prefix-lm-bidirectional", action="store_true",
                   help="(generative) Mark the whole input as a bidirectional PrefixLM "
                        "prefix (sets token_type_ids=1). Matches HRM-Text's training-time "
                        "prefix forward and makes 'mean'/'max' pooling meaningful for it. "
                        "(default: off)")
    p.add_argument("--torch-dtype", default=None,
                   help="(generative) Model dtype: 'auto' (bf16 on GPU, fp32 on CPU), "
                        "'float32', 'float16', or 'bfloat16'. (default: auto)")
    p.add_argument("--hidden", type=int, default=256,
                   help="Width of every hidden layer in the outcome MLP. (default: 256)")
    p.add_argument("--depth", type=int, default=2,
                   help="Number of hidden layers in the outcome (quantile) MLP. depth=2 "
                        "reproduces the v1 architecture. (default: 2)")
    p.add_argument("--max-epochs", type=int, default=200,
                   help="Hard cap on training epochs per head; early stopping usually "
                        "fires first. (default: 200)")
    p.add_argument("--base-patience", type=int, default=10,
                   help="Epochs without val-loss improvement before early-stopping. "
                        "Also used for the selection model when --selection is on. "
                        "(default: 10)")
    p.add_argument("--val-size", type=float, default=0.15,
                   help="Fraction of labeled rows held out per head for validation / "
                        "early stopping. (default: 0.15)")
    p.add_argument("--min-samples-per-head", type=int, default=32,
                   help="Heads with fewer observed values than this are skipped and "
                        "recorded as such in metadata.json. (default: 32)")

    # --- CQR conformal calibration ---
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Miscoverage rate. alpha=0.1 -> 90%% PI. Also sets the quantile "
                        "levels (alpha/2, 0.5, 1-alpha/2) trained against. (default: 0.1)")
    p.add_argument("--calib-size", type=float, default=0.15,
                   help="Fraction of labeled rows held out per head for the CQR "
                        "calibration step. (default: 0.15)")
    p.add_argument("--min-calib-samples", type=int, default=16,
                   help="Heads with fewer calibration samples save Q_hat=None and "
                        "produce NaN interval bounds at inference. (default: 16)")
    p.add_argument("--no-calibrate", action="store_true",
                   help="Disable the CQR conformal step. Point predictions still "
                        "produced; lower/upper bounds are NaN.")

    # --- Heckman selection correction ---
    p.add_argument("--selection", action="store_true",
                   help="Toggle the Heckman two-step path on. Without this flag the "
                        "pipeline runs as vanilla CQR.")
    p.add_argument("--selection-text-column", default=None,
                   help="Optional separate text column embedded for the SELECTION "
                        "equation only. Gives a true EXCLUSION RESTRICTION "
                        "(process/scheduling/triage text affects ordering but not "
                        "outcome value). Must be constant per id and non-empty for at "
                        "least most ids. Ignored if --selection is not set. (default: off)")
    p.add_argument("--selection-hidden", type=int, default=128,
                   help="Width of every hidden layer in the selection MLP. (default: 128)")
    p.add_argument("--selection-depth", type=int, default=2,
                   help="Number of hidden layers in the selection MLP. (default: 2)")
    p.add_argument("--selection-max-epochs", type=int, default=100,
                   help="Hard cap on epochs for the selection model. Patience reuses "
                        "--base-patience. (default: 100)")
    p.add_argument("--reparam-selection", action="store_true",
                   help="Use the heteroscedastic probit selection model: "
                        "z = mu(x) + sigma(x) * eta with eta ~ N(0, 1), sampled via "
                        "the reparameterization trick. Enables Klein-Vella (2010) "
                        "identification through heteroscedasticity even without an "
                        "exclusion restriction.")
    p.add_argument("--min-sigma", type=float, default=0.1,
                   help="Floor on sigma(x) for the reparam selection model. Keeps "
                        "gradients well-conditioned for very negative raw sigma "
                        "outputs. (default: 0.1)")

    # --- Caching & output ---
    p.add_argument("--cache-dir", default="cache",
                   help="Directory for embeddings.npy and id_order.json (and "
                        "selection_embeddings.npy when --selection-text-column is "
                        "used). (default: cache)")
    p.add_argument("--heads-dir", default="heads_cqr_heckman",
                   help="Directory for the per-head .pt files and metadata.json. "
                        "(default: heads_cqr_heckman)")
    p.add_argument("--skip-embedding", action="store_true",
                   help="Skip Stage 1 if cached embeddings already exist in "
                        "--cache-dir. Useful for iterating on Stage 2 hyperparameters "
                        "without re-running BERT.")

    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / "embeddings.npy"
    sel_emb_path = cache_dir / "selection_embeddings.npy"
    id_order_path = cache_dir / "id_order.json"

    long_df = pd.read_csv(args.data)
    required = {args.id_column, args.text_column, args.code_column, args.value_column}
    if args.question_column is not None: required.add(args.question_column)
    if args.selection_text_column is not None: required.add(args.selection_text_column)
    missing = required - set(long_df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if args.selection_text_column is not None and not args.selection:
        print(f"[warn] --selection-text-column is set but --selection is not; "
              f"selection text will be ignored.")

    sel_msg = ""
    if args.selection:
        sel_msg = "  |  SELECTION ON (Heckman)"
        if args.reparam_selection:
            sel_msg += " + reparam(het. probit)"
        if args.selection_text_column:
            sel_msg += " + separate selection text"

    print(f"[main] {len(long_df)} rows  |  {long_df[args.id_column].nunique()} ids  "
          f"|  {long_df[args.code_column].nunique()} codes{sel_msg}")

    # ---- Stage 1: embeddings ----
    use_sep_sel = args.selection and (args.selection_text_column is not None)
    if (args.skip_embedding and emb_path.exists() and id_order_path.exists()
            and (not use_sep_sel or sel_emb_path.exists())):
        print(f"[main] loading cached embeddings from {cache_dir}")
        embeddings = np.load(emb_path)
        selection_embeddings = np.load(sel_emb_path) if use_sep_sel else None
        id_order = json.loads(id_order_path.read_text())
        id_to_row = {id_val: i for i, id_val in enumerate(id_order)}
    else:
        result = build_embedding_index(
            long_df,
            id_col=args.id_column, text_col=args.text_column,
            model_name=args.model_name, question_col=args.question_column,
            selection_text_col=args.selection_text_column if args.selection else None,
            batch_size=args.batch_size, max_length=args.max_length,
            stride=args.chunk_stride, chunk_aggregation=args.chunk_aggregation,
            model_kind=args.model_kind, pooling=args.pooling,
            trust_remote_code=args.trust_remote_code,
            prefix_lm_bidirectional=args.prefix_lm_bidirectional,
            torch_dtype=args.torch_dtype)
        embeddings = result["embeddings"]
        selection_embeddings = result["selection_embeddings"]
        id_to_row = result["id_to_row"]
        id_order = result["id_order"]
        np.save(emb_path, embeddings)
        if selection_embeddings is not None:
            np.save(sel_emb_path, selection_embeddings)
        id_order_path.write_text(json.dumps(id_order, default=str))

    # ---- Build per-target ----
    per_target_data = long_to_per_target(
        long_df, embeddings=embeddings, id_to_row=id_to_row,
        id_col=args.id_column, code_col=args.code_column,
        value_col=args.value_column,
        include_selection=args.selection,
        selection_embeddings=selection_embeddings if args.selection else None)

    # ---- Stage 2: heads ----
    train_all_heads(
        per_target_data,
        out_dir=Path(args.heads_dir),
        val_size=args.val_size,
        min_samples_per_head=args.min_samples_per_head,
        base_patience=args.base_patience,
        calibrate=not args.no_calibrate,
        calib_size=args.calib_size,
        min_calib_samples=args.min_calib_samples,
        alpha=args.alpha,
        selection=args.selection,
        selection_hidden=args.selection_hidden,
        selection_depth=args.selection_depth,
        selection_max_epochs=args.selection_max_epochs,
        reparam_selection=args.reparam_selection,
        min_sigma=args.min_sigma,
        depth=args.depth,
        hidden=args.hidden,
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    main()

