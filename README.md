# HydraLab 
<img src="hydralab.png" align="right" width="225" alt="">

This README.md is intended for additional context to HydraLab and as a tutorial as to how to implement HydraLab. 

**Background**

**HydraLab is a method and tool to estimate simulataneous Conformalized Quantile Regression (CQR) heads with 95% Prediction Intervals (PIs) using a masked language model** for predicting continuous lab values and related measurements. These were trained using the ground truth of OMOP `measurement` domain values (LOINC-coded laboratory tests) from clinical free text derived from structured OMOP table data that is transformed into free-text clinical note like structured.

**NIH Clinical and Translational Science Award (CTSA) Acknowledgement:**
We are grateful to NIH CTSA program for support of this project through
the Consortium for Translational & Precision Health (CTPH) (NIH grant number UM1TR004539). The [CTPH](https://www.ctph-texas.org/) is a partnership between [Baylor College of Medicine](https://www.bcm.edu/) and the [Univerisity of Houston](https://www.uh.edu/). Please visit the [CTPH](https://www.ctph-texas.org/) for more information.  

**All-of-Us Acknowledgement:**
We gratefully acknowledge All of Us participants for their contributions, without whom this research would not have been possible. We also thank the National Institutes of Health’s All of Us Research Program for making available the participant data used to train the CQR heads in this project.”

This study used data from the All of Us Research Program’s Controlled Tier Dataset [CDR-8], available to authorized users on the Researcher Workbench.”

For more information about All-of-Us, please visit the [All of Us Research Hub](https://researchallofus.org/).

## Brief Description of Approach

A clinical encoder ([Clinical ModernBERT](https://huggingface.co/Simonlee711/Clinical_ModernBERT) transforms each patient's text history into a single embedding. Many small feed-forward "heads" (one per LOINC code, hence *Hydra*) read that shared embedding and each predict a lower, median, and upper quantile for its lab. Each head is then conformally calibrated so its interval reaches the target coverage (95% by default).

This repository ships a set of pre-trained reference heads and a three-notebook tutorial that walks through generating data, loading encoder weights, and updating the heads on new data, including automatically adding heads for LOINC codes the reference set has never seen.

---

## Repository layout

```
hydralab_cqr/
├── src/
│   ├── module_hydralab_cqr.py   # embedding, head training, CQR calibration, prediction
│   ├── module_hydralab_cqr_finetune.py  # fine-tuning via new-head growth, interval-score evaluation
│   └── module_structured_to_unstructured_rebuild.py# structured to text
├── heads/
│   └── clinical_modernbert_3/  # reference heads: depth 3, hidden width 256, alpha = 0.05
│       ├── head_NNNN.pt  # one checkpoint per LOINC head
│       ├── metadata.json # per-head name, status, y-scaling, Q_hat, calibration diagnostics
│       └── heads_depth_3_updated/ # output of Step 3 (fine-tuned + newly grown heads)
└── workflow_to_update_hydralab_cqr_example/
    ├── step1_simulated_aou_omop_quantile_data_generation.ipynb
    ├── step2_load_language_model_weights.ipynb
    ├── step3_finetune_cqr_single_model.ipynb
    └── aou_synthetic_output/  # Step 1 outputs + (cached) Step 3 embeddings
```

The head set contains 504 metadata entries: 500 trained heads and 4 skipped because their training labels had no variance in the All-of-Us trianing cohort (`status: skipped_no_variance`). The heads are retained nonetheless as placeholders for updated training in datasets HydraLab is applied to. 

---

## Setup

```bash
git clone https://github.com/thomtaylorbcm/hydralab_cqr.git
cd hydralab_cqr
pip install numpy pandas scipy scikit-learn matplotlib torch transformers accelerate tqdm rich ipywidgets
```

The notebooks use absolute paths of the form `/home/jupyter/repos/hydralab_cqr/...`. Before running, replace that prefix with the location of your clone in these cells:

| Notebook | *Variables you will need to edit* |
|---|---|
| Step 1 | `module_path`, `os.chdir(...)` |
| Step 3 | `module_path`, `BASE_DIR`, `DATA`, `CACHE_DIR` |

### Resource Constraining Considerations
**CPU-only machines.** If `transformers` fails to load a model because of a CUDA-built `torch`, reinstall the CPU wheels (the commented cell near the top of Step 3 does this):

```bash
pip uninstall torch torchaudio torchvision -y
pip install torch torchaudio torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

---

## Workflow example (using synthetic OMOP data generation)

Run the notebooks in order from `workflow_to_update_hydralab_cqr_example/`. Step 1 outputs and the Step 3 embeddings are already committed in `aou_synthetic_output/`, so each step can also be run on its own.

```
Steps at a high level:

Step 1: synthetic OMOP data to long-form (id, code, question, text, value) train / holdout CSVs (this is the expected structure of the input data); this is intended as a template for constructing new data sources.

Step 2: download encoder weights and cache locally, with simple test cells to see that text can be tokenized and input-output occurs as expected. The Clinical ModernBERT model (first in notebook) is the selected model (see paper) that the each HydraLab expects to feed the max-pooled hidden state features (k=768).

Step 3: tokenize and embed text and fine-tune existing heads AND grow new heads on LOINCs not matched to the metadata JSON heads already trained. You If there are new LOINCs in your data, HydraLab will train new heads on these models unless there is invariance in the observed values.
```

### Step 1: `step1_simulated_aou_omop_quantile_data_generation.ipynb`

**Purpose.** Builds a fully synthetic, reproducible training corpus that mimics *All of Us* data in the OMOP Common Data Model, with future lab values that are deliberately learnable from the patient's earlier text. *No real participant data are used.*

1. Draws a hidden phenotype per participant (adiposity, alcohol, smoking, activity, diet, diabetes, MASLD, hyperlipidemia, hypertension, medications) and generates OMOP-style `person`, `visit_occurrence`, `condition_occurrence`, `drug_exposure`, `measurement`, and `observation` tables (3 to 7 outpatient visits each; Participant Provided Information (PPI)-style survey answers typically collected at All-of-Us program entry).
2. Simulates five target labs at the **final** visit only, with heteroscedastic noise and phenotype-dependent ordering (labs are missing-not-at-random): VLDL-C `2091-7`, triglycerides `3043-7`, HbA1c `4548-4`, ALT `1742-6`, AST `1920-8`. This mimics having *both* existing and new LOINCs that will have heads fine-tuned (for existing LOINCs previously trained) *and* new heads for new LOINCs not previously observed in the All-of-Us training cohort. 
3. Renders a readable note for every encounter (`Name [VOCAB CODE]` for each item).
4. Uses the STU module to build each patient's cumulative context text up to the **second-to-last** encounter, so target labs never leak into the input.
5. Reshapes to one row per patient × measured lab and splits **by patient** (no patient appears in both partitions).
6. Audits the design: Spearman correlations against the hidden phenotype to ensure an association can be learned from the embedding features with these synthetic data.


**How to run.** Edit the paths, optionally change the parameters, then *Run All*.

| Parameter | Default | Meaning |
|---|---|---|
| `RNG_SEED` | `77030` | Data-generation seed used |
| `N_PATIENTS` | `1000` | *N* Synthetic participants to generate data for |
| `SPLIT_SEED` | `57006` | Train/holdout split seed (independent of `RNG_SEED`) |
| `TEST_SIZE` | `0.20` | Holdout fraction of patients |

**Outputs** (`aou_synthetic_output/`): the six OMOP tables, `encounter_notes.csv`, `stu_context_by_patient.csv`, `concept_dictionary.csv`, `text_to_future_labs.jsonl`, and the two files used downstream:

| File | Rows (defaults) | Columns |
|---|---|---|
| `quant_stack_train_test_labdf.csv` | 2,750 (775 patients) | `id, code, question, text, value` |
| `quant_holdout_labdf.csv` | 704 (194 patients) | `id, code, question, text, value` |

Reload them with `pd.read_csv(path, index_col=0)`.

> LOINC, SNOMED, RxNorm, and OMOP visit/gender codes are real. The integer `concept_id`s are illustrative placeholders; in a production *All of Us* build they map to [OHDSI Athena](https://athena.ohdsi.org/) concept IDs.

### Step 2: `step2_load_language_model_weights.ipynb`

**Purpose.** Downloads candidate encoder/decoder weights from HuggingFace into the local cache and confirms each one produces a hidden-state embedding. Step 3 runs with `TRANSFORMERS_OFFLINE="1"`, so the encoder it uses must already be cached.

*This step assumes you have access to HuggingFace Hub in the implementation environment. If not, you can download the GGUF open-weights and upload these weights if permitted by your IT environment. This is a common situation in many computational environments. We do not detail this approach here because of the diversity of different security layers and installation requirements that different operaitonal, IT, and regulatory environments may impose within given institution's computational infrastructure.* 

**Models referenced.**

| Model | Type | Embedding demo | Notes |
|---|---|---|---|
| [`Simonlee711/Clinical_ModernBERT`](https://huggingface.co/Simonlee711/Clinical_ModernBERT) | Encoder | `[CLS]`, 768-d | *Best performing in All-of-Us data with 95% CQR heads* |
| [`thomas-sounack/BioClinical-ModernBERT-base`](https://huggingface.co/thomas-sounack/BioClinical-ModernBERT-base) | Encoder | `[CLS]`, 768-d | Alternative clinical encoder |
| [`medicalai/ClinicalBERT`](https://huggingface.co/medicalai/ClinicalBERT) | Encoder | `[CLS]`, 768-d | Clinical EHR text |
| [`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) | Encoder | `[CLS]`, 768-d | BioBERT further trained on MIMIC-III notes |
| [`google/gemma-3-270m`](https://huggingface.co/google/gemma-3-270m) | Decoder | last token, 640-d | Gated; accept the license on HuggingFace |
| `sapientinc/HRM-Text-1B`, `google/medgemma-4b-pt` | Decoder | last token | Cells commented out |


**Running this notebook: **
1. Keep `os.environ["TRANSFORMERS_OFFLINE"] = "0"` so downloads are allowed (assuming you can access HuggingFace, download GGUFs locally and upload to your environment manually if not).
2. Set your token in the second code cell, which is a placeholder: `access_token = "hf_..."` (use `None` for public models).
3. Run the Clinical ModernBERT cells at minimum. The other sections are optional comparisons (see paper for model evaluation)

> The supplied heads are tied to the embedding space they were trained on (Clinical ModernBERT, 768-d, max pooling, question conditioning). Swapping in a different backbone requires a different set of heads. If this is desired, please contact us (thomas.taylor@bcm.edu)

### Step 3: `step3_finetune_cqr_single_model.ipynb`

**Purpose.** Shows how to bring new data into an existing head set: every head with enough new observations is fine-tuned from its saved multilayer perceptron weights and re-calibrated. Note that heads without new data are copied through unchanged. **Importantly, any LOINC code in the new data with no existing head has a *new* head trained from scratch. The result is written in the same checkpoint format, so it loads directly with `predict_all_heads`.** This allows HydraLab to grow and is the sentiment behind the naming of this methodology and tool referencing freshwater Hydra's that grow new tentacles. 

**Fine-tuning (updating) and new training pipeline: **
1. Loads the Step 1 train (used here as the *update* cohort) and holdout files. [*Replace these synthetic generated data with real data.*]

2. **Embeds** each patient once with `mh.build_embedding_index` (Clinical ModernBERT, encoder, `pooling="max"`, `chunk_aggregation="max"`, question-conditioned, `max_length=512`) and caches the arrays to `aou_synthetic_output/`.

3. **Updates** heads with `ft.finetune_all_heads`:
   - existing heads: warm start at a low learning rate with early stopping, stored `y_mean`/`y_std` held fixed, CQR offset $\hat{Q}$ re-fit on a fresh calibration split, support bounds widened to cover the new data;
   - new codes: trained from scratch with the same depth and width as the existing set.

4. Reports fine-tuning diagnostics (validation pinball loss and $\hat{Q}$ before/after), head-set growth, and the mean absolute change in first-layer weights.

5. **Evaluates** with `ft.evaluate_heads_on_holdout` (coverage, interval width, interval score) and saves `initial_vs_updated_holdout_eval.csv`.

With the committed synthetic data, the three codes already in the reference set (`2091-7`, `3043-7`, `4548-4`) are fine-tuned, and two ***new*** heads are grown for ALT `1742-6` and AST `1920-8`.

#### How to Run the Notebook:
1. Edit the paths (`module_path`, `BASE_DIR`, `DATA`, `CACHE_DIR`).
2. Make sure Step 2 has cached `Simonlee711/Clinical_ModernBERT`, or set `TRANSFORMERS_OFFLINE` to `"0"`. Set this variable before `transformers` is first imported for it to take effect.
3. Choose how to get embeddings:
   - **Compute:** run the `build_embedding_index` and cache-writing cells (about 20 minutes on CPU for 969 patients; much, much faster on most GPUs if available, including edge-device GPUs).
   - **Reuse:** skip those cells and run the two *reload cached embeddings* cells, which read the `.npy` / `.pkl` / `.json` files already in `aou_synthetic_output/`.
4. Run the remaining cells.

| Parameter | Default | Meaning |
|---|---|---|
| `MODEL_NAME` | `Simonlee711/Clinical_ModernBERT` | Encoder used for embeddings |
| `MAX_LENGTH` / `BATCH_SIZE` | `512` / `68` | Tokens per chunk / chunks per batch (lower `BATCH_SIZE` for small VRAM) |
| `ALPHA` | `0.05` | Miscoverage; quantiles $(\alpha/2,\ 0.5,\ 1-\alpha/2)$ = (0.025, 0.5, 0.975). Must match the heads |
| `FT_LR` | `1e-4` | Warm-start learning rate |
| `FT_MAX_EPOCHS` / `FT_PATIENCE` | `1000` / `10` | Epoch cap / early-stopping patience |
| `FT_MIN_SAMPLES` | `8` | Minimum new rows to fine-tune an ***existing*** head |
| `FT_FREEZE_HIDDEN` | `False` | `True` updates only the output layer (useful for very small updates) |
| `NEW_HEAD_MINSAMPLE` | `32` | Minimum new rows to grow a ***new*** head |
| `MIN_CALIB_SAMPLES` | `16` | Minimum calibration rows to re-fit $\hat{Q}$ |

**Outputs** (`heads/clinical_modernbert_3/heads_depth_3_updated/`): `head_NNNN.pt` for every head, `metadata.json` (existing entries gain `finetuned`, `n_finetune_obs`, `val_pinball_before/after`, `Q_hat_before/after`; new entries carry `newly_trained: true`), and `initial_vs_updated_holdout_eval.csv`.

> `ft.evaluate_heads_on_holdout(heads_dir, long_df, embeddings, id_order, alpha)` scores any head directory against any long-form set. For a like-for-like comparison between two head sets, pass both the same holdout rows and embeddings, and restrict to the codes both sets cover.

---

## Inference:  Predicting new data with a head set

```python
import sys
from pathlib import Path

import pandas as pd

sys.path.append("/path/to/hydralab_cqr/src")
import module_hydralab_cqr as mh


def predict_labs_from_text(long_df, heads_dir, model_name="Simonlee711/Clinical_ModernBERT",
                           batch_size=16, device="cpu"):
    """
    Predict calibrated lab-value intervals for every patient in a long-form table.

    Purpose
    -------
    Embeds each patient's text once with the same settings used to train the
    reference heads, then applies every trained head in `heads_dir`.

    Parameters
    ----------
    long_df : pandas.DataFrame
        Long-form table with columns `id` (patient identifier), `text`
        (cumulative clinical context, constant within an id) and `question`
        (natural-language prompt naming the lab; constant per id is assumed).
    heads_dir : str or pathlib.Path
        Directory containing `head_NNNN.pt` files and `metadata.json`
        (e.g. "heads/clinical_modernbert_3").
    model_name : str, optional
        HuggingFace encoder whose embedding space the heads were trained in.
    batch_size : int, optional
        Chunks per forward pass during embedding.
    device : str, optional
        Torch device, "cpu" or "cuda".

    Returns
    -------
    pandas.DataFrame
        One row per unique `id`. For each LOINC head: `{code}__pred` (median),
        `{code}__lower` and `{code}__upper` (conformal interval bounds, clipped
        to the head's plausible support), `{code}__oob_flag` (bool, a pre-clip
        bound overshot the support) and `{code}__oob_severity` (float).

    Notes
    -----
    Interval for each head: [q_lo(x) - Q_hat, q_hi(x) + Q_hat]
    (Romano, Patterson & Candes, NeurIPS 2019).
    """
    emb = mh.build_embedding_index(
        long_df, id_col="id", text_col="text", question_col="question",
        model_name=model_name, model_kind="encoder", pooling="max",
        chunk_aggregation="max", max_length=512, stride=0,
        batch_size=batch_size, device=device)
    preds = mh.predict_all_heads(Path(heads_dir), emb["embeddings"], device=device)
    preds.insert(0, "id", list(emb["id_order"]))
    return preds
```

---

## HydraLab Methodological Considerations

The key consideration in this methodological approach is two-fold

1. Text-regression frameworks allow us to incorporate all patient specific context. This allows for very patient-specific (personalized prediction) and ***also** leverages a language model to address heterogeneity and noise without requiring resource intensive time from an analyst, data engineering team, or biostatistician time to wrangle and clean refined cohorts with little ability to be adapted to future target environments. This renders translational science of tools and methods more difficult. Language models with text-regression can potentially side-step this challenge is the rationale for why it is implemented here. 
2. Clinical lab measurements (OMOP measurements table) are heterogenous and may cary extensive heteroskedasticity. This makes Gaussian methodologies (e.g., regression heads) much lest robust to this variation. The Conformalized Quantile Regresison approach is an advancement in the past few years that allows for better Prediction Interval converage of predictions than the more common and familir Gaussian linear regression approach. 

### Methodological Approach to HydraLab MLP training

**Quantile heads.** Each head is an MultiLayer Perceptron (MLP) ($\text{Linear} \rightarrow \text{GELU} \rightarrow \text{Dropout}$, repeated `depth` times, then a linear output) trained on standardized targets with the multi-quantile pinball loss appropriate for quantile regression (Koenker & Bassett, 1978):

$$
L_\tau(y, q) = \max\big(\tau\,(y - q),\ (\tau - 1)\,(y - q)\big), \qquad \tau \in \{\alpha/2,\ 0.5,\ 1 - \alpha/2\}
$$

Predictions are de-standardized with the head's stored `y_mean` and `y_std`, sorted to remove quantile crossing, and clipped to the head's observed support (i.e., the observed range of the training set).

**Conformal calibration** (Romano, Patterson & Candès, 2019). On a held-out calibration split of size $n$:

$$
E_i = \max\big(\hat{q}_{lo}(x_i) - y_i,\ y_i - \hat{q}_{hi}(x_i)\big), \qquad
\hat{Q} = \text{the } \lceil (n+1)(1-\alpha) \rceil / n \text{ empirical quantile of } \{E_i\}
$$

$$
C(x) = \big[\hat{q}_{lo}(x) - \hat{Q},\ \hat{q}_{hi}(x) + \hat{Q}\big], \qquad \Pr\big(Y \in C(X)\big) \ge 1 - \alpha \ \text{(under exchangeability)}
$$

$\hat{Q}$ can be negative, which shrinks an over-covering interval.

**Interval Score (IS)** (Winkler, 1972; Gneiting & Raftery, 2007) is used to evaluate lower is better. This is akin to RMSE if one were using a linear regression framework.

$$
IS_\alpha(l, u; y) = (u - l) + \frac{2}{\alpha}(l - y)\,\mathbb{1}\{y < l\} + \frac{2}{\alpha}(y - u)\,\mathbb{1}\{y > u\}
$$

---

## References

**Methods**
- Romano Y, Patterson E, Candès EJ. Conformalized quantile regression. *Advances in Neural Information Processing Systems* 32 (NeurIPS 2019). arXiv:1905.03222.
- Koenker R, Bassett G. Regression quantiles. *Econometrica.* 1978;46(1):33–50.
- Gneiting T, Raftery AE. Strictly proper scoring rules, prediction, and estimation. *J Am Stat Assoc.* 2007;102(477):359–378.
- Winkler RL. A decision-theoretic approach to interval estimation. *J Am Stat Assoc.* 1972;67(337):187–191.

**Language models referenced**
- Lee SA, Wu A, Chiang JN. Clinical ModernBERT: An efficient and long context encoder for biomedical text. arXiv:2504.03964 (2025).
- Sounack T, et al. BioClinical ModernBERT: A state-of-the-art long-context encoder for biomedical and clinical NLP. arXiv:2506.10896 (2025).
- Alsentzer E, et al. Publicly available clinical BERT embeddings. *Proceedings of the 2nd Clinical Natural Language Processing Workshop.* 2019:72–78.

**All-of-Us Training and Evaluation Data References**
- The All of Us Research Program Investigators. The "All of Us" Research Program. *N Engl J Med.* 2019;381(7):668–676.
- Hripcsak G, et al. Observational Health Data Sciences and Informatics (OHDSI): opportunities for observational researchers. *Stud Health Technol Inform.* 2015;216:574–578. [OMOP CDM documentation](https://ohdsi.github.io/CommonDataModel/).

---
## For additional questions

Please contact: 

thomas.taylor@bcm.edu 


## License

Apache License 2.0. See [LICENSE](LICENSE).
