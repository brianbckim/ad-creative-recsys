# 4. Environment & Tooling

This repo is intentionally “artifact-driven” and spans multiple ML stacks.
In practice, you will run:

- data prep + contracts: pandas/numpy
- retrieval teacher: PyTorch (`train_twotower.py`)
- sequence KD: TensorFlow in TF1-style graph/session mode (`din/train_listwise_kd.py`)
- copy-head + scoring: TF2/Keras (`train_copy_head.py`, `score_copies_with_din_and_copy_head.py`)
- text embeddings: scikit-learn + scipy sparse (`build_copy_embeddings.py`)

The key to a smooth setup is to separate:

- **environment correctness** (deps import and run)
- **run correctness** (artifacts come from the same index space)

## 4.1 Python Environments

### 4.1.1 Dependency baseline (`requirements.txt`)
The canonical dependency list is `requirements.txt`. It includes:

- `numpy`, `pandas`, `scikit-learn` (which pulls in `scipy`, used by `build_copy_embeddings.py`)
- `torch` (Two-Tower)
- `tensorflow` (or `tensorflow-macos` + `tensorflow-metal` on Apple Silicon)
- `keras` + `tensorboard` (copy-head)
- `openpyxl` (so `build_proxy.py` can load `.xlsx` via `pandas.read_excel`)

Important detail: the TensorFlow dependency is selected using environment markers.

- On **macOS arm64**, `tensorflow-macos==2.16.2` and `tensorflow-metal==1.2.0` are selected.
- Otherwise, `tensorflow==2.16.2` is selected.

This matters because the DIN stage is implemented in TF1-style graph/session mode (see `din/train_listwise_kd.py` using `tf.compat.v1` and `tf.disable_v2_behavior()`).

In the current code, DIN also relies on legacy `tf.layers.*` APIs (e.g., `tf.layers.batch_normalization`). That means:

- A plain “install `requirements.txt` and run everything in one environment” setup may fail for the DIN stage if your TensorFlow/Keras combination does not support `tf.layers.*` (notably, standalone Keras 3 environments).
- To run DIN reliably, you need an environment where those legacy APIs are available (for example, a TF1/Keras2-style environment, or a TF2 setup that still exposes the legacy v1 layers).

### 4.1.2 Recommended setup (single environment)
The simplest operational setup is one Python environment that can import both torch and tensorflow.

However, because the DIN stage is TF1-style graph code and uses legacy `tf.layers.*`, a single environment only works if your TensorFlow/Keras build supports those APIs. If it doesn’t, run DIN in a separate, compatible environment (see next section).

Example (venv):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Sanity checks (fast):

```bash
python -c "import numpy, pandas, sklearn, torch; import tensorflow as tf; print('torch', torch.__version__); print('tf', tf.__version__)"
```

### 4.1.3 Optional setup (separate environments)
If you run into dependency constraints (common when mixing PyTorch + TF + platform-specific builds), you *can* split environments by stage because boundaries are file-based:

- Env A: data prep + Two-Tower (pandas/numpy/torch)
- Env B: DIN KD + embedding export (tensorflow)
- Env C: ad copy scoring (tensorflow/keras)

This repo’s scripts communicate via CSV/JSONL/PKL/checkpoints, so cross-environment execution is valid as long as you keep the artifact bundle consistent (see Section 3.4).

### 4.1.4 TF “v1 graph mode” reality
Several scripts inside `din/` explicitly run TF in graph/session mode (`tf.compat.v1`, `tf.Session`). This has two practical implications:

- You should treat these scripts as “TF1-style programs” even when running in a TF2-compatible environment.
- Most debugging will look like TF1 debugging (graph resets, sessions, checkpoints), not eager-mode TF2 debugging.

## 4.2 Hardware Assumptions

This pipeline is designed to run end-to-end without a hard GPU requirement, but not every stage benefits equally from acceleration.

### 4.2.1 CPU-only is viable (with caveats)
- **Proxy / conversion / dataset builders** are pandas-heavy and typically CPU-bound.
- **Text embedding (TF-IDF + SVD)** can be memory-heavy because it builds a sparse TF-IDF matrix before SVD.
- **Ad copy scoring (copy-head + scoring)** intentionally disables GPU in code (see `tf.config.experimental.set_visible_devices([], "GPU")` in `train_copy_head.py` and `score_copies_with_din_and_copy_head.py`). Even on a machine with a GPU, those stages will run on CPU unless you change the scripts.

### 4.2.2 Two-Tower device selection (PyTorch)
`train_twotower.py` selects its device dynamically:

- `cuda` if available
- else `mps` if available (Apple Silicon)
- else `cpu`

It also supports multi-GPU `DataParallel` only when CUDA is available and `--dataparallel` is used.

Practical reading:

- If you have a CUDA GPU, Two-Tower training and candidate dumping can benefit.
- On Apple Silicon, MPS may accelerate the Two-Tower stage.
- If you are CPU-only, you can still run Two-Tower, but you may want to reduce training epochs and/or evaluation cost.

### 4.2.3 Evaluation cost guardrails
Two-Tower evaluation can be configured to score against all items (`--eval_all_items`) or to do sampled evaluation.

Because “all-items eval” can become very expensive for large catalogs, `train_twotower.py` includes a guard:

- if `n_items > --eval_all_items_max_guard` (default `200000`), it automatically switches to sampled eval.

This is a practical hardware constraint: even if your training runs fine, evaluation may become the bottleneck if you insist on all-items evaluation with a large item universe.

## 4.3 CLI Patterns & Flags

All top-level scripts use `argparse`. The most reliable workflow is:

1) run `python <script>.py --help`
2) start from defaults and only override paths/flags you need
3) keep outputs in a per-run directory so artifacts never mix

### 4.3.1 Common patterns across scripts
- **Inputs are explicit paths** (often `--src`, `--*_csv`, `--*_pkl`, `--ckpt`).
- **Outputs are explicit paths** (often `--out`, `--out_dir`, `--out_*`).
- **Optional artifacts are flag-controlled** (e.g., `--save_metrics`, `--out_csv`).
- **Seeds exist where sampling occurs** (`--seed` in conversion, Two-Tower training, DIN KD training, copy-head dataset builder, copy-head training).
	- Scoring is deterministic given fixed artifacts (it uses stable ordering + an explicit tie-break), so it does not require a seed.
	- Even with seeds, you should treat results as “reproducible at the artifact level” rather than bitwise identical across machines.

### 4.3.2 Stage-specific flags that matter most

**Proxy builder (`build_proxy.py`)**
- `--src`: Talkwalker export (CSV/XLSX)
- `--out`: output directory (default `./proxy`)
- labeling / signal shaping:
	- `--min_engagement_for_pos`, `--label_quantile`, `--label_groupby`
	- `--composite_label_quantile`, `--weights`, `--half_life_days`
- pseudo-user controls:
	- `--pseudo_user_mode` (`cohort|session|mixed`)
	- `--pseudo_user_k_max`, `--pseudo_user_target_hist`, `--session_gap_minutes`, `--session_domains`
- provenance:
	- quality report is on by default; disable with `--no_quality_report`
	- `--auto_tune` optionally writes an auto-tune report

**Index + Two-Tower dataset conversion (`convert_to_twotower.py`)**
- `--interactions`, `--items`: proxy CSVs
- `--label_col`: choose `label` or `label_ctr_proxy`
- split / sampling:
	- `--time_holdout`, `--per_user_split`, `--min_events_per_user`, `--neg_per_pos`, `--seed`
- `--auto_config` (`off|soft|hard`) optionally writes:
	- `twotower_autoconfig.json`
	- `train_suggest.json`

**Two-Tower trainer and candidate dump (`train_twotower.py`)**
- required inputs: `--train_pairs`, `--val_pairs`, `--user_map`, `--item_map`
- optional item feature usage: `--item_catalog` + `--use_item_features`
- saving / outputs:
	- `--out_dir` (default `./twotower_runs`)
	- `--save_best`, `--save_final`, `--save_metrics`
- teacher slate dump (for DIN KD):
	- `--dump_candidates_k > 0` produces `tt_candidates_topK.jsonl` at `--dump_candidates_out`
	- `--dump_score_type` chooses `dot` vs `sigmoid`
- optional embedding export:
	- `--export_embeddings` writes `user_emb.npy`, `item_emb.npy` and also re-exports maps into `--out_dir`

**DIN dataset builder (`make_din_dataset.py`)**
- required inputs:
	- `--user_map`, `--item_map` (used to map string IDs to dense indices)
- the script can build `din/dataset.pkl` from multiple input styles:
	- pairs path: `--pairs_csv_train` + `--pairs_csv_val` (typically `twotower_*_pairs.csv`)
	- single pairs path: `--pairs_csv` (a single CSV with `user_idx,item_idx,label`; the script splits internally)
	- proxy JSONL path: `--proxy_jsonl` (from `build_proxy.py`)
- theme/cate source:
	- `--item_catalog` + `--cate_col` + `--theme_col`
- output:
	- `--out` (default `./din/dataset.pkl`)
	- `--format` (`legacy` vs `std`)

**DIN listwise KD (`din/train_listwise_kd.py`)**
- required inputs:
	- `--dataset_pkl`, `--candidates`, `--user_map`, `--item_map`
- optional: `--proxy_jsonl` (if provided, positives are derived from that file; otherwise from `dataset.pkl`)
- key training knobs:
	- `--K` (candidate slate size), `--tau` (softmax temperature for KD distribution)
	- `--lambda_kd` (KD vs CE mix), `--point_loss_weight` (how much original pointwise DIN loss matters)
- output:
	- `--out_dir` (default `./din_kd_runs`)
	- saves `din_kd_best.ckpt` when training total loss improves

**DIN embedding export (`din/export_user_theme_embeddings.py`)**
- `--ckpt`: checkpoint path
- `--dataset_pkl`: used to rebuild graph shapes
- `--out_user_csv`, `--out_theme_csv`: embedding table exports

**Copy embeddings (`build_copy_embeddings.py`)**
- can embed *any* CSV containing an ID column + a text column:
	- `--catalog_csv`, `--id_col`, `--text_col`
- `--mode`:
	- `fit`: learns artifacts and writes embeddings
	- `transform`: reuses artifacts and writes embeddings for new rows
- `--artifact_prefix` must match across fit/transform if you want a shared text space.

**Item→copy mapping report (`report_item_to_copy_mapping.py`)**
- `--out_mapping_csv` is optional; without it the script prints statistics only.

**Ad copy scoring sample builder (`build_copy_head_dataset.py`)**
- mapping strategy:
	- `--item_to_copy=nearest_text` (requires `--item_embeddings_csv` and `--copy_emb_csv`; recommended default)
	- `--item_to_copy=direct` (requires `--copy_catalog_csv` with a mapping column; use only when ID universe match is verified)
	- `--item_to_copy_map_csv` can provide a precomputed mapping (and `--min_sim`, `--min_gap` can filter it)
- positive selection uses OR logic across thresholds:
	- `--label_column` + `--label_positive_value`
	- `--engagement_column` + `--min_engagement`
- sampling:
	- `--num_neg_per_pos`, `--max_pos_samples`, `--seed`
- output:
	- `--out_csv` (default `copy_train_samples.csv`)

**Ad copy scoring training (`train_copy_head.py`)**
- `--train_samples_csv` is required
- sample-weight controls:
	- `--sample_weight_col` (optional), `--weight_transform`, `--disable_weight_normalize`
- output:
	- `--out_model` (default `copy_head.keras`)

**Global scoring (`score_copies_with_din_and_copy_head.py`)**
- output CSV is optional:
	- `--out_csv` controls whether a ranking CSV is written
- theme weighting:
	- `--theme_weight_temperature` applies a power transform to per-theme weights
	- `--theme_uniform_mix` blends in a uniform prior

## 4.4 Logs & Provenance

Because this pipeline is file-contract driven, “what happened in the run” is best captured as a combination of:

1) the artifacts themselves (CSV/JSONL/PKL/checkpoints)
2) lightweight JSON reports emitted by some stages
3) stdout logs of each script invocation

### 4.4.1 Built-in provenance artifacts (code-defined)
- Proxy stage (`build_proxy.py`):
	- `quality_<tag>.json` is written unless `--no_quality_report` is set
	- `auto_tune_report_<tag>.json` is written when `--auto_tune` is used
- Conversion stage (`convert_to_twotower.py`):
	- `twotower_autoconfig.json` + `train_suggest.json` are written when `--auto_config` is `soft` or `hard`
- Two-Tower stage (`train_twotower.py`):
	- `metrics.jsonl` is written when `--save_metrics` is set
	- teacher slates are written when `--dump_candidates_k > 0`
- DIN KD stage (`din/train_listwise_kd.py`):
	- `din_kd_best.ckpt` is saved when training `total_loss` improves

### 4.4.2 Practical logging pattern
Most scripts print `[write] ...` / `[save] ...` / `[info] ...` lines. Treat stdout as an official part of provenance.

For example, for each stage you can capture logs via:

```bash
python build_proxy.py ... 2>&1 | tee runs/<tag>/logs/01_build_proxy.log
```

Even if you choose separate Python environments per stage, keeping these per-stage log files alongside artifacts makes later debugging (contract mismatches, unexpected drops, missing IDs) dramatically easier.

### 4.4.3 What “provenance” means in this repo
Given the proxy nature of labels and the dense-index joins, provenance should answer:

- Which exact input export produced the proxy tables?
- Which pseudo-user strategy and labeling knobs were used?
- Which index spaces (`user_map.csv`, `item_map.csv`, `theme_list`) are the source-of-truth?
- Which teacher slate file was used for KD?
- Which embedding tables (and artifact prefixes) were used to build copy-head samples and rankings?

If you can answer those questions for a run, you can usually reproduce results or diagnose why a run diverged.

