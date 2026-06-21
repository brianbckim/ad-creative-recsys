# 5. End-to-End Pipeline
## Stage 0 — Preconditions
This section is a “happy path” walkthrough that you can run end-to-end.

The pipeline is **artifact-driven**: every stage reads/writes files, not in-memory calls.
That is what makes it reproducible — and also what makes it easy to break by mixing artifacts from different runs.

To make the commands copy-pastable, we will assume you keep everything for a run under a single tag directory.

### 0.1 Inputs you need

1) **Talkwalker export** (or any similarly structured external observation export)
	- Supported formats: `.csv` or `.xlsx`.
	- This is *not* a user log. It is observation data (text + engagement-like metrics).

2) **Copy catalog** (`copy_catalog.csv`)
	- Minimal schema by default:
		- `copy_id` (int)
		- `copy_text` (string)
	- If your columns differ, you can pass `--id_col/--text_col` in `build_copy_embeddings.py` and
	  `--copy_catalog_id_col/--copy_catalog_text_col` in `score_copies_with_din_and_copy_head.py`.

### 0.2 Recommended run directory layout

The scripts have defaults like `./proxy/`, `./twotower_data/`, etc.
Those defaults are fine for experimentation, but they make it too easy to mix runs.

Instead, use a per-run directory:

```bash
TAG=$(date +%Y%m%d_%H%M%S)
RUN_DIR=runs/$TAG
mkdir -p $RUN_DIR/{inputs,logs,proxy,twotower_data,twotower_runs,din,din_kd_runs,copy}
```

Put your source files under `runs/<tag>/inputs/` (or point the scripts to wherever you keep them).

### 0.3 One important mental model (before running anything)

- `build_proxy.py` creates **string IDs**: `user_id` and `item_id`.
	- Here, “user” is usually a **pseudo-user** (cohort/session), not a real person.
- `convert_to_twotower.py` freezes **dense integer index spaces**:
	- `user_idx` and `item_idx` (these become embedding row indices).
- DIN + copy-head stages operate on **embedding tables keyed by dense indices**.
- Labels are **proxy signals** (weak supervision). Names like `label_ctr_proxy` are *not* a claim about real click CTR.

If you keep those distinctions clear, the pipeline becomes much easier to reason about.

## Stage 1 — Proxy → Two-Tower
Stage 1 converts messy external exports into a stable “interaction-like” contract, then trains a Two-Tower teacher and dumps Top-K teacher slates.

### 1.1 Build proxy contracts (`build_proxy.py`)

**Goal:** create stable `item_id`, pseudo `user_id`, proxy features, proxy labels, and a DIN-friendly JSONL with histories.

Minimal command:

```bash
python build_proxy.py \
	--src $RUN_DIR/inputs/talkwalker_export.xlsx \
	--out $RUN_DIR/proxy
```

What this writes (filenames include an internal timestamp tag):
- `proxy_items_<tag>.csv`
- `proxy_interactions_<tag>.csv`
- `proxy_din_interactions_<tag>.jsonl`
- `quality_<tag>.json` (written by default; disable with `--no_quality_report`)
- `auto_tune_report_<tag>.json` (only if `--auto_tune` is used)

Common knobs you may actually want to touch:

- Pseudo-user strategy
	- `--pseudo_user_mode cohort|session|mixed` (default `cohort`)
	- `--session_gap_minutes` (default `360`)
	- `--session_domains` (default empty; if set, `mixed` can use session only for those domains)

- Positive labeling
	- `--min_engagement_for_pos` (default `1`) creates `label` from engagement totals.
	- `--label_quantile` optionally creates `label` by quantile over non-zero engagements.
	- `--label_groupby none|source_type|domain|lang` controls whether thresholds are group-normalized.

- Composite proxy label
	- `label_ctr_proxy` is derived from `proxy_ctr_composite` quantiles (controlled by `--composite_label_quantile`, default `0.7`).
	- Even though the name contains “CTR”, it is a composite proxy signal (volume/eng_rate/sent/recency), not a click CTR.

Sanity checks right after Stage 1.1:

```bash
ls -lh $RUN_DIR/proxy | head

PROXY_INTER=$(ls -1t $RUN_DIR/proxy/proxy_interactions_*.csv | head -1)
echo "Using: $PROXY_INTER"

# Inspect basic schema
python - "$PROXY_INTER" << 'PY'
import sys
import pandas as pd

p = sys.argv[1]
df = pd.read_csv(p)
print('rows', len(df))
print('label mean', df['label'].mean() if 'label' in df else None)
print('label_ctr_proxy mean', df['label_ctr_proxy'].mean() if 'label_ctr_proxy' in df else None)
PY
```

### 1.2 Convert proxy contracts into Two-Tower contracts (`convert_to_twotower.py`)

**Goal:** freeze dense index spaces (`user_idx`, `item_idx`) and create minimal pair tables for Two-Tower training.

Pick the proxy files you want to convert:

```bash
PROXY_ITEMS=$(ls -1t $RUN_DIR/proxy/proxy_items_*.csv | head -1)
PROXY_INTER=$(ls -1t $RUN_DIR/proxy/proxy_interactions_*.csv | head -1)
```

Convert with defaults:

```bash
python convert_to_twotower.py \
	--interactions $PROXY_INTER \
	--items $PROXY_ITEMS \
	--out_dir $RUN_DIR/twotower_data
```

Key flags (code-level defaults):
- `--label_col label|label_ctr_proxy` (default `label`)
- `--time_holdout` (default `0.2`)
- `--neg_per_pos` (default `4`)
- `--seed` (default `42`)
- `--auto_config off|soft|hard` (default `off`; writes `twotower_autoconfig.json` + `train_suggest.json` when enabled)

What this writes:
- `user_map.csv` (`user_id` → `user_idx`)
- `item_map.csv` (`item_id` → `item_idx`)
- `twotower_train_pairs.csv`, `twotower_val_pairs.csv` (columns are intentionally minimal: `user_idx,item_idx,label`)
- `item_catalog.csv` (keyed by `item_idx`, may include proxy numeric features and split-safe recency columns)

Important practical implication:
- The pair CSVs do **not** include timestamps.
	- If you later need true chronological sequences for DIN, prefer the `proxy_din_interactions_*.jsonl` path (Stage 2).

### 1.3 Train Two-Tower teacher and dump Top-K candidate slates (`train_twotower.py`)

**Goal:** train a retrieval model over `(user_idx, item_idx)`, then dump teacher slates for DIN KD.

Basic training + dump (recommended):

```bash
python train_twotower.py \
	--train_pairs $RUN_DIR/twotower_data/twotower_train_pairs.csv \
	--val_pairs   $RUN_DIR/twotower_data/twotower_val_pairs.csv \
	--user_map    $RUN_DIR/twotower_data/user_map.csv \
	--item_map    $RUN_DIR/twotower_data/item_map.csv \
	--item_catalog $RUN_DIR/twotower_data/item_catalog.csv \
	--use_item_features \
	--out_dir $RUN_DIR/twotower_runs \
	--save_best --save_final --save_metrics \
	--dump_candidates_k 200 \
	--dump_candidates_out $RUN_DIR/tt_candidates_topK.jsonl \
	--dump_score_type dot
```

Notes grounded in code:
- `--dump_candidates_k` must be `> 0` to write `tt_candidates_topK.jsonl`.
- Dumped `candidate_scores` are either `dot` scores or `sigmoid` scores depending on `--dump_score_type`.
	- In docs, treat them as “teacher scores”; don’t over-interpret their absolute scale.
- If you enable `--eval_all_items` and `n_items > --eval_all_items_max_guard` (default `200000`), the script auto-switches back to sampled eval.

Optional: dump candidates from an existing checkpoint (no training):

```bash
python train_twotower.py \
	--train_pairs $RUN_DIR/twotower_data/twotower_train_pairs.csv \
	--val_pairs   $RUN_DIR/twotower_data/twotower_val_pairs.csv \
	--user_map    $RUN_DIR/twotower_data/user_map.csv \
	--item_map    $RUN_DIR/twotower_data/item_map.csv \
	--load_checkpoint $RUN_DIR/twotower_runs/twotower_best.pt \
	--mode dump \
	--dump_candidates_k 200 \
	--dump_candidates_out $RUN_DIR/tt_candidates_topK.jsonl
```

Checkpoint: at the end of Stage 1 you should have these three “bridge artifacts”:
- `runs/<tag>/twotower_data/user_map.csv`
- `runs/<tag>/twotower_data/item_map.csv`
- `runs/<tag>/tt_candidates_topK.jsonl`

## Stage 2 — DIN KD (TF1 graph mode)
Stage 2 creates the DIN dataset pickle and trains a TF1-style DIN model via listwise knowledge distillation.

### 2.1 Build `din/dataset.pkl` (`make_din_dataset.py`)

You have two input paths:

- **Pairs path** (`--pairs_csv_train` + `--pairs_csv_val`): easy, but pair CSVs don’t contain timestamps.
- **Proxy JSONL path** (`--proxy_jsonl`): better aligned with chronological sequence modeling because `proxy_din_interactions_*.jsonl` contains timestamps (used for chronological splitting when present). Note: history fields may exist in the JSONL, but `make_din_dataset.py` currently constructs histories from the event stream internally.

Recommended (proxy JSONL path):

```bash
PROXY_DIN=$(ls -1t $RUN_DIR/proxy/proxy_din_interactions_*.jsonl | head -1)

python make_din_dataset.py \
	--proxy_jsonl $PROXY_DIN \
	--user_map $RUN_DIR/twotower_data/user_map.csv \
	--item_map $RUN_DIR/twotower_data/item_map.csv \
	--item_catalog $RUN_DIR/twotower_data/item_catalog.csv \
	--theme_col theme_raw \
	--out $RUN_DIR/din/dataset.pkl
```

Notes grounded in code:
- `make_din_dataset.py` writes a pickle stream containing:
	- `train_set`, `test_set`, `cate_list`, `theme_list`, and `(n_users, n_items, cate_count, theme_count)`.
- The per-user split is chronological if timestamps exist; otherwise it falls back to the (stable) input row order.
	- If you feed pair CSVs without timestamps, you cannot recover chronology, so the resulting “sequence” only reflects your input ordering.

### 2.2 Train listwise KD DIN (`din/train_listwise_kd.py`)

**Inputs:**
- `dataset.pkl`
- `tt_candidates_topK.jsonl` (teacher slates)
- `user_map.csv`, `item_map.csv`

Minimal KD command:

```bash
python din/train_listwise_kd.py \
	--dataset_pkl $RUN_DIR/din/dataset.pkl \
	--candidates $RUN_DIR/tt_candidates_topK.jsonl \
	--user_map $RUN_DIR/twotower_data/user_map.csv \
	--item_map $RUN_DIR/twotower_data/item_map.csv \
	--proxy_jsonl $PROXY_DIN \
	--out_dir $RUN_DIR/din_kd_runs \
	--K 200 \
	--seed 42
```

Key behavior to keep in mind:
- This is TF1-style graph/session execution (`tf.compat.v1`, `tf.Session`).
- The KD distribution is derived by softmax over teacher scores with temperature `tau`:
	- teacher logits are scaled by `1/tau` before softmax.
- The “best” checkpoint is saved when training average `total_loss` improves.
	- It is not selected by a validation metric.

### 2.3 Export DIN embeddings to CSV (`din/export_user_theme_embeddings.py`)

The next stage (copy-head + scoring) consumes **CSV embedding tables**.

Important index semantics:
- The exported `user_id` and `theme_id` are **embedding row indices** (`0..N-1`).
- They are not the original Talkwalker string IDs.

Export with filenames that downstream scripts can consume consistently:

```bash
python din/export_user_theme_embeddings.py \
	--dataset_pkl $RUN_DIR/din/dataset.pkl \
	--ckpt $RUN_DIR/din_kd_runs/din_kd_best.ckpt \
	--out_user_csv $RUN_DIR/din/din_user_embeddings.csv \
	--out_theme_csv $RUN_DIR/din/din_theme_embeddings.csv
```

## Stage 3 — Ad Copy Scoring & Global Ranking
Stage 3 is where “item-level proxy interactions” get converted into “copy-level ranking.”

There are four sub-steps:
1) build text embeddings for copies and items (same text space; used by nearest-text item→copy mapping)
2) (optional) assess/persist item→copy mapping quality
3) build copy-head supervised samples
4) train copy-head and produce global ranking

### 3.1 Build copy embeddings (`build_copy_embeddings.py`)

What you need depends on your mapping strategy:
- **Copy embeddings** are required for downstream **copy-head training and scoring**.
- **Item embeddings** are required for the recommended default `nearest_text` item→copy mapping (including the mapping report script).
	- Item embeddings are optional only if you choose `--item_to_copy=direct`.

If you need *both* item and copy embeddings in the *same* text space, the minimal pattern is:
- run `fit` on **one** catalog (this also writes an embedding CSV for that catalog), then
- run `transform` on the **other** catalog with the same `--artifact_prefix`.
You only need to run `transform` on the same catalog again if you want a different `--out_copy_emb` filename.

Recommended: define the embedding space from `item_catalog.csv`, then transform `copy_catalog.csv` into that same space:

```bash
# Fit on item catalog (defines artifacts + writes item embeddings)
python build_copy_embeddings.py \
	--catalog_csv $RUN_DIR/twotower_data/item_catalog.csv \
	--id_col item_idx \
	--text_col text \
	--mode fit \
	--emb_dim 64 \
	--artifact_prefix $RUN_DIR/copy/copy_embedding_artifacts \
	--out_copy_emb $RUN_DIR/copy/item_embeddings.csv

# Transform copy catalog into the same space
python build_copy_embeddings.py \
	--catalog_csv $RUN_DIR/inputs/copy_catalog.csv \
	--id_col copy_id \
	--text_col copy_text \
	--mode transform \
	--artifact_prefix $RUN_DIR/copy/copy_embedding_artifacts \
	--out_copy_emb $RUN_DIR/copy/copy_embeddings.csv
```

Alternative (valid, but may behave differently): define the embedding space from `copy_catalog.csv`, then transform `item_catalog.csv` into that same space.
Because `fit` defines the TF‑IDF vocabulary/IDF and the SVD space, the choice of which side you `fit` on can change similarity behavior.

```bash
# Fit on copy catalog (defines artifacts + writes copy embeddings)
python build_copy_embeddings.py \
	--catalog_csv $RUN_DIR/inputs/copy_catalog.csv \
	--id_col copy_id \
	--text_col copy_text \
	--mode fit \
	--emb_dim 64 \
	--artifact_prefix $RUN_DIR/copy/copy_embedding_artifacts \
	--out_copy_emb $RUN_DIR/copy/copy_embeddings.csv

# Transform item catalog into the same space
python build_copy_embeddings.py \
	--catalog_csv $RUN_DIR/twotower_data/item_catalog.csv \
	--id_col item_idx \
	--text_col text \
	--mode transform \
	--artifact_prefix $RUN_DIR/copy/copy_embedding_artifacts \
	--out_copy_emb $RUN_DIR/copy/item_embeddings.csv
```

Practical note about tokenization:
- The TF-IDF tokenizer in this repo is regex-based and is most effective for Latin/ASCII-like tokens.
	If your copy is mainly non-Latin text, expect this embedding space to lose information unless you change the tokenizer.

### 3.2 (Optional) Report / persist item→copy mapping quality (`report_item_to_copy_mapping.py`)

This step is “diagnostics first.”
Nearest-text mapping is a heuristic bridge; if similarity is low, label transfer becomes noisy.

```bash
python report_item_to_copy_mapping.py \
	--item_embeddings_csv $RUN_DIR/copy/item_embeddings.csv \
	--copy_embeddings_csv $RUN_DIR/copy/copy_embeddings.csv \
	--item_catalog_csv $RUN_DIR/twotower_data/item_catalog.csv \
	--out_mapping_csv $RUN_DIR/copy/item_to_copy_map.csv
```

The optional mapping CSV schema is:
- `item_idx, copy_id, sim, sim_gap`

### 3.3 Build ad copy scoring training samples (copy-head) (`build_copy_head_dataset.py`)

This converts proxy interactions into supervised triples:
- `(user_id, theme_id, copy_id, label[, sample_weight])`

Where:
- `user_id` and `theme_id` are dense integer indices compatible with the DIN-exported embedding tables.
- `copy_id` is the dense integer ID from your copy catalog / copy embedding table.

Recommended: nearest-text mapping using the precomputed mapping CSV (lets you filter by similarity if you want):

```bash
python build_copy_head_dataset.py \
	--proxy_interactions_csv $PROXY_INTER \
	--user_map_csv $RUN_DIR/twotower_data/user_map.csv \
	--item_map_csv $RUN_DIR/twotower_data/item_map.csv \
	--item_catalog_csv $RUN_DIR/twotower_data/item_catalog.csv \
	--theme_col theme_raw \
	--item_to_copy nearest_text \
	--item_embeddings_csv $RUN_DIR/copy/item_embeddings.csv \
	--item_to_copy_map_csv $RUN_DIR/copy/item_to_copy_map.csv \
	--min_sim 0.0 \
	--min_gap 0.0 \
	--user_emb_csv $RUN_DIR/din/din_user_embeddings.csv \
	--theme_emb_csv $RUN_DIR/din/din_theme_embeddings.csv \
	--copy_emb_csv $RUN_DIR/copy/copy_embeddings.csv \
	--label_column label \
	--label_positive_value 1.0 \
	--engagement_column engagement_total \
	--min_engagement 1.0 \
	--num_neg_per_pos 3 \
	--seed 42 \
	--out_csv $RUN_DIR/copy/copy_head_train_samples.csv
```

Notes grounded in code (so you don’t get surprised):
- A row becomes a positive if it matches the label threshold **or** the engagement threshold.
- If you set `--weight_column`, the script will generate `sample_weight`.
	- In the proxy-driven builder, sampled negatives inherit the same weight as their originating positive.
- If `--proxy_interactions_csv` is missing/unreadable, the script falls back to a toy dataset.
	- For a real run, you want the proxy-driven path.

### 3.4 Train ad copy scoring model (copy-head) (`train_copy_head.py`)

By default, this script disables GPU visibility via TensorFlow config to keep runtime predictable.

Train and save a model file that the scorer expects:

```bash
python train_copy_head.py \
	--user_emb_csv $RUN_DIR/din/din_user_embeddings.csv \
	--theme_emb_csv $RUN_DIR/din/din_theme_embeddings.csv \
	--copy_emb_csv $RUN_DIR/copy/copy_embeddings.csv \
	--train_samples_csv $RUN_DIR/copy/copy_head_train_samples.csv \
	--out_model $RUN_DIR/copy/copy_head_model.keras \
	--seed 42
```

Important correctness check:
- `train_copy_head.py` bounds-checks IDs.
	If you see errors like “index exceeds available embeddings,” it almost always means you mixed index spaces across runs.

### 3.5 Produce global ranking (`score_copies_with_din_and_copy_head.py`)

This script builds per-theme “contexts” from the training positives, scores every copy under each theme context, and aggregates across themes.

Important semantics:
- The flag name `--theme_weight_temperature` is a bit misleading if you expect softmax-temperature semantics.
	- In code, it applies a **power/exponent transform** to raw theme weights: `w := w^temperature`, then normalizes and mixes with a uniform prior.
- `--top_k` controls console display only; if you pass `--out_csv`, the written CSV contains the full ranking over all copies.

Run scoring (writes ranking CSV only if `--out_csv` is provided):

```bash
python score_copies_with_din_and_copy_head.py \
	--copy_catalog_csv $RUN_DIR/inputs/copy_catalog.csv \
	--copy_embeddings_csv $RUN_DIR/copy/copy_embeddings.csv \
	--user_emb_csv $RUN_DIR/din/din_user_embeddings.csv \
	--theme_emb_csv $RUN_DIR/din/din_theme_embeddings.csv \
	--copy_head_model $RUN_DIR/copy/copy_head_model.keras \
	--train_samples_csv $RUN_DIR/copy/copy_head_train_samples.csv \
	--theme_weight_temperature 0.5 \
	--theme_uniform_mix 0.1 \
	--top_k 20 \
	--out_csv $RUN_DIR/copy/final_ranking.csv
```

At this point, the operational output is:
- the ranking CSV you wrote via `--out_csv` (e.g., `runs/<tag>/copy/final_ranking.csv`)

If you want a single default pick from this run, start with the row where `rank == 1`.

And you still have the full lineage of artifacts (proxy tables, index maps, teacher slates, DIN checkpoints, embedding tables) needed to reproduce or debug the ranking.

