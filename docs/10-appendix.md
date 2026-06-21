# 10. Appendix
## 10.1 Command Reference

This appendix is a “how to call the code” index.

General conventions used below:
- You can always run `python <script>.py --help` to see the authoritative flags.
- Examples assume you keep a single run directory per experiment tag (e.g., `$RUN_DIR=runs/<tag>`).
- Most stages write artifacts to disk; later stages read those artifacts. When a stage fails, check that you didn’t mix artifacts from different tags.

### Stage 1 — Proxy contract

#### `build_proxy.py`
Turns a Talkwalker export into the proxy contract (items + interactions + DIN JSONL).

Required
- `--src`: input file (`.csv` or `.xlsx`)

Common options
- `--out` (default `./proxy`): output directory where `proxy_*_<tag>` are written
- `--pseudo_user_mode` (default `cohort`, choices `cohort|session|mixed`): pseudo-user construction strategy
- `--pseudo_user_k_max` (default `32`): upper bound on pseudo-user buckets per cohort
- `--pseudo_user_target_hist` (default `12`): target interaction count per pseudo-user (used by adaptive bucketing)
- `--session_gap_minutes` (default `360`): session split threshold for `session` mode
- `--session_domains` (default empty string): comma-separated domains to treat as “sessionized” in `mixed` mode
- `--min_engagement_for_pos` (default `1`): positive threshold for the engagement-based label
- `--label_quantile` (default `None`): alternative positive definition using quantiles
- `--composite_label_quantile` (default `0.7`): quantile threshold for `label_ctr_proxy` (composite proxy)
- `--half_life_days` (default `7.0`): recency decay parameter used in composite scoring
- `--weights` (default `0.25,0.25,0.25,0.25`): weights used inside the composite proxy
- `--label_groupby` (default `none`, choices `none|source_type|domain|lang`): group-wise normalization for label construction
- `--dedup_keep` (default `earliest`, choices `earliest|max_eng|textlen`): how to pick a representative row when deduplicating
- `--max_history_len` (default `20`): maximum history length emitted for DIN JSONL
- `--no_quality_report`: disables `quality_<tag>.json`
- `--auto_tune`: emits an auto-tune report (and optionally `--auto_report_out`)

Outputs (written under `--out`)
- `proxy_items_<tag>.csv`
- `proxy_interactions_<tag>.csv`
- `proxy_din_interactions_<tag>.jsonl`
- `quality_<tag>.json` (unless disabled)

### Stage 2 — Two‑Tower input contract

#### `convert_to_twotower.py`
Converts proxy artifacts into a dense-index universe + Two‑Tower training pairs.

Required
- `--interactions`: proxy interactions CSV (must contain `user_id,item_id,timestamp` and the selected label column)
- `--items`: proxy items CSV (must contain `item_id`)

Common options
- `--label_col` (default `label`, choices `label|label_ctr_proxy`): which proxy label to use
- `--time_holdout` (default `0.2`): holdout fraction for time split
- `--per_user_split`: split per user instead of global time cutoff
- `--min_events_per_user` (default `2`): drop users with fewer events
- `--neg_per_pos` (default `4`): negatives per positive (negative sampling)
- `--seed` (default `42`)
- `--quality_json`: optional `quality_<tag>.json` for sanity checks
- `--auto_tune_json`: optional auto-tune report from proxy stage
- `--auto_config` (default `off`, choices `off|soft|hard`): writes suggested config JSONs for training
- `--out_dir` (default `./twotower_data`)
- `--half_life_days` (default `7.0`): used when generating split-safe recency variants

Outputs (written under `--out_dir`)
- `twotower_train_pairs.csv` / `twotower_val_pairs.csv`
- `user_map.csv` / `item_map.csv`
- `item_catalog.csv` (includes `item_idx` and pass-through proxy feature columns when present)

### Stage 3 — Two‑Tower training + teacher slate dumping

#### `train_twotower.py`
Trains the retrieval model, evaluates pointwise + ranking metrics, and can dump teacher slates for DIN KD.

Required
- `--train_pairs`, `--val_pairs`: from `convert_to_twotower.py`
- `--user_map`, `--item_map`: from `convert_to_twotower.py`

Key training options
- `--dim` (default `64`): embedding dimension
- `--batch_size` (default `2048`), `--epochs` (default `10`), `--lr` (default `1e-3`)
- `--weight_decay` (default `1e-5`)
- `--seed` (default `42`)
- `--ensure_negatives` (default `0`): if >0, adds negatives if the dataset is too positive-heavy
- `--scheduler` (default `none`, choices `none|cosine|steplr`), `--step_size`, `--gamma`
- `--grad_clip` (default `0.0`): gradient clipping threshold (0 disables)
- `--emb_reg` (default `0.0`): embedding L2 regularization

Evaluation options
- `--k` (default `10,20`): ranking cutoffs
- `--eval_all_items`: rank against all items (can be expensive)
- `--eval_all_items_max_guard` (default `200000`): safety guard for all-items eval
- `--eval_sample_k` (default `1000`): sampled negatives for ranking eval when not all-items
- `--eval_item_batch` (default `65536`): item batch size for all-items scoring
- `--early_metric` (default `auc`, choices `auc|ndcg`)
- `--early_stop` (default `0`): patience (0 disables)

Saving / outputs
- `--out_dir` (default `./twotower_runs`)
- `--save_best`, `--save_final`: checkpoint writing
- `--save_metrics` and `--metrics_out` (default `metrics.jsonl`)

Teacher slate dumping (for DIN KD)
- To dump *without training*, use `--mode dump` (or set `--epochs 0`)
- `--dump_candidates_k` (default `0`, set >0 to enable)
- `--dump_candidates_out` (default `./tt_candidates_topK.jsonl`)
- `--dump_users_scope` (default `val`, choices `val|train|all`)
- `--dump_score_type` (default `dot`, choices `dot|sigmoid`)

Optional embedding export
- `--export_embeddings`: dumps Two‑Tower embedding tables (used mainly for inspection)

### Stage 4 — DIN dataset preparation

#### `make_din_dataset.py`
Builds `din/dataset.pkl` for DIN training.

Required
- `--user_map`, `--item_map`: from `convert_to_twotower.py`

Input modes (provide exactly one)
- `--pairs_csv_train` + `--pairs_csv_val`: typically the Two‑Tower train/val pairs
- `--pairs_csv`: single CSV; internally split per-user chronologically
- `--proxy_jsonl`: JSONL proxy interactions (recommended if you want timestamps preserved)

Common options
- `--item_catalog`: path to `item_catalog.csv` if you want categories/themes
- `--cate_col`: column in item catalog used as cate feature
- `--theme_col`: column in item catalog used as theme feature (string values will be re-encoded into dense IDs)
- `--format` (default `legacy`, choices `legacy|std`): dataset tuple format
- `--test_ratio` (default `0.1`)
- `--out` (default `./din/dataset.pkl`)

### Stage 5 — DIN listwise KD training

#### `din/train_listwise_kd.py`
Trains a DIN student on teacher slates (`tt_candidates_topK.jsonl`) using listwise KD.

Required
- `--dataset_pkl`: output from `make_din_dataset.py`
- `--candidates`: Two‑Tower dump JSONL (`tt_candidates_topK.jsonl`)
- `--user_map`, `--item_map`: to map positives/candidates consistently

Common options
- `--proxy_jsonl`: optional; if provided, positives come from JSONL instead of dataset
- `--ckpt_in`: optional; warm-start checkpoint
- `--out_dir` (default `./din_kd_runs`)
- `--epochs` (default `3`), `--batch_size` (default `128`)
- `--K` (default `200`): candidate slate size used in training (must match your dump `K` conceptually)
- `--tau` (default `1.5`): temperature for teacher/student distribution smoothing
- `--lambda_kd` (default `0.7`): KD loss weight
- `--lr` (default `5e-4`)
- `--point_loss_weight` (default `1.0`): weight for DIN’s original pointwise loss component

Outputs
- `din_kd_best.ckpt` written under `--out_dir` (selected by minimum training `total_loss` in this implementation)

### Stage 6 — Export DIN embeddings for downstream use

#### `din/export_user_theme_embeddings.py`
Exports DIN embedding tables into CSVs used by the copy layer.

Required
- `--ckpt`: DIN checkpoint (e.g., `din_kd_best.ckpt`)

Options
- `--dataset_pkl` (default `din/dataset.pkl`): needed to know table sizes
- `--out_user_csv` (default `user_embeddings.csv`)
- `--out_theme_csv` (default `theme_embeddings.csv`)

Outputs
- user embedding CSV with columns: `user_id, emb_*`
- theme embedding CSV with columns: `theme_id, emb_*`

### Stage 7 — Copy layer

#### `build_copy_embeddings.py`
Builds TF‑IDF+SVD embeddings for a given catalog CSV. Despite the name, it can embed any CSV with an ID and text column.

Common options
- `--catalog_csv` (default `copy_catalog.csv`)
- `--id_col` (default `copy_id`)
- `--text_col` (default `copy_text`)
- `--mode` (default `fit`, choices `fit|transform`)
- `--out_copy_emb` (default `copy_embeddings.csv`)
- `--artifact_prefix` (default `copy_embedding_artifacts`): where vocab/SVD artifacts are saved/loaded
- `--emb_dim` (default `64`): SVD dimension in fit mode
- `--min_freq` (default `1`): vocabulary cutoff
- `--svd_n_iter` (default `7`), `--svd_random_state` (default `0`)

Outputs
- Embedding CSV: `{id_col}, emb_*`
- Fit mode additionally writes artifacts under `{artifact_prefix}.*`
- For nearest-text item→copy mapping, you typically run:
	- `fit` on one side (writes artifacts + embeddings for that catalog), then
	- `transform` on the other side using the same `--artifact_prefix`.

#### `report_item_to_copy_mapping.py`
Reports a nearest-neighbor mapping from item embeddings to copy embeddings (cosine similarity).

Optional: you do not need to run this to train or score. It is mainly for inspecting mapping quality and (optionally) persisting an `item_to_copy_map.csv` that you can feed into `build_copy_head_dataset.py` via `--item_to_copy_map_csv`.

Required
- `--item_embeddings_csv`, `--copy_embeddings_csv`

Common options
- `--item_id_col` (default `item_idx`)
- `--copy_id_col` (default `copy_id`)
- `--item_catalog_csv` (default `None`): if provided, can print example texts
- `--item_text_col` (default `text`)
- `--topk_examples` (default `3`)
- `--out_mapping_csv` (default `None`): persist mapping CSV for downstream filtering

#### `build_copy_head_dataset.py`
Builds copy-head training samples `(user_id, theme_id, copy_id, label[, sample_weight])`.

Proxy mode (recommended)
- `--proxy_interactions_csv`: proxy interactions from the Proxy stage

Required-ish (defaults exist but you typically point them to `$RUN_DIR`)
- `--user_map_csv` (default `twotower_data/user_map.csv`)
- `--item_map_csv` (default `twotower_data/item_map.csv`)
- `--item_catalog_csv` (default `twotower_data/item_catalog.csv`)

Theme options
- `--theme_col` (default `theme_raw`): theme column in item catalog (string values are re-encoded)
- `--exclude_theme_id` (default `None`): drops a dominant “unknown” theme bucket if needed
- `--copy_catalog_theme_col` (default `None`): currently not used by code; theme_id is derived from `--item_catalog_csv` via `--theme_col`

Critical invariant: to keep `theme_id` aligned with `theme_embeddings.csv`, use the same `item_catalog.csv` and `--theme_col` as `make_din_dataset.py` (both scripts re-encode string themes to dense IDs in file order).

Item→copy mapping options
- `--item_to_copy` (default `nearest_text`, choices `nearest_text|direct`)
	- `nearest_text` needs `--item_embeddings_csv` and `--copy_emb_csv`
	- `direct` uses `--copy_catalog_csv` and maps proxy `item_id` to `copy_id` via `--copy_catalog_item_col` (recommended only when you have verified the ID universe match)
- `--item_to_copy_map_csv`: optional mapping CSV with `item_idx,copy_id,(sim,sim_gap)`
- `--min_sim` / `--min_gap`: filters applied if mapping CSV contains those columns

Label/weight options
- `--label_column` (default `label`) and `--label_positive_value` (default `1.0`)
- `--engagement_column` (default `engagement_total`) and `--min_engagement` (default `1.0`)
- `--weight_column` (default `None`): if provided, becomes `sample_weight` in output

Sampling options
- `--max_pos_samples` (default `20000`)
- `--num_neg_per_pos` (default `3`)
- `--seed` (default `42`)
- `--out_csv` (default `copy_train_samples.csv`)
	- If you want to rely on scorer defaults later, set `--out_csv copy_head_train_samples.csv`.

Toy fallback mode
- If `--proxy_interactions_csv` is missing, the script will emit a toy dataset using `--user_emb_csv/--theme_emb_csv/--copy_emb_csv` to infer sizes.

#### `train_copy_head.py`
Trains the TF2/Keras copy head.

Required
- `--train_samples_csv`: output from `build_copy_head_dataset.py`

Common options
- `--user_emb_csv` (default `user_embeddings.csv`)
- `--theme_emb_csv` (default `theme_embeddings.csv`)
- `--copy_emb_csv` (default `copy_embeddings.csv`)
- `--sample_weight_col` (default `sample_weight`)
- `--weight_transform` (default `log1p`, choices `none|log1p|sqrt`)
- `--disable_weight_normalize`: disables mean normalization
- `--out_model` (default `copy_head.keras`)
	- The scorer defaults to `copy_head_model.keras`, so it’s usually easiest to train with `--out_model copy_head_model.keras` (or pass `--copy_head_model` at scoring time).
- `--epochs` (default `10`), `--batch_size` (default `256`)

Important contract
- IDs in `train_samples_csv` are treated as embedding-table indices. `copy_id` must be a dense index aligned with the rows in your copy embedding CSV.

#### `score_copies_with_din_and_copy_head.py`
Produces the final global copy ranking CSV.

Inputs
- Copy catalog + copy embeddings + DIN user/theme embeddings + copy-head model + training samples

Common options
- `--copy_catalog_csv` (default `copy_catalog.csv`)
- `--copy_embeddings_csv` (default `copy_embeddings.csv`)
- `--user_emb_csv` (default `din_user_embeddings.csv`)
- `--theme_emb_csv` (default `din_theme_embeddings.csv`)
- `--copy_head_model` (default `copy_head_model.keras`)
- `--train_samples_csv` (default `copy_head_train_samples.csv`)

Naming note
- `din/export_user_theme_embeddings.py` defaults to `user_embeddings.csv` / `theme_embeddings.csv`.
- If you keep those default names, pass `--user_emb_csv user_embeddings.csv --theme_emb_csv theme_embeddings.csv` here.

Theme-context / weighting options
- `--weight_column` (default `sample_weight`)
- `--theme_min_positives` (default `10`)
- `--exclude_theme_id` (default `None`)
- `--theme_weight_temperature` (default `0.5`)
- `--theme_uniform_mix` (default `0.1`)

Output
- Prints top contexts and top-K ranking; writes CSV only if `--out_csv` is provided

---

## 10.2 Config Templates

This repo does not use a single YAML/JSON “master config”; instead the “config” is:
- a run directory (`$RUN_DIR`)
- a set of artifacts inside it
- the exact CLI flags that produced those artifacts

Below are templates that make runs reproducible and reduce accidental cross-tag mixing.

### Template A — Run directory + helper variables

```bash
# Choose a tag and an explicit run directory
TAG=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="runs/$TAG"

mkdir -p "$RUN_DIR" \
	"$RUN_DIR/inputs" \
	"$RUN_DIR/proxy" \
	"$RUN_DIR/twotower_data" \
	"$RUN_DIR/twotower" \
	"$RUN_DIR/din" \
	"$RUN_DIR/copy"

# Optional: keep a lightweight manifest of what you ran
RUN_MANIFEST="$RUN_DIR/run_manifest.txt"
echo "tag=$TAG" > "$RUN_MANIFEST"
echo "created_at=$(date -Iseconds)" >> "$RUN_MANIFEST"
```

### Template B — “Pick latest artifact” helpers (for debugging)

When scripts produce timestamped filenames (`*_YYYYMMDD_...`), you often want a stable way to select the newest one.

```bash
LATEST_PROXY_INTERACTIONS=$(ls -1t $RUN_DIR/proxy/proxy_interactions_*.csv | head -n 1)
LATEST_PROXY_ITEMS=$(ls -1t $RUN_DIR/proxy/proxy_items_*.csv | head -n 1)
LATEST_PROXY_DIN=$(ls -1t $RUN_DIR/proxy/proxy_din_interactions_*.jsonl | head -n 1)

echo "proxy_interactions=$LATEST_PROXY_INTERACTIONS"
echo "proxy_items=$LATEST_PROXY_ITEMS"
echo "proxy_din=$LATEST_PROXY_DIN"
```

### Template C — Minimal “artifact bundle” layout

This is the smallest set of paths you typically want to keep together.

```text
$RUN_DIR/
	inputs/
		talkwalker_export.xlsx (or .csv)
		copy_catalog.csv
	proxy/
		proxy_items_<tag>.csv
		proxy_interactions_<tag>.csv
		proxy_din_interactions_<tag>.jsonl
		quality_<tag>.json (optional)
	twotower_data/
		user_map.csv
		item_map.csv
		item_catalog.csv
		twotower_train_pairs.csv
		twotower_val_pairs.csv
	twotower/
		twotower_best.pt (if saved)
		tt_candidates_topK.jsonl
	din/
		dataset.pkl
		din_kd_best.ckpt*
		din_user_embeddings.csv
		din_theme_embeddings.csv
	copy/
		copy_embeddings.csv
		item_embeddings.csv (required for nearest-text item→copy mapping; optional only if using direct mapping)
		item_to_copy_map.csv (optional)
		copy_head_train_samples.csv
		copy_head_model.keras
		final_ranking.csv
```

### Template D — Embedding-space pinning

If you want runs to be comparable, keep the same embedding artifacts prefix when transforming.

```bash
# Fit once (defines the embedding space)
ARTIFACT_PREFIX="$RUN_DIR/copy/tfidf_svd_space"

python build_copy_embeddings.py \
	--mode fit \
	--catalog_csv $RUN_DIR/inputs/item_catalog_like.csv \
	--id_col item_idx \
	--text_col text \
	--artifact_prefix "$ARTIFACT_PREFIX" \
	--out_copy_emb $RUN_DIR/copy/item_embeddings.csv

# Transform many times (reuses the same space)
python build_copy_embeddings.py \
	--mode transform \
	--catalog_csv $RUN_DIR/inputs/copy_catalog.csv \
	--id_col copy_id \
	--text_col copy_text \
	--artifact_prefix "$ARTIFACT_PREFIX" \
	--out_copy_emb $RUN_DIR/copy/copy_embeddings.csv
```

---

## 10.3 Glossary & Terminology Map

This section maps the repository’s terms to the concrete artifacts and ID spaces used in code.

### IDs: string keys vs dense indices

- `user_id` (string): a pseudo-user identifier produced by `build_proxy.py`.
	- It may represent a cohort bucket or a session bucket, not necessarily a real person.

- `user_idx` (int, dense): `0..(num_users-1)` index used by embedding tables.
	- Written in `twotower_data/user_map.csv`.

- `item_id` (string): a stable item identifier produced by `build_proxy.py`.
	- If a URL exists, it tends to drive stability. Otherwise it falls back to a normalized text/time key.

- `item_idx` (int, dense): `0..(num_items-1)` index used by Two‑Tower and DIN.
	- Written in `twotower_data/item_map.csv` and `twotower_data/item_catalog.csv`.

- `theme_raw` (string-ish): a raw theme/category value.
	- In this repo, it is typically carried in `item_catalog.csv` as a column and later re-encoded.

- `theme_id` (int, dense): `0..(num_themes-1)` theme index used by DIN and the copy layer.
	- Created by re-encoding `theme_raw` (or an integer theme column) inside `make_din_dataset.py` / `build_copy_head_dataset.py`.

- `copy_id` (int): identifier for copy candidates.
	- For **copy-head training**, `copy_id` must behave like an embedding-table row index aligned with the rows of `copy_embeddings.csv` (dense `0..N-1` is the safe mental model).
	- For **ranking/scoring**, embeddings are loaded into an ID→vector mapping; in principle IDs can be non-dense, but in practice keeping `copy_id` dense and consistent across artifacts avoids hard-to-debug mismatches.
	- If your source catalog uses arbitrary IDs, introduce a remapping step to make it dense and stable.

### Artifacts and what they mean

- Proxy contract
	- `proxy_items_<tag>.csv`: deduplicated items with text + proxy features
	- `proxy_interactions_<tag>.csv`: interactions (pseudo-user ↔ item) with timestamps and labels
	- `proxy_din_interactions_<tag>.jsonl`: DIN-friendly interaction records (includes histories)

- Two‑Tower contract
	- `twotower_train_pairs.csv` / `twotower_val_pairs.csv`: `(user_idx,item_idx,label)` pairs for retrieval training
	- `item_catalog.csv`: item features (includes numeric proxy features when present)

- Teacher slate
	- `tt_candidates_topK.jsonl`: per-user Top‑K candidate items dumped by Two‑Tower
	- Used by DIN KD as the candidate universe it learns to rank within.

- DIN dataset
	- `din/dataset.pkl`: pickled dataset payload expected by the TF1 DIN implementation.

- DIN KD
	- `din_kd_best.ckpt*`: best checkpoint chosen by minimum training `total_loss` in this implementation.

- DIN embedding exports
	- `user_embeddings.csv` / `theme_embeddings.csv` are the default output names from `din/export_user_theme_embeddings.py`.
	- This README often renames them to `din_user_embeddings.csv` / `din_theme_embeddings.csv` to make it explicit that they came from DIN.
	- In both cases, the schema is `user_id, emb_*` and `theme_id, emb_*` and the IDs are dense integer indices.

- Text embeddings
	- `copy_embeddings.csv` / `item_embeddings.csv`: `{id}, emb_*` for copy/item text
	- Embedding columns are always named `emb_0, emb_1, ...`.

- Copy-head samples
	- `copy_head_train_samples.csv`: `(user_id, theme_id, copy_id, label[, sample_weight])`
	- These integer IDs are indices into the exported embedding tables.

### Model terms

- Two‑Tower: dot-product retrieval model used as a fast candidate generator and teacher.
- Teacher: Two‑Tower outputs (Top‑K candidate set + score) used as supervision for the student.
- Student: DIN trained to rank within teacher slates.
- KD (knowledge distillation): training where the student learns a distribution over candidates similar to the teacher’s.
- `K`: candidate slate size (Top‑K). This shows up in dumping, KD training, and reranking.
- `tau` (temperature): softens distributions; higher `tau` makes teacher probabilities less peaky.
- Copy head: lightweight model that combines (user emb, theme emb, copy emb) to score copy candidates.


