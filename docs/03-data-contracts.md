# 3. Data Contracts
## 3.1 Talkwalker Inputs
This pipeline intentionally treats the “Talkwalker export” as a *source dataset* rather than a strict schema. The reason is visible in code: `build_proxy.py` first normalizes column names to **snake_case** and then *opportunistically* picks whichever columns exist (`title`, `content`, `url`, `published`, etc.).

That said, there is still a practical contract if you want the downstream models to be meaningful: you generally want enough information to (a) build text, (b) build a stable item ID, (c) provide a plausible event timestamp, and (d) provide at least one engagement-like signal.

The implementation is permissive: if timestamps or engagement columns are missing, the proxy step can still produce outputs, but the resulting labels/features can become degenerate (e.g., nearly all zeros), which usually makes later training uninformative.

### 3.1.1 File formats
`build_proxy.py` supports:
- `.csv` (via `pandas.read_csv`)
- `.xlsx` (via `pandas.read_excel(sheet_name=0)`)

### 3.1.2 Column normalization
Before any logic runs, columns are normalized by:
- stripping whitespace
- replacing non-word characters with `_`
- collapsing repeated `_`
- lowercasing

So a column like `Published At` becomes `published_at`.

### 3.1.3 Minimal fields the proxy step needs
The minimum set depends on what you want downstream, but the proxy builder is designed to work with surprisingly messy exports.

**A) Text construction**

`build_proxy.py` constructs a `text` field by combining title-like and content-like columns when available.

**B) URL and domain extraction**

Used to stabilize IDs and derive `domain`.

If an URL-like column is present, it is used to derive `domain` and stabilize IDs.

`domain` is extracted from the parsed URL.

**C) Timestamps**

Used to define recency features, build event time, and (optionally) session pseudo-users.

If timestamp-like columns are present, they are parsed with `pandas.to_datetime(..., utc=True)` and a “preferred timestamp” is derived for downstream use.

If neither exists, the proxy builder will still emit a `timestamp` column by filling missing values with a default epoch-like value downstream.

**D) Engagement and exposure metrics**

The proxy builder does *not* require a single canonical metric name. It derives engagement/exposure proxy totals from whatever engagement-like columns are available (commonly the `article_extended_attributes.*` family, plus exposure-like columns such as `impressions` / `reach` when present).

It then splits them into:
- **exposure-like metrics** (views/reach/impressions)
- **action-like metrics** (shares/likes/retweets/etc.)

And defines:
- `engagement_actions_total = sum(action_like_metrics)`
- `exposure_total = sum(exposure_like_metrics)`
- `engagement_total = engagement_actions_total`

This is the value used for labeling (`label`) and also saved into the proxy contracts. If no engagement columns are present, these totals will be zero, which typically collapses labels.

### 3.1.4 Optional but important metadata columns
These are not strictly required for the pipeline to run, but they affect cohorting, filtering, or later “theme” modeling.

- `lang` → saved as `lang`
- `nsfw_level` → mapped to `nsfw_level_num`
- `sentiment` → mapped to `sentiment_num`
- `source_type`, `post_type` → used for cohort keys or group-normalized labeling depending on args

### 3.1.5 Theme field selection (`theme_raw`)
If a theme/category-like column exists in the export, the proxy step captures it as `theme_raw`; otherwise `theme_raw` is an empty string.

This `theme_raw` value flows into later contracts and is often the starting point for DIN/copy-head “theme_id” indexing.

### 3.1.6 What the proxy builder guarantees
Once `build_proxy.py` succeeds, you can rely on:

- Stable string IDs:
	- `item_id`: a stable hash
		- based on URL if available
		- else based on `(domain, normalized_text_snippet, date)`
	- `user_id`: pseudo-user string based on `--pseudo_user_mode` (`cohort|session|mixed`)
- A proxy interaction table with `(user_id, item_id, timestamp)` and binary label columns.

## 3.2 Intermediate CSV / JSONL Contracts
This section is the “hard interface” between scripts. A good mental model is:

- **CSV/JSONL files are the system boundary.**
- The exact filenames can change via CLI args, but the **columns and join keys** below are what downstream code assumes.

To keep this section actionable, each contract has:
- **Produced by** (script)
- **Consumed by** (next scripts)
- **Required columns** (the minimum to avoid runtime errors)
- **Key invariants** (how to keep index spaces aligned)

### 3.2.1 `proxy_items_<tag>.csv`
**Produced by:** `build_proxy.py`

**Consumed by:** `convert_to_twotower.py` (items side)

**Primary key:** `item_id` (string)

**Typical columns (subset; only some are guaranteed to exist)**
- Identity/text:
	- `item_id` (string)
	- `text` (string)
	- `url_canonical` (string)
	- `domain` (string)
- Metadata:
	- `lang` (string)
	- `sentiment_num` (float)
	- `nsfw_level_num` (float)
	- `source_type` (string)
	- `post_type` (string)
	- `theme_raw` (string)
- Aggregated engagement/exposure:
	- `engagement_total` (float)
	- `engagement_actions_total` (float)
	- `exposure_total` (float)
- Proxy feature block (saved at item level):
	- `proxy_volume`, `proxy_eng_rate`, `proxy_sent_pos`, `proxy_recency` (float)
	- normalized variants `*_n` (float)
	- `proxy_ctr_composite` (float)
	- `label_ctr_proxy` (int; 0/1)

**Invariants**
- `item_id` must be stable across repeated runs if you want reproducible mapping.
- `text` is what later text-embedding steps will use if you embed items.

### 3.2.2 `proxy_interactions_<tag>.csv`
**Produced by:** `build_proxy.py`

**Consumed by:** `convert_to_twotower.py` (interactions side), `build_copy_head_dataset.py` (for sample building)

**Join keys**
- to proxy items: `item_id`

**Required columns (as enforced by `convert_to_twotower.py`)**
- `user_id` (string)
- `item_id` (string)
- `timestamp` (datetime-like string)
- at least one label column specified by `--label_col` in `convert_to_twotower.py`

**Columns written by `build_proxy.py` (minimum guarantee)**
- `user_id` (string)
- `item_id` (string)
- `timestamp` (naive datetime string; source is parsed as UTC then tz removed)
- `label` (int; 0/1)
- `engagement_total` (float)

**Additional columns added by proxy builder when available**
- `label_ctr_proxy` (int; derived from `proxy_ctr_composite` quantile)
- `proxy_ctr_composite` (float)
- `theme_raw` (string; merged from item table when present)

**Invariants**
- `label` and `label_ctr_proxy` are distinct constructs; downstream steps explicitly choose which label column to use.

### 3.2.3 `proxy_din_interactions_<tag>.jsonl`
**Produced by:** `build_proxy.py`

**Consumed by:** `make_din_dataset.py` via `--proxy_jsonl`

**Format:** JSON Lines (one JSON object per interaction)

**Fields written by `build_proxy.py`**
- `user_id` (string)
- `item_id` (string)
- `timestamp` (ISO string)
- `label` (int; 0/1)
- `engagement_total` (int)
- `hist_item_ids` (list of string item IDs)
- `hist_len` (int)

**Invariants**
- History is constructed by sorting `proxy_interactions` by `(user_id, timestamp)`.
- `hist_item_ids` are string item IDs for convenience/inspection. Current `make_din_dataset.py` does **not** consume these fields; it rebuilds histories internally from the event stream after mapping `user_id/item_id` (or `user_idx/item_idx`) and using `timestamp` when present.

### 3.2.4 `quality_<tag>.json` and `auto_tune_report_<tag>.json`
**Produced by:** `build_proxy.py` (quality report on by default; auto-tune report only when `--auto_tune`)

**Consumed by:** `convert_to_twotower.py` only for *suggestions* (`--auto_config` and optional inputs), not as a hard dependency.

These files are primarily “run provenance”. They do not participate in joins, but they help explain why labeling / cohorting behaved a certain way.

### 3.2.5 `user_map.csv` and `item_map.csv`
**Produced by:** `convert_to_twotower.py`

**Consumed by:** almost everything after conversion (`train_twotower.py`, `make_din_dataset.py`, DIN KD, copy-head dataset builder)

**Schema**
- `user_map.csv`: `user_id` (string), `user_idx` (int)
- `item_map.csv`: `item_id` (string), `item_idx` (int)

**Invariants (critical)**
- `user_idx` and `item_idx` define the canonical dense index spaces.
- Downstream embedding tables assume indices are in `[0..max_idx]` and are used as embedding row IDs.

### 3.2.6 `twotower_train_pairs.csv` / `twotower_val_pairs.csv`
**Produced by:** `convert_to_twotower.py`

**Consumed by:** `train_twotower.py` and optionally `make_din_dataset.py` (pairs-based path)

**Schema (as written by `convert_to_twotower.py`)**
- `user_idx` (int)
- `item_idx` (int)
- `label` (int; 0/1)

Notes:
- `convert_to_twotower.py` reads timestamps from proxy interactions, but the saved pair CSVs are intentionally minimal.

### 3.2.7 `item_catalog.csv`
**Produced by:** `convert_to_twotower.py`

**Consumed by:** `train_twotower.py` (optional numeric item features), `make_din_dataset.py` (cate/theme list building), `build_copy_head_dataset.py` (theme mapping)

**Keys**
- Always includes `item_id` (string) and `item_idx` (int).

**Typical columns (depending on what exists in proxy items)**
- `text` (string)
- Proxy numeric features:
	- `proxy_*` (float)
	- normalized `proxy_*_n` (float)
- Metadata:
	- `domain`, `lang`, `sentiment_num`
	- `theme_raw` (string)
- Split-safe recency variants may be added when computable:
	- `proxy_recency_train`, `proxy_recency_train_n`
	- `proxy_recency_val`, `proxy_recency_val_n`

**Invariants**
- If you want theme-aware downstream steps, you must ensure `theme_raw` (or whichever theme column you choose later) is present and stable.

### 3.2.8 `tt_candidates_topK.jsonl`
**Produced by:** `train_twotower.py` when `--dump_candidates_k > 0`

**Consumed by:** `din/train_listwise_kd.py`

**Format:** JSON Lines (one record per user)

**Fields written by `train_twotower.py`**
- `user_idx` (int)
- `user_id` (string or null)
- `candidate_item_idx` (list[int])
- `candidate_item_id` (list[string] or null)
- `candidate_scores` (list[float])
- `k` (int)
- `score_type` (string; e.g., `dot` or `sigmoid` depending on `--dump_score_type`)

**Invariants**
- Candidate slates are expressed in `item_idx` and must match the item universe used to create `din/dataset.pkl`.
- KD assumes `candidate_item_idx` and `candidate_scores` are aligned arrays of the same length.

### 3.2.9 `din/dataset.pkl`
**Produced by:** `make_din_dataset.py`

**Consumed by:** DIN training (`din/train_listwise_kd.py`) and embedding export (`din/export_user_theme_embeddings.py`)

**Binary format:** Python `pickle` stream (multiple objects dumped sequentially)

`make_din_dataset.py` writes, in order:
1) `train_set`
2) `test_set`
3) `cate_list`
4) `theme_list`
5) `(n_users, n_items, cate_count, theme_count)`

**Train/test sample tuple formats**
- `--format legacy` (default)
	- `train_set` elements: `(user_idx, hist_item_idx_list, target_item_idx, label)`
	- `test_set` elements: `(user_idx, hist_item_idx_list, [pos_item_idx, neg_item_idx])`
- `--format std`
	- train/test elements: `(user_idx, hist_item_idx_list, hist_cate_list, target_item_idx, target_cate, label)`

**How `cate_list` and `theme_list` are built**
- Both are length `n_items` arrays indexed by `item_idx`.
- If the corresponding column in `item_catalog.csv` is not an integer dtype, the script re-encodes unique values to dense integer IDs.

**Invariants**
- `n_items` must match the `item_map.csv` universe.
- `theme_list` defines the theme index space used by DIN (and later by copy-head via exported theme embeddings).

### 3.2.10 `user_embeddings.csv` and `theme_embeddings.csv`
**Produced by:** `din/export_user_theme_embeddings.py`

**Consumed by:** `train_copy_head.py`, `score_copies_with_din_and_copy_head.py`

Note: `build_copy_head_dataset.py` only reads these in its **toy fallback** path (when `--proxy_interactions_csv` is missing) to infer embedding-table sizes.

**Schema**
- `user_embeddings.csv`: `user_id` (int) + `emb_0..emb_{D-1}` (float)
- `theme_embeddings.csv`: `theme_id` (int) + `emb_0..emb_{D-1}` (float)

**Invariants**
- The IDs here are **embedding row indices** created as `0..N-1` in export, not original Talkwalker string IDs.
- `theme_id` must match the theme indexing implied by `theme_list` inside `din/dataset.pkl`.

### 3.2.11 `copy_embeddings.csv` / `item_embeddings.csv`
**Produced by:** `build_copy_embeddings.py`

**Consumed by:**
- mapping/reporting: `report_item_to_copy_mapping.py`
- copy-head dataset building (nearest-text mapping): `build_copy_head_dataset.py`

**Schema (generic; depends on `--id_col`)**
- `<id_col>` (often `copy_id` or `item_idx`)
- `emb_0..emb_{D-1}` (float)

**Embedding-space invariants**
- If you want item↔copy mapping in a single text space:
	- run `build_copy_embeddings.py --mode fit` on one side (commonly the copy catalog) to create artifacts
	- then run `--mode transform` on the other side using the same `--artifact_prefix`

### 3.2.12 `item_to_copy_map.csv` (optional persisted mapping)
**Produced by:** `report_item_to_copy_mapping.py` when `--out_mapping_csv` is provided

**Consumed by:** `build_copy_head_dataset.py` via `--item_to_copy_map_csv`

**Schema**
- `item_idx` (int)
- `copy_id` (int)
- `sim` (float32; cosine similarity)
- `sim_gap` (float32; top1 - top2 similarity gap)

**Invariants**
- `item_idx` must match `item_map.csv` and `item_catalog.csv`.
- `copy_id` must match the copy embedding table and copy catalog.

### 3.2.13 `copy_head_train_samples.csv` (copy-head supervised samples)
**Produced by:** `build_copy_head_dataset.py` (default output name in code: `copy_train_samples.csv`)

**Consumed by:** `train_copy_head.py` and `score_copies_with_din_and_copy_head.py` (as “training positives” source for contexts)

**Naming note**
- Any filename/path is fine as long as you pass it consistently.
- The dataset builder defaults to `copy_train_samples.csv`, while the scorer defaults to `copy_head_train_samples.csv`.
	- Either align filenames, or pass `--train_samples_csv` to `score_copies_with_din_and_copy_head.py`.

**Schema (from code; required columns)**
- `user_id` (int; dense index)
- `theme_id` (int; dense index)
- `copy_id` (int; dense index)
- `label` (float; 1.0 for positives, 0.0 for sampled negatives)

**Optional columns**
- `sample_weight` (float): created when `--weight_column` is provided

**How positives are selected (high-level)**
- A row is positive if it meets either:
	- `label_column >= label_positive_value`, or
	- `engagement_column >= min_engagement`

Negatives are then sampled per positive (`--num_neg_per_pos`).

### 3.2.14 Ranking CSV (optional operational output)
**Produced by:** `score_copies_with_din_and_copy_head.py` when `--out_csv` is provided

The output filename/path is whatever you pass via `--out_csv` (the docs often use `final_ranking.csv` as a convention).

**Schema (as written by the scorer)**
- `rank` (int; 1..N)
- `copy_id` (int)
- `score` (float)
- `copy_text` (string)

**Operational use**
- A simple operational default is to use the row where `rank == 1`.

## 3.3 Copy Catalog
The “copy catalog” is the source-of-truth table for the copy candidates you want to rank.

There are two slightly different uses of this table in the repo:

1) **Text embedding input** (`build_copy_embeddings.py`)
2) **Text display** (`score_copies_with_din_and_copy_head.py`) and (optional) **direct item→copy mapping** (`build_copy_head_dataset.py`, recommended only when ID universe match is verified)

Because of that, the safest way to think about the copy catalog contract is:

- It must have an **ID column** and a **text column**.
- The ID values should be consistent with the IDs used in `copy_embeddings.csv` and the copy-head sample CSV.

### 3.3.1 Minimal schema
With default CLI args, scripts expect:

- `copy_id` (integer)
- `copy_text` (string)

Important: `copy_id` is treated as an **embedding row index** downstream (not an arbitrary business ID). In practice it should be a dense range like `0..N-1` with no gaps.

If your data uses different names, pass them explicitly:
- `build_copy_embeddings.py --id_col ... --text_col ...`
- `score_copies_with_din_and_copy_head.py --copy_catalog_id_col ... --copy_catalog_text_col ...`

### 3.3.2 Recommended “operational” schema
For real runs, you will usually want extra columns for traceability and debugging.

Recommended additions:
- `copy_key` (string; original business ID)
- `language` / `locale`
- `campaign_id` / `adgroup_id` / `creative_id` (if applicable)
- optional theme metadata (if you want theme-aware analysis outside the model)

Keep `copy_id` as the dense numeric ID used by embedding joins; keep your original IDs in separate columns.

### 3.3.3 Direct mapping mode (optional)
`build_copy_head_dataset.py` supports `--item_to_copy=direct`, which uses a mapping inside the copy catalog.

This mode is recommended only when you have verified that proxy `item_id` values match the copy-catalog mapping key column.

In this mode, you must provide (names configurable by flags):
- `copy_catalog_id_col`: the integer `copy_id`
- `copy_catalog_item_col`: a column that matches the proxy `item_id` you want to map from

This is only needed if you already have a deterministic item→copy mapping and want to avoid nearest-text mapping.

## 3.4 Tag & Version Conventions
This repo is artifact-driven: most “bugs” in practice are contract mismatches across runs (maps from run A + embeddings from run B, etc.). The conventions below are not enforced by code everywhere, but they are aligned with how the scripts behave.

### 3.4.1 Timestamp tags
`build_proxy.py` writes its outputs with a timestamp tag in the filename:

- `proxy_items_<tag>.csv`
- `proxy_interactions_<tag>.csv`
- `proxy_din_interactions_<tag>.jsonl`

Where `<tag>` is generated as:

- `%Y%m%d_%H%M%S` (e.g., `20260122_134501`)

The tag is meant to represent “a concrete, reproducible data snapshot.”

### 3.4.2 Keep index-space artifacts together
At minimum, treat the following as an atomic bundle:

- `user_map.csv`
- `item_map.csv`
- `item_catalog.csv`
- `din/dataset.pkl`
- `user_embeddings.csv` / `theme_embeddings.csv`

These define:
- the dense index spaces (`user_idx`, `item_idx`)
- the theme index space (`theme_id`)
- the embedding row IDs used by later TF2/Keras stages

If any one of these comes from a different run, you can get silent joins (rows dropped) or shifted meanings (e.g., theme IDs no longer refer to the same semantics).

### 3.4.3 Copy-embedding artifact prefixes
`build_copy_embeddings.py` uses an explicit artifact prefix to keep text spaces stable:

- `<prefix>.vocab.json`
- `<prefix>.svd.npz`

If you plan to map items to copies by text similarity, the copy and item embeddings must be produced with the same artifact prefix (fit once, transform elsewhere).

### 3.4.4 Suggested run directory layout (convention)
The scripts accept explicit paths, so you can adopt a consistent layout like:

- `runs/<tag>/proxy/...`
- `runs/<tag>/twotower_data/...`
- `runs/<tag>/twotower_runs/...`
- `runs/<tag>/din/...`
- `runs/<tag>/copy/...`

The essential principle is not the folder names; it is that artifacts that share an index space are versioned and moved together.

