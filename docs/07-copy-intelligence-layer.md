# 7. Copy Intelligence Layer

Stage 3 produced a global ranking file, but it may not have been obvious *what exactly the “copy intelligence” layer is doing*.

This section explains the core idea and the concrete contracts involved:
- DIN produces **user/theme embeddings** (dense index spaces).
- `build_copy_embeddings.py` produces **text embeddings** for both items and copy candidates (same latent space).
- `build_copy_head_dataset.py` turns proxy interactions into supervised samples: `(user_id, theme_id, copy_id, label[, sample_weight])`.
- `train_copy_head.py` trains a lightweight model that corrects “copy wording differences” conditional on a user/theme context.
- `score_copies_with_din_and_copy_head.py` uses recent positives to build theme contexts, then produces a **single global ranking** across all copies.

The most common source of bugs at this layer is mixing **different index universes** (e.g., embeddings exported from tag A but datasets built from tag B). When in doubt, keep everything under one `$RUN_DIR`.

---

## 7.1 Embedding Fit vs. Transform

**Script:** `build_copy_embeddings.py` (scikit-learn TF‑IDF + TruncatedSVD)

Despite the name, this script is a general tool:
- It reads *any* CSV with an ID column and a text column.
- It produces a dense embedding table with columns: `<id_col>, emb_0, emb_1, ...`.

There are two modes:

### Mode A: `fit` (define the embedding space)

When `--mode fit`:
- It tokenizes text using a simple regex tokenizer.
- It builds a TF‑IDF matrix.
- It learns a TruncatedSVD projection to dimension `--emb_dim`.
- It writes *both*:
	- embedding artifacts (vocab + IDF + SVD components), and
	- an embedding CSV for the input rows.

**Artifacts (by `--artifact_prefix`)**
- `<prefix>.vocab.json`: contains the vocabulary, the IDF vector, and the embedding dimension.
- `<prefix>.svd.npz`: contains SVD components.

**Important modeling detail (grounded in code):**
- Tokenization is `re.findall(r"[a-z0-9']+", text.lower())`.
	- This means non‑Latin scripts may produce few tokens and weak embeddings unless your text includes Latin/number tokens.
	- If your data is mostly non‑Latin, consider replacing this layer with a tokenizer/embedding approach that can represent your text well (character n‑grams, subword tokenizers, etc.).

Recommended: fit on `item_catalog.csv` (often the larger/more diverse text corpus), then transform `copy_catalog.csv` into that same space.

```bash
# Fit on item catalog (defines artifacts + writes item embeddings)
python build_copy_embeddings.py \
	--mode fit \
	--catalog_csv $RUN_DIR/twotower_data/item_catalog.csv \
	--id_col item_idx \
	--text_col text \
	--emb_dim 64 \
	--artifact_prefix $RUN_DIR/copy/copy_embedding_artifacts \
	--out_copy_emb $RUN_DIR/copy/item_embeddings.csv

# Transform copy catalog into the same space
python build_copy_embeddings.py \
	--mode transform \
	--catalog_csv $RUN_DIR/inputs/copy_catalog.csv \
	--id_col copy_id \
	--text_col copy_text \
	--artifact_prefix $RUN_DIR/copy/copy_embedding_artifacts \
	--out_copy_emb $RUN_DIR/copy/copy_embeddings.csv
```

Alternative (valid, but may behave differently): fit on `copy_catalog.csv` (defines the space), then transform `item_catalog.csv`.
The only hard requirement is that both runs share the same `--artifact_prefix`, but which side you `fit` on can change similarity behavior.

```bash
# Fit on copy catalog (defines artifacts + writes copy embeddings)
python build_copy_embeddings.py \
	--mode fit \
	--catalog_csv $RUN_DIR/inputs/copy_catalog.csv \
	--id_col copy_id \
	--text_col copy_text \
	--emb_dim 64 \
	--artifact_prefix $RUN_DIR/copy/copy_embedding_artifacts \
	--out_copy_emb $RUN_DIR/copy/copy_embeddings.csv

# Transform item catalog into the same space
python build_copy_embeddings.py \
	--mode transform \
	--catalog_csv $RUN_DIR/twotower_data/item_catalog.csv \
	--id_col item_idx \
	--text_col text \
	--artifact_prefix $RUN_DIR/copy/copy_embedding_artifacts \
	--out_copy_emb $RUN_DIR/copy/item_embeddings.csv
```

### Mode B: `transform` (embed another catalog in the same space)

When `--mode transform`:
- It loads the artifacts from `--artifact_prefix`.
- It computes TF‑IDF using the same vocabulary + IDF.
- It projects into the same SVD space.

Typical use: transform the item catalog so you can map each item to a copy by text similarity.

```bash
python build_copy_embeddings.py \
	--mode transform \
	--catalog_csv $RUN_DIR/twotower_data/item_catalog.csv \
	--id_col item_idx \
	--text_col text \
	--artifact_prefix $RUN_DIR/copy/copy_embedding_artifacts \
	--out_copy_emb $RUN_DIR/copy/item_embeddings.csv
```

If you used the **item-catalog fit** pattern above, this is the corresponding transform step for the copy catalog:

```bash
python build_copy_embeddings.py \
	--mode transform \
	--catalog_csv $RUN_DIR/inputs/copy_catalog.csv \
	--id_col copy_id \
	--text_col copy_text \
	--artifact_prefix $RUN_DIR/copy/copy_embedding_artifacts \
	--out_copy_emb $RUN_DIR/copy/copy_embeddings.csv
```

Resulting contracts:
- `$RUN_DIR/copy/copy_embeddings.csv` has `copy_id` and `emb_*`.
- `$RUN_DIR/copy/item_embeddings.csv` has `item_idx` and `emb_*`.
- Both are in the same latent space *because they share the same artifacts prefix*.

---

## 7.2 Theme Assignment & Dataset Construction

This stage connects “proxy item interactions” to “copy candidates” so you can train a copy-aware model.

Why this bridge is needed (important concept):
- The proxy interaction signal comes from Talkwalker-derived events like `(user_id, item_id, label/engagement_*)`.
- The ranking target universe is the copy catalog `(copy_id, copy_text)`.

In an “industry-clean” setup you typically have an explicit item↔copy mapping table (or some tracking instrumentation) that ties content exposure to the copy variant shown.
When you do **not** have that table, this repo’s `nearest_text` path is a pragmatic workaround: it maps each item to a “closest” copy by text similarity and effectively **transfers labels** from `(user,item)` to `(user,mapped_copy)`.
As a result, model quality depends heavily on the assumption that “items a user engages with are text-close to good copy candidates”.

There are two distinct mappings happening here:
1) **item → theme** (for context)
2) **item → copy** (to create supervised targets)

### 7.2.1 item → theme (`theme_id`)

**Script:** `build_copy_head_dataset.py`

Theme assignment is performed using `twotower_data/item_catalog.csv`:
- The script reads `item_idx` and `--theme_col` (default `theme_raw`).
- It converts the theme values (strings are allowed) into a dense integer space `theme_id = 0..T-1`.
	- Missing values are treated as `"NA"`.

Practical consequence:
- `theme_id` here is *not* the original theme string; it is a dense index.
- If you later want to interpret themes, you must keep the mapping logic consistent.

If you have an “unknown/NA theme” that dominates:
- Consider using `--exclude_theme_id` to drop it during dataset building.
	- This is often `0` (because `"NA"` is typically encountered early), but you should confirm by checking your `item_catalog.csv` distribution.

### 7.2.2 item → copy (`copy_id`)

The dataset builder supports two main strategies (plus an optional precomputed bridge file):

**A) `--item_to_copy nearest_text` (default)**
- Uses `item_embeddings.csv` and `copy_embeddings.csv`.
- L2-normalizes embeddings and picks the nearest copy for each item by cosine similarity.
- Requires that both embedding tables have matching `emb_*` dimensionality.

**Optional quality control via a precomputed mapping CSV**
- You can generate a mapping report (and optionally persist it) using:
	- `report_item_to_copy_mapping.py`
- That script computes, for each item:
	- best matching `copy_id`,
	- `sim` (top1 cosine similarity), and
	- `sim_gap` (top1 - top2).

If you pass `--item_to_copy_map_csv`, the dataset builder will merge mapping columns into the proxy interactions and you can filter:
- `--min_sim`: drop low-similarity mappings
- `--min_gap`: drop ambiguous mappings where top1 and top2 are too close

**B) `--item_to_copy direct`**
- Treats proxy `item_id` values as directly mappable to the copy catalog.
- This is **not recommended** unless you have validated that proxy `item_id` and the copy-catalog key column are truly the same identifier universe.
- Requires `--copy_catalog_csv` to contain:
	- `--copy_catalog_item_col` (default `copy_id`) that matches proxy `item_id` values, and
	- `--copy_catalog_id_col` (default `copy_id`) which is the integer `copy_id`.

Important caveat: the default `--copy_catalog_item_col=copy_id` is usually **not** what you want if your proxy `item_id` is a hashed/string identifier. In that common case, you must set `--copy_catalog_item_col` to the copy-catalog column whose values are the *same identifier universe* as proxy `item_id`; otherwise the mapping will be empty and `direct` will fail.

In practice, prefer `nearest_text` unless you can confidently prove the ID match.

### 7.2.3 Positive/negative sampling (what becomes a label)

The dataset builder reads proxy interactions and defines positives using an OR rule:
- Positive if `label_column >= label_positive_value`, OR
- Positive if `engagement_column >= min_engagement`.

Defaults:
- `--label_column label`, `--label_positive_value 1.0`
- `--engagement_column engagement_total`, `--min_engagement 1.0`

Then for each positive, it samples `--num_neg_per_pos` random negatives from the observed `copy_id` pool.

Weights (optional):
- If you set `--weight_column`, the script writes `sample_weight` for both positives and their sampled negatives.
	- Negatives inherit the same weight as their originating positive (this is intentional in code).

Important operational behavior:
- If `--proxy_interactions_csv` is missing or unreadable, the script falls back to a toy dataset built from embedding table index ranges.
	- For real runs you almost always want the proxy-driven path.

Recommended dataset build command (proxy-driven):

```bash
PROXY_INTER=$(ls -1t $RUN_DIR/proxy/proxy_interactions_*.csv | head -1)

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
	--copy_emb_csv $RUN_DIR/copy/copy_embeddings.csv \
	--label_column label \
	--label_positive_value 1.0 \
	--engagement_column engagement_total \
	--min_engagement 1.0 \
	--num_neg_per_pos 3 \
	--seed 42 \
	--out_csv $RUN_DIR/copy/copy_head_train_samples.csv
```

Expected output contract:
- `$RUN_DIR/copy/copy_head_train_samples.csv` with columns:
	- `user_id` (dense index, compatible with `din_user_embeddings.csv`)
	- `theme_id` (dense index, compatible with `din_theme_embeddings.csv`)
	- `copy_id` (must be a dense integer index aligned with `copy_embeddings.csv`)
	- `label` (0/1 float)
	- `sample_weight` (optional)

---

## 7.3 Ad Copy Scoring Training (TensorFlow 2)

**Scripts:**
- `copy_head_model.py` defines the model architecture.
- `train_copy_head.py` loads embeddings + samples and trains the model.

### What the copy-head model learns

The copy-head is a lightweight MLP that predicts the probability of a positive outcome given:
- a user embedding vector (from DIN export),
- a theme embedding vector (from DIN export), and
- a copy embedding vector (from TF‑IDF+SVD).

The architecture (from `copy_head_model.py`) is:
- inputs: `user_emb`, `theme_emb`, `copy_emb`
- feature construction: `delta = copy_emb - theme_emb`, then `concat = [user_emb, theme_emb, delta]`
- MLP: Dense(128) → Dropout → Dense(64) → Dropout → Dense(1) → sigmoid
- loss: binary cross entropy
- metric: AUC

### ID semantics (common pitfall)

The training script does not “join by ID” in a database sense. It does **direct indexing**:
- It sorts the embedding CSVs by `user_id/theme_id/copy_id`.
- It treats the integer IDs in `copy_head_train_samples.csv` as **row indices** into the embedding matrices.
- It bounds-checks that the max ID is within the embedding table size.

That means these IDs must behave like array indices:
- `user_id` and `theme_id` are safe because `din/export_user_theme_embeddings.py` exports them as `0..N-1`.
- `copy_id` must also be `0..(num_copies-1)` and must match the rows in `copy_embeddings.csv`.

If your copy catalog uses arbitrary IDs (e.g., UUID-like integers or sparse IDs), you must remap them to a dense `copy_id` space before:
- building `copy_embeddings.csv`,
- building `copy_head_train_samples.csv`, and
- training the copy-head.

So if you see an error like “index exceeds available embeddings,” it almost always means:
- the samples were built from a different run/tag than the embeddings, or
- your copy catalog and copy embeddings do not match.

### Sample weights (optional)

If `copy_head_train_samples.csv` contains a `sample_weight` column:
- `train_copy_head.py` applies a transform (default `log1p`) and then normalizes by mean.
- You can change this behavior via:
	- `--weight_transform none|log1p|sqrt`
	- `--disable_weight_normalize`

### Training command (recommended)

```bash
python train_copy_head.py \
	--user_emb_csv $RUN_DIR/din/din_user_embeddings.csv \
	--theme_emb_csv $RUN_DIR/din/din_theme_embeddings.csv \
	--copy_emb_csv $RUN_DIR/copy/copy_embeddings.csv \
	--train_samples_csv $RUN_DIR/copy/copy_head_train_samples.csv \
	--out_model $RUN_DIR/copy/copy_head_model.keras \
	--epochs 10 \
	--batch_size 256 \
	--seed 42
```

Implementation note grounded in code:
- Both `train_copy_head.py` and `score_copies_with_din_and_copy_head.py` disable GPU visibility for predictability.
	- This is helpful for consistent runs, but if you want GPU training you will need to adjust the scripts.

---

## 7.4 Global Scoring & Theme Weighting

**Script:** `score_copies_with_din_and_copy_head.py`

Goal: produce a single global ranking over copy candidates:
- It builds “theme contexts” from training positives.
- It scores every copy under each theme context using the trained copy-head.
- It aggregates theme-specific scores into a final global score via a weighted sum.

### 7.4.1 Theme contexts (what is a “context” here?)

The scorer reads `copy_head_train_samples.csv` and keeps rows with `label >= 1.0`.

For each theme:
- It computes a weighted average of user vectors:
	- `user_vec(theme) = Σ (user_emb(user) * w) / Σ w`
- It tracks:
	- `raw_weight(theme) = Σ w` (if no weights are present, this is proportional to positive count)
	- `count(theme) = number of positives`

Filtering:
- `--theme_min_positives` (default `10`) drops themes with fewer positives, unless nothing qualifies.

### 7.4.2 Theme weights (temperature is a power transform)

The script computes a probability distribution over themes based on `raw_weight`.

Important semantics (from code and GUIDE):
- `--theme_weight_temperature` is **not** a softmax temperature.
- It is an exponent/power transform:
$$w := \frac{(\mathrm{raw\_weight})^{\mathrm{temperature}}}{\sum_t (\mathrm{raw\_weight}_t)^{\mathrm{temperature}}}$$

Then it optionally mixes in a uniform prior:
$$w := (1-\alpha)w + \alpha\,\mathrm{Uniform}$$
where `α = --theme_uniform_mix`.

Practical effect:
- temperature < 1.0 (default 0.5) flattens differences between themes.
- mixing with uniform prevents a single theme from fully dominating if your positives are skewed.

### 7.4.3 Scoring all copies

For each theme context, the scorer runs the copy-head on every copy embedding:
- inputs are broadcasted so the model predicts one score per copy.
- then it adds `theme_weight * score` into a global score vector.

This implies runtime is roughly proportional to:
$$O(N_{\text{themes}} \times N_{\text{copies}})$$

So:
- If you have a very large copy catalog, consider raising `--theme_min_positives`, excluding the “unknown theme”, or limiting the copy catalog for quick iteration.

### Output

The script always prints:
- a short summary of top theme weights (`--context_summary_k`, default 5)
- top-ranked copies up to `--top_k`

It writes a CSV only if you pass `--out_csv`.

Recommended scoring command:

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

The output CSV (if enabled) includes:
- `rank`, `copy_id`, `score`, and `copy_text`.

Interpretation note:
- These scores are not calibrated click probabilities.
- They are a model-driven ranking score built from proxy positives, a theme-weighted context heuristic, and a copy-head trained on those proxy-derived samples.

