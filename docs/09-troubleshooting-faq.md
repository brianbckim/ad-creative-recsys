# 9. Troubleshooting & FAQ
This repository is intentionally “artifact-first”: each stage writes concrete CSV/JSONL/checkpoints that become the next stage’s input.

That design makes debugging much easier, but it also means most failures are not “model bugs” — they are **contract mismatches** (wrong file, wrong tag, wrong ID universe, wrong columns).

When something fails, start with this quick triage:

1) Identify the failing stage (Proxy / Two‑Tower / DIN / Copy layer).
2) Confirm you’re using a single run directory (`$RUN_DIR`) consistently.
3) Validate the input artifact schema (required columns) and ID semantics (string ID vs dense index).
4) Only then tune hyperparameters or worry about model quality.

## 9.1 Environment Issues

### “It doesn’t import” / dependency installation problems

Symptoms
- `ModuleNotFoundError: No module named ...`
- Import errors around `tensorflow`, `torch`, `sklearn`, or `keras`

What’s happening
- This repo mixes **PyTorch** (Two‑Tower) and **TensorFlow compat.v1** (DIN) and **TF2/Keras** (copy head / scoring).
- If your environment is half-installed (or mixing incompatible wheels), the first failure is usually during import.

What to do
- Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

- If you use multiple Python environments (conda/venv), confirm which interpreter you are running:

```bash
which python
python -c "import sys; print(sys.executable)"
python -c "import sys; print(sys.version)"
```

### Apple Silicon (macOS arm64) + TensorFlow / DIN

Symptoms
- TensorFlow installs, but DIN crashes or runs extremely slowly.
- You see architecture-related errors (arm64/x86_64 mismatch), or TF GPU acceleration doesn’t kick in.

What’s happening in this repo
- DIN scripts use `tf.compat.v1` and call `tf.disable_v2_behavior()`.
- On Apple Silicon, you typically want the `tensorflow-macos` + `tensorflow-metal` stack (see `requirements.txt`).
- Some compat.v1 graph workloads may still run on CPU depending on the ops involved; that is normal.

What to do
- Confirm your Python architecture matches your TF wheel:

```bash
python -c "import platform; print(platform.platform()); print('machine=', platform.machine())"
```

- If you intentionally run an x86_64 environment under Rosetta for compatibility, keep the entire toolchain consistent (Python + pip wheels + libraries all x86_64).

### “Why is my GPU not being used?” (copy head / scoring)

Symptoms
- You have a GPU-capable TensorFlow install, but `train_copy_head.py` and/or `score_copies_with_din_and_copy_head.py` runs on CPU.

What’s happening
- Both scripts explicitly disable GPU devices:
	- `train_copy_head.py`: `tf.config.experimental.set_visible_devices([], "GPU")`
	- `score_copies_with_din_and_copy_head.py`: same

Why this exists
- It reduces device-specific variability and avoids GPU/driver edge cases for a lightweight model.

What to do
- Treat CPU execution as the default for copy-head/scoring.
- If you truly need GPU acceleration, you can remove/adjust that line — but do that consciously and keep it consistent across runs.

### Two‑Tower metrics are `NaN` (AUC/PR-AUC)

Symptoms
- Training runs, but `roc_auc` / `pr_auc` are printed as `NaN`.

What’s happening
- `train_twotower.py` computes ROC-AUC / PR-AUC only if scikit‑learn is importable.
- If scikit‑learn isn’t installed (or fails to import), the code falls back to `NaN`.

What to do
- Ensure `scikit-learn` is installed and importable:

```bash
python -c "import sklearn; import sklearn.metrics; print('sklearn OK')"
```

### Out-of-memory (OOM) or very slow evaluation

Where it happens
- Two‑Tower ranking evaluation can become expensive if you evaluate against “all items”.
- Large candidate dumps (high `K`, many users) can blow up disk/time.
- Copy scoring is roughly proportional to `(#themes considered) × (#copies)`.

What to do
- Prefer sampled ranking evaluation (the default pattern) until you are stable.
- Reduce `--batch_size` (Two‑Tower / copy head) if you see memory spikes.
- Keep `K` moderate during iteration (DIN KD and any reranking step operates over that `K`).
- If copy catalog is huge, reduce the number of theme contexts used in scoring (for example, by raising the scorer’s minimum positives threshold or excluding an “unknown theme” bucket).

## 9.2 Data Contract Breakages

Most “mysterious” pipeline failures are explained by one of the following:

- **Wrong file**: mixing artifacts from different tags/run directories.
- **Wrong schema**: missing required columns.
- **Wrong ID semantics**: using a string ID where the code expects a dense integer index (or vice versa).

Below are the contracts that commonly break, along with the exact failure modes you’ll see.

### `convert_to_twotower.py` fails early (missing columns / label)

Common errors
- `Missing columns in interactions: {...}`
- `label_col '...' not found. Available: [...]`
- `items CSV must contain 'item_id'.`

What the script expects
- Interactions CSV must contain: `user_id`, `item_id`, `timestamp`, and a label column selected by `--label_col`.
- Timestamps are parsed with `pd.to_datetime(..., utc=True)` and invalid timestamps are dropped.

How to diagnose quickly
- Print the header and confirm timestamp parseability:

```bash
PROXY_INTERACTIONS=$(ls -1t $RUN_DIR/proxy/proxy_interactions_*.csv | head -n 1)

python - << PY
import pandas as pd

df = pd.read_csv("$PROXY_INTERACTIONS")
print(df.columns.tolist())
print(df[["timestamp"]].head())
frac_bad = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).isna().mean()
print(frac_bad, "fraction invalid timestamp")
PY
```

### `make_din_dataset.py` produces empty rows (or DIN training later fails)

Common symptoms
- `[load-jsonl] kept=0  skipped_user=...  skipped_item=...`
- `RuntimeError: No usable rows after mapping. Check maps and jsonl.`
- DIN training runs but behaves oddly because histories are not meaningful.

What’s happening
- The DIN dataset builder maps `user_id`/`item_id` into the dense index universe from `user_map.csv` / `item_map.csv`.
- If you pass a JSONL created from a different run (different maps), most rows won’t map.

What to do
- Ensure `--user_map` and `--item_map` come from the same `$RUN_DIR/twotower_data/` as the JSONL you’re using.
- If you care about chronological histories, prefer the JSONL path that includes timestamps.
	- Pair CSVs (`twotower_*_pairs.csv`) contain only `(user_idx,item_idx,label)` by default, so they cannot reconstruct real time order.

### `din/train_listwise_kd.py` errors about candidates/users

Common errors
- `RuntimeError: No users in candidates after filtering. Check tt_candidates_topK.jsonl and maps.`

What’s happening
- The KD trainer filters candidates by index range:
	- `user_idx` must be within `[0, user_count)`
	- `candidate_item_idx` must be within `[0, item_count)`
- If the candidate dump was created with different maps (or a different dataset), nearly everything gets filtered out.

What to do
- Treat `dataset.pkl`, `user_map.csv`, `item_map.csv`, and the Two‑Tower candidate JSONL as a *bundle*.
- Verify that `user_count`/`item_count` implied by `dataset.pkl` match the max indices in the candidate JSONL.

### DIN dataset format mismatch (pickle meta tuple)

Common error
- `ValueError: Unsupported meta tuple length: ...`

What’s happening
- `din/train_listwise_kd.py` expects a specific pickle payload structure and meta tuple length (3 or 4).
- If `dataset.pkl` was produced by a different script/version or hand-modified, the loader may not understand it.

What to do
- Recreate `dataset.pkl` using this repo’s `make_din_dataset.py` and keep it co-located with the maps and candidates you use.

### Copy embedding / mapping scripts complain about `emb_*` columns

Common errors
- `... must contain at least one column prefixed with 'emb_'`
- `Embedding dim mismatch: items=... copies=...`

What’s happening
- The embedding CSV contract in this repo is strict:
	- one ID column (`item_idx` or `copy_id`)
	- many embedding columns named `emb_0`, `emb_1`, ...
- Item embeddings and copy embeddings must have the same embedding dimension if you want cosine-based mapping.

What to do
- Confirm your embedding CSV columns:

```bash
python - << PY
import pandas as pd

df = pd.read_csv("$RUN_DIR/copy/copy_embeddings.csv")
print("cols:", df.columns[:10].tolist(), "...")
print("num emb cols:", sum(c.startswith("emb_") for c in df.columns))
PY
```

### Copy-head training fails with “index exceeds available embeddings”

Common error
- `copy_id index X exceeds available embeddings (size=Y). Ensure copy/user/theme IDs align with the embedding tables.`

What’s happening
- `train_copy_head.py` does **direct array indexing** into the embedding matrices.
- That means `user_id`, `theme_id`, `copy_id` inside `copy_head_train_samples.csv` are not join keys — they must be **dense indices aligned to the embedding tables**.

Most common root causes
- `copy_id` values are arbitrary IDs from a catalog rather than a 0..N-1 index aligned with `copy_embeddings.csv`.
- You rebuilt `copy_embeddings.csv` with a different catalog/version but reused an older training samples CSV.

What to do
- Ensure the copy catalog you use for scoring is the same catalog used to build `copy_embeddings.csv`.
- If your copy catalog has arbitrary IDs, remap them to a dense index once, and keep that mapping fixed across embedding build → dataset build → training → scoring.

### Final scoring fails: missing `copy_id` in embeddings

Common error
- `The following copy_id values are missing from copy_embeddings.csv: ... Re-run build_copy_embeddings.py using the same copy catalog.`

What’s happening
- `score_copies_with_din_and_copy_head.py` validates that every `copy_id` in the copy catalog has a corresponding row in `copy_embeddings.csv`.

What to do
- Re-run copy embedding generation using the exact same copy catalog file that you pass to the scorer.

## 9.3 Model Drift & Maintenance

Even if the pipeline runs end-to-end, you’ll see ranking changes over time. In this repo, drift typically comes from **data drift** and **artifact drift**, not random noise.

### What typically causes drift

- Proxy contract changes:
	- different Talkwalker export window
	- different cohort/pseudo-user grouping behavior
	- changes in engagement distributions that shift your proxy labels

- ID-universe changes:
	- new items/users appearing or disappearing will change `user_map.csv` / `item_map.csv` sizes and the meaning of indices

- Embedding space changes:
	- re-fitting TF‑IDF/SVD (new vocabulary, new SVD basis) changes the geometry of text similarity
	- a new embedding space invalidates previously computed item→copy mapping and any copy-head dataset built from it

- Candidate slate changes:
	- changing Two‑Tower model, `K`, or dump settings changes what DIN is trained to rank

### Maintenance practices that keep you sane

Treat each run tag as immutable
- Keep a single `$RUN_DIR` per tag and avoid mixing artifacts across tags.
- If you re-run a stage, prefer writing into a new tag/run directory so you can diff outputs.

Pin the text embedding space when you need comparability
- If you want item→copy similarity to be comparable across runs, keep the same TF‑IDF vocabulary + SVD artifacts (same `artifact_prefix`) for transform.
- If you intentionally refit, assume you must regenerate downstream artifacts:
	- item embeddings
	- copy embeddings
		- item→copy mapping report (if you rely on it / persist it)
	- copy-head training samples
	- copy-head model

Track a small set of “health numbers” per run
- Proxy stage:
	- number of items, number of interactions, label positive rate
- Two‑Tower stage:
	- number of users/items (from maps), evaluation metrics, candidate dump coverage
- DIN stage:
	- dataset sizes, KD training loss curve (note: best checkpoint selection is by training loss in this repo)
- Copy stage:
	- number of positives used for theme contexts, top theme weights, final top‑K copies

Be careful interpreting probabilities
- Many outputs are ranking scores trained on proxy labels.
- Changes in proxy construction can shift score distributions even if rankings remain acceptable.

When updating the copy catalog
- Expect ranking changes because:
	- copy texts changed
	- the TF‑IDF feature space coverage changed
	- the copy embedding table changed size/order
- Operationally: regenerate `copy_embeddings.csv` and ensure `copy_id` remains a stable dense index (or keep a stable remapping layer).

