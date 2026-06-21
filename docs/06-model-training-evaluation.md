# 6. Model Training & Evaluation
This section focuses on *how the models are actually trained and evaluated in this repo*, what files they read/write, and how to interpret the numbers you see in the console and artifacts under `runs/<tag>/`.

The main thing to keep in mind is that this pipeline is usually trained on **proxy interactions** and often on **pseudo-users** (see Stage 1). That means the evaluation metrics are best interpreted as:
- “Are we learning a consistent preference signal under this proxy definition?” (good for debugging and iteration), not
- “Is this the true online CTR?” (not directly measurable here).

---

## 6.1 Two-Tower Training

**Script:** `train_twotower.py` (PyTorch)

**Role in the pipeline:**
- Trains a fast retrieval model that produces a **teacher checkpoint** (`twotower_best.pt`) and/or a **teacher candidate slate** (`tt_candidates_topK.jsonl`) for DIN KD.
- In this repo, the Two-Tower is primarily a *candidate generator and teacher*, not necessarily a generalizable “user model” for unseen real users.

### Inputs

From Stage 1 (Two-Tower contracts):
- `twotower_train_pairs.csv` and `twotower_val_pairs.csv` (must contain `user_idx,item_idx,label`)
- `user_map.csv` / `item_map.csv` (string IDs → dense indices)
- Optional: `item_catalog.csv` (for numeric proxy features)

### What the model is

In code it is:
- `user_emb[user_idx]` and `item_emb[item_idx]`
- score = dot-product of the two vectors
- trained with `BCEWithLogitsLoss` on the pair labels

Optional item-feature path:
- If you pass `--use_item_features` and `--item_catalog`, the trainer will automatically pick numeric columns that exist in `item_catalog.csv` (e.g. `proxy_ctr_composite`, `proxy_volume`, `proxy_eng_rate`, `proxy_sent`, recency variants, sentiment/engagement totals, etc.).
- Those numeric columns are standardized (mean/std) and passed through a small MLP, then combined with item embeddings using either:
	- `--combine_item sum` (default): `item_vec = item_emb + mlp(feats)`
	- `--combine_item concat`: `item_vec = proj([item_emb, mlp(feats)])`

This “numeric feature assist” is helpful when you want the model to exploit stable proxy statistics without touching text.

### Training command (recommended)

This is the same “happy path” as Stage 1, but with the most important training-time options explained:

```bash
python train_twotower.py \
	--train_pairs $RUN_DIR/twotower_data/twotower_train_pairs.csv \
	--val_pairs   $RUN_DIR/twotower_data/twotower_val_pairs.csv \
	--user_map    $RUN_DIR/twotower_data/user_map.csv \
	--item_map    $RUN_DIR/twotower_data/item_map.csv \
	--item_catalog $RUN_DIR/twotower_data/item_catalog.csv \
	--use_item_features \
	--dim 64 \
	--batch_size 2048 \
	--epochs 10 \
	--lr 1e-3 \
	--weight_decay 1e-5 \
	--k 10,20 \
	--eval_sample_k 1000 \
	--early_metric auc \
	--save_best \
	--save_metrics \
	--out_dir $RUN_DIR/twotower_runs
```

Key flags grounded in code:
- `--ensure_negatives N`: if your train pairs do not have enough negatives, the trainer can *add random negatives* so that negatives ≥ positives × N.
- `--eval_all_items`: ranking metrics computed against **all items** (can be expensive). By default the trainer does *sampled ranking eval* using `--eval_sample_k` negatives per user.
	- There is a safety guard: if `n_items > --eval_all_items_max_guard` (default `200000`), the script automatically switches back to sampled eval.
- `--early_metric auc|ndcg`: controls which metric defines “best checkpoint”.
	- `auc` uses pointwise ROC-AUC.
	- `ndcg` uses `ndcg@k0` where `k0` is the first value from `--k`.
- `--save_best`: writes `$OUT_DIR/twotower_best.pt` only when the chosen early metric improves.
- `--save_metrics`: writes a JSONL file (default name `metrics.jsonl`) with one record per epoch.

### Evaluation: what gets computed

Every epoch, the trainer computes:
- Pointwise metrics (on `val_pairs`):
	- ROC-AUC and PR-AUC **only if** scikit-learn is available and the validation labels contain both classes.
	- Otherwise, it prints `nan`.
- Ranking metrics (grouped by `user_idx` on `val_pairs`):
	- `recall@k` and `ndcg@k` for each `k` in `--k`.
	- If `--eval_all_items` is off (default), it ranks within the set `{positives + sampled negatives}`.
	- If `--eval_all_items` is on, it ranks against the full item universe.

Interpretation tip:
- AUC/PR-AUC tell you how well the model separates positives from negatives *on the proxy labels*.
- Recall/NDCG tell you whether a user’s proxy-positives appear in the top-ranked items.

### Candidate dumping (teacher slate creation)

You can dump a Top-K slate either:

1) **At the end of training** (most common):

```bash
python train_twotower.py \
	... \
	--save_best \
	--dump_candidates_k 200 \
	--dump_candidates_out $RUN_DIR/tt_candidates_topK.jsonl
```

2) **Without training (dump-only)** from a checkpoint (fast for re-generating slates):

```bash
python train_twotower.py \
	--mode dump \
	--epochs 0 \
	--train_pairs $RUN_DIR/twotower_data/twotower_train_pairs.csv \
	--val_pairs   $RUN_DIR/twotower_data/twotower_val_pairs.csv \
	--user_map    $RUN_DIR/twotower_data/user_map.csv \
	--item_map    $RUN_DIR/twotower_data/item_map.csv \
	--item_catalog $RUN_DIR/twotower_data/item_catalog.csv \
	--use_item_features \
	--load_checkpoint $RUN_DIR/twotower_runs/twotower_best.pt \
	--dump_candidates_k 200 \
	--dump_candidates_out $RUN_DIR/tt_candidates_topK.jsonl
```

What gets written into each JSONL line:
- `user_idx` plus (when available) original `user_id`
- `candidate_item_idx` plus (when available) `candidate_item_id`
- `candidate_scores`
- `score_type` (`dot` by default; can be `sigmoid` via `--dump_score_type`)

Practical guidance:
- Start with `--dump_score_type dot` unless you have a specific reason to squash.
	- In later docs we treat it as a generic “teacher score” because downstream scripts can apply their own transformations.

### Artifacts (Two-Tower)

Under `$RUN_DIR/twotower_runs/` you may have:
- `twotower_best.pt` (only if `--save_best`)
- `twotower_final.pt` (only if `--save_final`)
- `metrics.jsonl` (only if `--save_metrics`)
- Optional: `user_emb.npy`, `item_emb.npy`, plus copies of maps (only if `--export_embeddings`)

The checkpoint file is a “bundle” with:
- model state dict
- metadata: trainer version, dims, item feature column list, normalization stats, and the args used

That metadata is used to prevent accidental misuse across runs (e.g., mismatch of user/item counts or feature columns).

---

## 6.2 DIN KD Training

**Script:** `din/train_listwise_kd.py` (TensorFlow 1 graph mode via `tf.compat.v1`)

**Role in the pipeline:**
- Trains a DIN student model to re-rank Two-Tower candidates.
- The training signal is dominated by **teacher slates** (`tt_candidates_topK.jsonl`) and optionally strengthened by “where positives are” information.

### Inputs

Required:
- `din/dataset.pkl` from `make_din_dataset.py`
- `tt_candidates_topK.jsonl` from the Two-Tower dumping step
- `user_map.csv` / `item_map.csv` (used to resolve IDs when reading positives)

Optional but recommended:
- `proxy_din_interactions_*.jsonl` via `--proxy_jsonl`
	- When provided and readable, the script extracts positives directly from proxy events (`label == 1`).
	- If not provided, it falls back to scanning positives from the `train_set` inside `dataset.pkl`.

Important alignment constraint:
- The KD trainer only trains on users that appear in `tt_candidates_topK.jsonl` (it builds `users = sorted(cands.keys())`).
	- If your candidate file contains very few users (or none), KD will either train on a tiny subset or fail early.

### What the KD objective is (as implemented)

For each training example, it builds a K-sized candidate list and three distributions:

1) **Teacher distribution $q$**
- Take the teacher scores for the user’s slate.
- Apply temperature scaling and softmax:
$$q = \mathrm{softmax}(s/\tau)$$

2) **Student distribution $\pi$**
- The DIN model produces logits for the same slate.
- Apply the *same* temperature scaling and softmax:
$$\pi = \mathrm{softmax}(\ell/\tau)$$

3) **Positive-label distribution $y$ (optional)**
- If the script can identify positives for the user (from proxy JSONL or dataset), it marks which slate items are positives.
- It then normalizes positives into a distribution over the slate.

Then the total loss is:
- KD loss: $\mathrm{KL}(q\;||\;\pi)$
- Optional “label CE” on the positives distribution (only for users that have positives; controlled by a `label_mask`)
- Plus the original DIN pointwise loss term from the DIN model (`model.loss`), scaled by `--point_loss_weight`

Mixing coefficients are exposed as CLI flags:
- `--lambda_kd` (default `0.7`): how much weight goes to KD vs label-CE inside the listwise part
- `--point_loss_weight` (default `1.0`): how much of DIN’s original pointwise loss to add

### Training command (recommended)

```bash
PROXY_DIN=$(ls -1t $RUN_DIR/proxy/proxy_din_interactions_*.jsonl | head -1)

python din/train_listwise_kd.py \
	--dataset_pkl $RUN_DIR/din/dataset.pkl \
	--candidates $RUN_DIR/tt_candidates_topK.jsonl \
	--user_map $RUN_DIR/twotower_data/user_map.csv \
	--item_map $RUN_DIR/twotower_data/item_map.csv \
	--proxy_jsonl $PROXY_DIN \
	--out_dir $RUN_DIR/din_kd_runs \
	--epochs 3 \
	--batch_size 128 \
	--K 200 \
	--tau 1.5 \
	--lambda_kd 0.7 \
	--lr 5e-4 \
	--seed 42
```

Notes:
- `PROXY_DIN` should point to the exact `proxy_din_interactions_*.jsonl` for this run.
- The trainer pads/truncates candidates to exactly `K`.
- It also forces the current training target item to appear in the slate (injects it into the last position if needed), so every example is “about a target in-slate”.

### Checkpointing behavior (important)

This script does **not** run a validation loop.

Instead, it saves the “best” checkpoint purely by **training average total loss**:
- After each epoch it prints `[epoch X] total_loss=...`.
- If the epoch average total loss is lower than the current best, it saves:
	- `$OUT_DIR/din_kd_best.ckpt` (TF checkpoint prefix)

This is fine for:
- deterministic reproduction and pipeline continuity, and
- monitoring whether training is behaving sensibly,

but it is not a guarantee of best generalization.

## 6.3 Metrics & Monitoring

This repo intentionally keeps monitoring lightweight and file-based. The goal is reproducibility and fast debugging per tag.

### Two-Tower: what to log and where

Console output per epoch includes:
- training loss (average over training pairs)
- validation ROC-AUC / PR-AUC (when scikit-learn is available and labels are not degenerate)
- validation ranking metrics: `recall@k`, `ndcg@k`

For long runs, prefer saving metrics:
- Use `--save_metrics` to write `$RUN_DIR/twotower_runs/metrics.jsonl`.
- Each line is a JSON dict with `epoch`, `loss`, `roc_auc`, `pr_auc`, and ranking metrics.

Suggested practice:
- Commit the `runs/<tag>/twotower_runs/metrics.jsonl` artifact (or archive it) together with the `twotower_best.pt` bundle.
- When comparing tags, compare both:
	- the metrics trend, and
	- the upstream proxy label stats (Stage 1 sanity checks), because proxy drift can masquerade as “model improvement/regression”.

### DIN KD: what to monitor

The KD trainer prints:
- counts (`users_with_candidates`, `item_count`, `cate_count`, `theme_count`)
- whether positives came from `proxy_jsonl` or `dataset.pkl`
- per-epoch `total_loss`
- checkpoint save messages

Practical checks when `total_loss` looks wrong:
- If it explodes or becomes `nan`, check:
	- `K` vs candidate file quality (are candidate scores finite?)
	- whether your candidate JSONL includes users outside the dataset’s `user_count` (they are filtered)
	- whether your maps/dataset belong to the same run/tag
- If it decreases but you later see poor downstream behavior, consider tuning:
	- `--tau` (temperature)
	- `--lambda_kd` (KD vs CE balance)
	- `--point_loss_weight` (how much original pointwise loss to keep)

### Cross-stage monitoring (recommended minimal checklist)

For each tag, you should be able to answer these quickly from artifacts:
- Proxy: Are label rates and basic stats stable across tags?
- Two-Tower: Does validation AUC/Recall/NDCG look reasonable and stable?
- Teacher slate: Does `tt_candidates_topK.jsonl` have the expected number of users and `k`?
- DIN KD: Does `total_loss` decrease across epochs and produce a checkpoint?

This pipeline is designed so that if one stage looks off, you can usually pinpoint it by checking a small number of files under `runs/<tag>/` rather than re-running the entire system.

