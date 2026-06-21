# 8. Validation & QA
This section is deliberately “boring” and operational: it is meant to catch contract breaks early.

## 8.1 Regression Checklist (per run)

Run these checks whenever you:
- change preprocessing logic,
- retrain a model,
- swap environments,
- or mix artifacts across machines.

**A) Proxy artifacts (raw → proxy)**
- Proxy interactions contain the required columns used downstream (`user_id`, `item_id`, `timestamp`, `label`, `engagement_total`).
- DIN-friendly JSONL events include `hist_item_ids`, `hist_len`, and `timestamp` fields.
- Basic sanity: non-empty rows, no obviously broken timestamps (all-null / not parseable).

**B) Index-space artifacts (conversion output)**
- `user_map.csv` is a bijection-like mapping (no duplicate `user_id`, `user_idx` dense in `[0..max]`).
- `item_map.csv` is a bijection-like mapping (no duplicate `item_id`, `item_idx` dense in `[0..max]`).
- Pair tables contain `user_idx`, `item_idx`, `label` and have non-empty train/val splits.
- `item_catalog.csv` joins on `item_idx` and has expected metadata columns (at minimum, `item_idx` plus any optional feature columns you plan to use).

**C) Two-Tower teacher outputs**
- Teacher checkpoint exists if you trained in a mode that saves it.
- Teacher slate JSONL exists only if you enabled candidate dumping.
- Candidate JSONL sanity: each record has `user_idx`, `candidate_item_idx[]`, `candidate_scores[]`, and `k` is consistent.

**D) DIN dataset + DIN outputs**
- `dataset.pkl` exists and is readable by the DIN scripts.
- Theme/cate lists are aligned to the item index space (length equals `n_items`).
- Exported `user_embeddings.csv` / `theme_embeddings.csv` have:
	- an integer ID column (`user_id` / `theme_id`), and
	- `emb_*` columns.

**E) Copy layer outputs**
- `copy_embeddings.csv` has `copy_id` and `emb_*` columns.
- Copy-head training samples contain required columns (`user_id`, `theme_id`, `copy_id`, `label`) and optionally `sample_weight`.
- Copy-head model file is loadable by the scorer.
- Ranking CSV (if produced) has `rank`, `copy_id`, `score`, `copy_text`.

## 8.2 Artifact Consistency

Most “mysterious” failures are mismatched index spaces. The non-negotiable invariants are:

- **Map/index alignment**
	- `max(user_idx) + 1` equals the number of rows in any user-indexed embedding table.
	- `max(item_idx) + 1` equals the item universe size used by teacher slates and DIN dataset metadata.
- **Theme alignment**
	- `theme_id` used in copy-head samples must be compatible with the theme embedding table exported from DIN.
	- If you exclude a theme ID during sample building or scoring, ensure that exclusion is applied consistently (don’t compare runs that excluded different theme IDs without noting it).
- **Embedding-space alignment (text embeddings)**
	- If you map items to copies by embedding similarity, item embeddings and copy embeddings must be produced from the same text-embedding artifacts (same vocab/IDF/SVD components).

Practical rule: treat each run as an atomic bundle (maps + dataset + embeddings + models). If you mix bundles, expect silent joins and shifted semantics.

## 8.3 Sanity Dashboards / Notebooks

You do not need fancy tooling here; the goal is fast “shape checks” and distribution checks.

Minimum useful plots/tables (any notebook or ad-hoc script is fine):

- **Proxy stage**: events per user (histogram), `hist_len` distribution, label rate, engagement distribution.
- **Two-Tower**: train/val size, positive rate after negative sampling, optional metrics curve over epochs.
- **Teacher slates**: distribution of `k`, count of users covered, basic score stats (min/mean/max).
- **DIN**: training loss trend, fraction of users/items missing from candidates after filtering.
- **Copy-head samples**: positives per theme, share of excluded theme (if any), `copy_id` frequency skew.
- **Ranking**: top-K outputs look plausible; score distribution; are a few copies dominating the top ranks?

## 8.4 Ranking Diff How-To

When you change something (data, mapping thresholds, model, exclusion rules), compare two ranking outputs.

Use placeholders and keep this workflow file-name agnostic:

- Inputs: `<RANKING_A_CSV>`, `<RANKING_B_CSV>` (both must contain `copy_id`, `rank`, `score`).
- Join on `copy_id`, then compute:
	- `delta_rank = rank_b - rank_a` (negative means “moved up”),
	- `delta_score = score_b - score_a`.
- Report:
	- Top movers by absolute `delta_rank`,
	- Top new entrants into top-K,
	- Score correlation on the intersection.

If you want a quick one-liner style diff, you can do it in Python with pandas:

```bash
python - <<'PY'
import pandas as pd

a = pd.read_csv("<RANKING_A_CSV>")
b = pd.read_csv("<RANKING_B_CSV>")

need = {"copy_id","rank","score"}
for name, df in [("A", a), ("B", b)]:

    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"{name} missing columns: {sorted(miss)}")

m = a[["copy_id","rank","score"]].merge(

    b[["copy_id","rank","score"]], on="copy_id", how="inner", suffixes=("_a","_b")
)
m["delta_rank"] = m["rank_b"] - m["rank_a"]
m["delta_score"] = m["score_b"] - m["score_a"]

print("[joined]", len(m), "copies")
print("[top movers] by |delta_rank|")
print(m.reindex(m["delta_rank"].abs().sort_values(ascending=False).index).head(20))
print("[score corr]", m[["score_a","score_b"]].corr().iloc[0,1])
PY
```

