import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return mat / denom


def _percentiles(x: np.ndarray, ps=(0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {p: float("nan") for p in ps}
    return {p: float(np.percentile(x, p)) for p in ps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--item_embeddings_csv", type=Path, required=True,
                    help="CSV: item_idx, emb_* (e.g., item_embeddings_<tag>.csv)")
    ap.add_argument("--copy_embeddings_csv", type=Path, required=True,
                    help="CSV: copy_id, emb_* (e.g., copy_embeddings_<tag>.csv)")
    ap.add_argument("--item_id_col", default="item_idx")
    ap.add_argument("--copy_id_col", default="copy_id")
    ap.add_argument("--item_catalog_csv", type=Path, default=None,
                    help="Optional: item_catalog.csv to print example item texts")
    ap.add_argument("--item_text_col", default="text")
    ap.add_argument("--topk_examples", type=int, default=3,
                    help="How many sample items to print per copy")
    ap.add_argument("--out_mapping_csv", type=Path, default=None,
                    help="Optional output mapping CSV (item_idx, copy_id, sim, sim_gap)")
    args = ap.parse_args()

    item_df = pd.read_csv(args.item_embeddings_csv)
    copy_df = pd.read_csv(args.copy_embeddings_csv)

    if args.item_id_col not in item_df.columns:
        raise SystemExit(f"{args.item_embeddings_csv} must contain column {args.item_id_col}")
    if args.copy_id_col not in copy_df.columns:
        raise SystemExit(f"{args.copy_embeddings_csv} must contain column {args.copy_id_col}")

    item_emb_cols = [c for c in item_df.columns if c.startswith("emb_")]
    copy_emb_cols = [c for c in copy_df.columns if c.startswith("emb_")]
    if not item_emb_cols or not copy_emb_cols:
        raise SystemExit("Embedding CSVs must contain emb_* columns")
    if len(item_emb_cols) != len(copy_emb_cols):
        raise SystemExit(f"Embedding dim mismatch: items={len(item_emb_cols)} copies={len(copy_emb_cols)}")

    item_ids = item_df[args.item_id_col].astype("int64").values
    copy_ids = copy_df[args.copy_id_col].astype("int64").values

    item_mat = _l2_normalize(item_df[item_emb_cols].values.astype("float32"))
    copy_mat = _l2_normalize(copy_df[copy_emb_cols].values.astype("float32"))

    sims = item_mat.dot(copy_mat.T).astype("float32")

    top2_idx = np.argpartition(sims, kth=-2, axis=1)[:, -2:]
    top2_sims = np.take_along_axis(sims, top2_idx, axis=1)
    order = np.argsort(top2_sims, axis=1)
    best_pos = order[:, 1]
    second_pos = order[:, 0]
    best_j = top2_idx[np.arange(len(item_ids)), best_pos]
    second_j = top2_idx[np.arange(len(item_ids)), second_pos]
    best_sim = top2_sims[np.arange(len(item_ids)), best_pos]
    second_sim = top2_sims[np.arange(len(item_ids)), second_pos]
    sim_gap = (best_sim - second_sim).astype("float32")

    mapped_copy = copy_ids[best_j]

    counts = pd.Series(mapped_copy).value_counts().sort_index()
    count_stats = _percentiles(counts.values, ps=(0, 5, 10, 25, 50, 75, 90, 95, 99, 100))

    sim_stats = _percentiles(best_sim, ps=(0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100))
    gap_stats = _percentiles(sim_gap, ps=(0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100))

    print("=== item -> copy mapping report ===")
    print(f"items={len(item_ids)} copies={len(copy_ids)} dim={len(item_emb_cols)}")
    print("[counts] mapped items per copy (percentiles)")
    for p in sorted(count_stats.keys()):
        print(f"  p{p:>3}: {count_stats[p]:.0f}")

    top = counts.sort_values(ascending=False)
    print("[counts] top copies by mapped items")
    for cid, cnt in top.head(10).items():
        print(f"  copy_id={int(cid)}\titems={int(cnt)}\tshare={cnt/len(item_ids):.3f}")

    print("[sim] top1 cosine similarity (percentiles)")
    for p in sorted(sim_stats.keys()):
        print(f"  p{p:>3}: {sim_stats[p]:.4f}")

    print("[gap] top1-top2 cosine gap (percentiles)")
    for p in sorted(gap_stats.keys()):
        print(f"  p{p:>3}: {gap_stats[p]:.4f}")

    if args.item_catalog_csv and args.item_catalog_csv.exists():
        ic = pd.read_csv(args.item_catalog_csv, usecols=["item_idx", args.item_text_col])
        ic["item_idx"] = ic["item_idx"].astype("int64")
        text_map = dict(zip(ic["item_idx"].values, ic[args.item_text_col].astype(str).values))

        print("[examples] top items per copy (by similarity)")
        for cid in copy_ids:
            mask = mapped_copy == cid
            if not np.any(mask):
                continue
            idxs = np.where(mask)[0]
            idxs = idxs[np.argsort(best_sim[idxs])[::-1]]
            take = idxs[: max(0, int(args.topk_examples))]
            print(f"  copy_id={int(cid)} mapped_items={int(mask.sum())}")
            for k, ii in enumerate(take, start=1):
                iid = int(item_ids[ii])
                simv = float(best_sim[ii])
                txt = text_map.get(iid, "")
                txt = (txt[:140] + "...") if len(txt) > 140 else txt
                print(f"    {k}. sim={simv:.4f}\titem_idx={iid}\ttext={txt}")

    if args.out_mapping_csv:
        out = pd.DataFrame(
            {
                "item_idx": item_ids,
                "copy_id": mapped_copy.astype("int64"),
                "sim": best_sim.astype("float32"),
                "sim_gap": sim_gap.astype("float32"),
            }
        )
        out.to_csv(args.out_mapping_csv, index=False)
        print(f"[write] mapping -> {args.out_mapping_csv}")


if __name__ == "__main__":
    main()
