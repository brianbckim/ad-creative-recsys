from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def load_embedding_table(path: Path, id_col: str) -> Tuple[Dict[int, np.ndarray], int]:
    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise SystemExit(f"{path} must contain column {id_col}")
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise SystemExit(f"{path} must contain at least one column prefixed with 'emb_'")
    ids = df[id_col].values.astype("int64")
    emb = df[emb_cols].values.astype("float32")
    mapping: Dict[int, np.ndarray] = {}
    for idx, eid in enumerate(ids):
        mapping[int(eid)] = emb[idx]
    return mapping, int(emb.shape[1])


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(mat, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return mat / denom


def build_item_to_copy_map_by_text(
    item_embeddings_csv: Path,
    copy_embeddings_csv: Path,
    item_id_col: str = "item_idx",
    copy_id_col: str = "copy_id",
) -> Dict[int, int]:
    item_df = pd.read_csv(item_embeddings_csv)
    copy_df = pd.read_csv(copy_embeddings_csv)

    if item_id_col not in item_df.columns:
        raise SystemExit(f"{item_embeddings_csv} must contain column {item_id_col}")
    if copy_id_col not in copy_df.columns:
        raise SystemExit(f"{copy_embeddings_csv} must contain column {copy_id_col}")

    item_emb_cols = [c for c in item_df.columns if c.startswith("emb_")]
    copy_emb_cols = [c for c in copy_df.columns if c.startswith("emb_")]
    if not item_emb_cols or not copy_emb_cols:
        raise SystemExit("Embedding CSVs must contain emb_* columns")
    if len(item_emb_cols) != len(copy_emb_cols):
        raise SystemExit(
            f"Embedding dim mismatch: items={len(item_emb_cols)} copies={len(copy_emb_cols)}"
        )

    item_ids = item_df[item_id_col].astype("int64").values
    item_mat = _l2_normalize(item_df[item_emb_cols].values.astype("float32"))

    copy_ids = copy_df[copy_id_col].astype("int64").values
    copy_mat = _l2_normalize(copy_df[copy_emb_cols].values.astype("float32"))

    sims = item_mat.dot(copy_mat.T)
    best = np.argmax(sims, axis=1)
    return {int(iid): int(copy_ids[j]) for iid, j in zip(item_ids, best)}

def load_embedding_index(path: Path, id_col: str) -> np.ndarray:
    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise SystemExit(f"{path} must contain column {id_col}")
    ids = df[id_col].values.astype("int64")
    if ids.min() < 0:
        raise SystemExit(f"{path}: {id_col} must be >= 0")
    return ids


def build_toy_samples(
    num_users: int,
    num_themes: int,
    num_copies: int,
    num_pos: int,
    num_neg_per_pos: int,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    user_ids = rng.integers(0, num_users, size=num_pos, dtype="int64")
    theme_ids = rng.integers(0, num_themes, size=num_pos, dtype="int64")
    copy_ids = rng.integers(0, num_copies, size=num_pos, dtype="int64")

    rows = []
    for u, t, c in zip(user_ids, theme_ids, copy_ids):
        rows.append({"user_id": u, "theme_id": t, "copy_id": c, "label": 1.0})
        for _ in range(num_neg_per_pos):
            neg_c = c
            while neg_c == c:
                neg_c = int(rng.integers(0, num_copies))
            rows.append({"user_id": u, "theme_id": t, "copy_id": neg_c, "label": 0.0})

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

def load_map_csv(path: Path, key_col: str, val_col: str) -> Tuple[Dict[str, int], int]:
    df = pd.read_csv(path)
    if not {key_col, val_col}.issubset(df.columns):
        raise SystemExit(f"{path} must contain columns {key_col},{val_col}")
    mapping = {str(k): int(v) for k, v in zip(df[key_col], df[val_col])}
    total = int(df[val_col].max()) + 1
    return mapping, total


def build_theme_mapping(item_catalog: Path, theme_col: Optional[str]) -> Dict[int, int]:
    df = pd.read_csv(item_catalog, low_memory=False)
    if "item_idx" not in df.columns:
        raise SystemExit(f"{item_catalog} must contain item_idx column")

    if theme_col and theme_col in df.columns:
        ser = pd.Series(df[theme_col], dtype="string").fillna("NA")
        unique_themes = pd.unique(ser)
        theme_lookup = {val: idx for idx, val in enumerate(unique_themes)}
        df["_theme_id"] = ser.map(theme_lookup).astype(int)
    else:
        df["_theme_id"] = 0

    return dict(zip(df["item_idx"].astype(int), df["_theme_id"].astype(int)))


def build_theme_mapping_from_copy_catalog(df: pd.DataFrame, id_col: str, theme_col: Optional[str]) -> Dict[int, int]:
    if not theme_col or theme_col not in df.columns:
        return {}
    ser = pd.Series(df[theme_col], dtype="string").fillna("NA")
    unique_themes = pd.unique(ser)
    theme_lookup = {val: idx for idx, val in enumerate(unique_themes)}
    ids = df[id_col].astype(int)
    mapping = {int(cid): int(theme_lookup[val]) for cid, val in zip(ids, ser)}
    return mapping


def load_copy_catalog_map(path: Path, id_col: str, item_col: str,
                          theme_col: Optional[str]) -> Tuple[Dict[str, int], Dict[int, int]]:
    df = pd.read_csv(path)
    need = {id_col, item_col}
    if not need.issubset(df.columns):
        raise SystemExit(f"{path} must contain columns {need}")
    id_map = {str(row[item_col]): int(row[id_col]) for _, row in df.iterrows()}
    theme_map = build_theme_mapping_from_copy_catalog(df, id_col, theme_col)
    return id_map, theme_map


def sample_neg_copy(rng: np.random.Generator, pool: np.ndarray, forbidden: Iterable[int]) -> int:
    forbid = set(forbidden)
    if len(forbid) >= len(pool):
        for cid in pool:
            if cid not in forbid:
                return int(cid)
        return int(pool[0])
    while True:
        cid = int(rng.choice(pool))
        if cid not in forbid:
            return cid


def build_samples_from_proxy(args: argparse.Namespace) -> pd.DataFrame:
    interactions = pd.read_csv(args.proxy_interactions_csv, low_memory=False)
    need_cols = {"user_id", "item_id"}
    if not need_cols.issubset(interactions.columns):
        raise SystemExit(f"{args.proxy_interactions_csv} must contain columns {need_cols}")

    user_map, num_users = load_map_csv(args.user_map_csv, "user_id", "user_idx")
    item_map, _ = load_map_csv(args.item_map_csv, "item_id", "item_idx")

    item_theme_map = build_theme_mapping(args.item_catalog_csv, args.theme_col)

    interactions["_raw_item_id"] = interactions["item_id"].astype(str)
    interactions["user_id"] = interactions["user_id"].map(user_map)
    interactions["item_idx"] = interactions["item_id"].map(item_map)
    interactions = interactions.dropna(subset=["user_id", "item_idx"])
    interactions["user_id"] = interactions["user_id"].astype(int)
    interactions["item_idx"] = interactions["item_idx"].astype(int)
    interactions["theme_id"] = interactions["item_idx"].map(item_theme_map)
    if args.exclude_theme_id is not None:
        interactions = interactions[
            interactions["theme_id"].astype("float").ne(float(args.exclude_theme_id))
        ]
    interactions = interactions.dropna(subset=["theme_id"])
    interactions["theme_id"] = interactions["theme_id"].astype(int)

    copy_id_map: Optional[Dict[str, int]] = None
    if args.copy_catalog_csv and args.copy_catalog_csv.exists():
        df_cat = pd.read_csv(args.copy_catalog_csv)
        if args.copy_catalog_item_col in df_cat.columns and args.copy_catalog_id_col in df_cat.columns:
            copy_id_map = {
                str(row[args.copy_catalog_item_col]): int(row[args.copy_catalog_id_col])
                for _, row in df_cat.iterrows()
            }

    if args.item_to_copy == "direct" and copy_id_map is None:
        raise SystemExit(
            "--item_to_copy=direct requires copy_catalog_csv containing both "
            f"'{args.copy_catalog_item_col}' and '{args.copy_catalog_id_col}'."
        )

    if args.item_to_copy_map_csv and args.item_to_copy_map_csv.exists():
        m = pd.read_csv(args.item_to_copy_map_csv)
        need = {"item_idx", "copy_id"}
        if not need.issubset(m.columns):
            raise SystemExit(f"{args.item_to_copy_map_csv} must contain columns {need}")
        m = m[[c for c in ["item_idx", "copy_id", "sim", "sim_gap"] if c in m.columns]].copy()
        m["item_idx"] = m["item_idx"].astype(int)
        m["copy_id"] = m["copy_id"].astype(int)

        interactions = interactions.merge(m, on="item_idx", how="left")
        interactions["copy_id"] = interactions["copy_id"].astype("float")
        if "sim" in interactions.columns and args.min_sim > 0:
            interactions = interactions[interactions["sim"].astype(float) >= float(args.min_sim)]
        if "sim_gap" in interactions.columns and args.min_gap > 0:
            interactions = interactions[interactions["sim_gap"].astype(float) >= float(args.min_gap)]
    else:
        if args.item_to_copy == "direct":
            interactions["copy_id"] = interactions["_raw_item_id"].map(lambda x: copy_id_map.get(str(x)))
        else:
            if not (args.item_embeddings_csv and args.item_embeddings_csv.exists()):
                raise SystemExit("--item_to_copy=nearest_text requires --item_embeddings_csv")
            if not (args.copy_emb_csv and args.copy_emb_csv.exists()):
                raise SystemExit("--item_to_copy=nearest_text requires --copy_emb_csv")
            item_to_copy = build_item_to_copy_map_by_text(
                args.item_embeddings_csv,
                args.copy_emb_csv,
                item_id_col="item_idx",
                copy_id_col="copy_id",
            )
            interactions["copy_id"] = interactions["item_idx"].map(item_to_copy)

    interactions = interactions.dropna(subset=["copy_id"])
    interactions["copy_id"] = interactions["copy_id"].astype(int)
    num_copies = int(interactions["copy_id"].nunique())

    pos_mask = pd.Series(False, index=interactions.index)
    if args.label_column and args.label_column in interactions.columns:
        pos_mask |= interactions[args.label_column].astype(float) >= args.label_positive_value
    if args.engagement_column and args.engagement_column in interactions.columns:
        pos_mask |= interactions[args.engagement_column].astype(float) >= args.min_engagement

    positives = interactions[pos_mask].copy()
    if positives.empty:
        raise RuntimeError("No positive samples after applying thresholds.")

    if args.max_pos_samples and len(positives) > args.max_pos_samples:
        positives = positives.sample(args.max_pos_samples, random_state=args.seed)

    rng = np.random.default_rng(args.seed)
    copy_pool = interactions["copy_id"].unique()
    records = []

    def weight_for(row: pd.Series) -> Optional[float]:
        if args.weight_column and args.weight_column in row:
            return float(row[args.weight_column])
        return None

    for _, row in positives.iterrows():
        w = weight_for(row)
        user_id = int(row["user_id"])
        theme_id = int(row["theme_id"])
        copy_id = int(row["copy_id"])
        pos = {"user_id": user_id, "theme_id": theme_id, "copy_id": copy_id, "label": 1.0}
        if w is not None:
            pos["sample_weight"] = w
        records.append(pos)

        forbid = {copy_id}
        for _ in range(args.num_neg_per_pos):
            neg_copy = sample_neg_copy(rng, copy_pool, forbid)
            forbid.add(neg_copy)
            neg = {"user_id": user_id, "theme_id": theme_id, "copy_id": neg_copy, "label": 0.0}
            if w is not None:
                neg["sample_weight"] = w
            records.append(neg)

    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    print(f"[proxy] positives={len(positives)} total_rows={len(df)} users={num_users} copies={num_copies}")
    return df
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--proxy_interactions_csv", type=Path, default=None,
                    help="Proxy interactions CSV generated by build_proxy.py")
    ap.add_argument("--user_map_csv", type=Path, default=Path("twotower_data/user_map.csv"))
    ap.add_argument("--item_map_csv", type=Path, default=Path("twotower_data/item_map.csv"))
    ap.add_argument("--item_catalog_csv", type=Path, default=Path("twotower_data/item_catalog.csv"))
    ap.add_argument("--theme_col", default="theme_raw")
    ap.add_argument(
        "--exclude_theme_id",
        type=int,
        default=None,
        help=(
            "Drop rows where computed theme_id equals this value. "
            "Useful to exclude an 'NA/unknown' theme that can dominate contexts (often theme_id=0)."
        ),
    )
    ap.add_argument("--copy_catalog_csv", type=Path, default=None,
                    help="Optional catalog containing the exact copy_id space")
    ap.add_argument("--copy_catalog_id_col", default="copy_id")
    ap.add_argument("--copy_catalog_item_col", default="copy_id",
                    help="Column from copy catalog that matches proxy item_id")
    ap.add_argument("--copy_catalog_theme_col", default=None,
                    help="Theme column inside copy catalog; overrides --theme_col if provided")

    ap.add_argument(
        "--item_to_copy",
        choices=["nearest_text", "direct"],
        default="nearest_text",
        help=(
            "How to map proxy item_id/item_idx to copy_id. "
            "'nearest_text' maps each item_idx to the nearest copy embedding (requires --item_embeddings_csv and --copy_emb_csv). "
            "'direct' uses copy_catalog_csv mapping via --copy_catalog_item_col."
        ),
    )
    ap.add_argument(
        "--item_embeddings_csv",
        type=Path,
        default=None,
        help="Item embedding CSV with columns: item_idx, emb_* (used by --item_to_copy=nearest_text)",
    )
    ap.add_argument(
        "--item_to_copy_map_csv",
        type=Path,
        default=None,
        help="Optional precomputed mapping CSV with columns: item_idx, copy_id, (sim, sim_gap)",
    )
    ap.add_argument(
        "--min_sim",
        type=float,
        default=0.0,
        help="If mapping CSV includes sim, drop rows with sim < min_sim",
    )
    ap.add_argument(
        "--min_gap",
        type=float,
        default=0.0,
        help="If mapping CSV includes sim_gap, drop rows with sim_gap < min_gap",
    )
    ap.add_argument("--label_column", default="label")
    ap.add_argument("--label_positive_value", type=float, default=1.0)
    ap.add_argument("--engagement_column", default="engagement_total")
    ap.add_argument("--min_engagement", type=float, default=1.0)
    ap.add_argument("--max_pos_samples", type=int, default=20000,
                    help="Limit positives for efficiency (<=0 keeps all)")
    ap.add_argument("--num_neg_per_pos", type=int, default=3)
    ap.add_argument("--weight_column", default=None,
                    help="Optional column name (e.g., engagement_total) for sample_weight")

    ap.add_argument("--user_emb_csv", type=Path, default=Path("din_user_embeddings.csv"))
    ap.add_argument("--theme_emb_csv", type=Path, default=Path("din_theme_embeddings.csv"))
    ap.add_argument("--copy_emb_csv", type=Path, default=Path("copy_embeddings.csv"))
    ap.add_argument("--num_pos", type=int, default=1000)
    ap.add_argument("--num_neg_per_pos_toy", type=int, default=3)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", type=Path, default=Path("copy_train_samples.csv"))
    return ap.parse_args()


def main():
    args = parse_args()

    if args.proxy_interactions_csv and args.proxy_interactions_csv.exists():
        if args.max_pos_samples <= 0:
            args.max_pos_samples = None
        samples = build_samples_from_proxy(args)
    else:
        print("[warn] proxy_interactions_csv missing -> falling back to toy dataset.")
        user_ids = load_embedding_index(args.user_emb_csv, "user_id")
        theme_ids = load_embedding_index(args.theme_emb_csv, "theme_id")
        copy_ids = load_embedding_index(args.copy_emb_csv, "copy_id")

        samples = build_toy_samples(
            num_users=int(user_ids.max()) + 1,
            num_themes=int(theme_ids.max()) + 1,
            num_copies=int(copy_ids.max()) + 1,
            num_pos=args.num_pos,
            num_neg_per_pos=args.num_neg_per_pos_toy,
            seed=args.seed,
        )

    samples.to_csv(args.out_csv, index=False)
    print(f"[write] {args.out_csv} (rows={len(samples)})")


if __name__ == "__main__":
    main()
