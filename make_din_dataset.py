import argparse, json, random, pickle
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple

def read_map_csv(path: str, id_col: str, idx_col: str):
    df = pd.read_csv(path, dtype={id_col: str})
    if not {id_col, idx_col}.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: {id_col},{idx_col}")
    m = {str(k): int(v) for k, v in zip(df[id_col], df[idx_col])}
    n = int(df[idx_col].max()) + 1
    return m, n

def build_cate_list(n_items: int, item_catalog: Optional[str], cate_col: Optional[str]) -> Tuple[list, int]:
    if item_catalog and Path(item_catalog).exists() and cate_col:
        ic = pd.read_csv(item_catalog, low_memory=False)
        if "item_idx" in ic.columns and cate_col in ic.columns:
            col = ic[cate_col]
            if not pd.api.types.is_integer_dtype(col):
                cats = {v: i for i, v in enumerate(pd.Series(col, dtype="string").fillna("NA").unique())}
                ic["_cate_id"] = ic[cate_col].astype("string").fillna("NA").map(cats).astype(int)
                use_col = "_cate_id"
            else:
                use_col = cate_col
            out = np.zeros((n_items,), dtype=np.int64)
            for _, r in ic.iterrows():
                idx = int(r["item_idx"])
                if 0 <= idx < n_items:
                    out[idx] = int(r[use_col])
            cate_count = int(out.max()) + 1
            return out.tolist(), cate_count
    return [0] * n_items, 1


def build_theme_list(n_items: int, item_catalog: Optional[str], theme_col: Optional[str]) -> Tuple[list, int]:
    if item_catalog and Path(item_catalog).exists() and theme_col:
        ic = pd.read_csv(item_catalog, low_memory=False)
        if "item_idx" in ic.columns and theme_col in ic.columns:
            col = ic[theme_col]
            if not pd.api.types.is_integer_dtype(col):
                themes = {v: i for i, v in enumerate(pd.Series(col, dtype="string").fillna("NA").unique())}
                ic["_theme_id"] = ic[theme_col].astype("string").fillna("NA").map(themes).astype(int)
                use_col = "_theme_id"
            else:
                use_col = theme_col

            out = np.zeros((n_items,), dtype=np.int64)
            for _, r in ic.iterrows():
                idx = int(r["item_idx"])
                if 0 <= idx < n_items:
                    out[idx] = int(r[use_col])
            theme_count = int(out.max()) + 1
            return out.tolist(), theme_count

    return [0] * n_items, 1

def load_proxy_din_jsonl(path: str, user_id2idx: Dict[str,int], item_id2idx: Dict[str,int]):
    rows = []
    miss_user = miss_item = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)

            if "user_idx" in r and r["user_idx"] not in (None, "", " "):
                try:
                    uid = int(r["user_idx"])
                except Exception:
                    uid = user_id2idx.get(str(r["user_idx"]))
            else:
                uid = user_id2idx.get(str(r.get("user_id")))
            if uid is None:
                miss_user += 1
                continue

            if "item_idx" in r and r["item_idx"] not in (None, "", " "):
                try:
                    iid = int(r["item_idx"])
                except Exception:
                    iid = item_id2idx.get(str(r["item_idx"]))
            else:
                iid = item_id2idx.get(str(r.get("item_id")))
            if iid is None:
                miss_item += 1
                continue

            lab = int(r.get("label", 0))
            ts  = r.get("timestamp")
            rows.append((uid, iid, lab, ts))
    print(f"[load-jsonl] kept={len(rows)}  skipped_user={miss_user}  skipped_item={miss_item}")
    return rows

def load_pairs_csv(path: str):
    df = pd.read_csv(path)
    need = {"user_idx","item_idx","label"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: {need}")
    ts = df["timestamp"] if "timestamp" in df.columns else None
    rows = list(zip(df["user_idx"].astype(int), df["item_idx"].astype(int), df["label"].astype(int), (ts if ts is not None else [None]*len(df))))
    print(f"[load-csv] {path} rows={len(rows)}")
    return rows

def split_by_user_chrono(rows, test_ratio=0.1, seed=42):
    random.seed(seed)
    by_user = {}
    for uid, iid, lab, ts in rows:
        by_user.setdefault(uid, []).append((uid, iid, lab, ts))
    train, test = [], []
    for uid, lst in by_user.items():
        lst = sorted(lst, key=lambda x: (x[3] or ""))
        n = len(lst)
        if n <= 1:
            train += lst
            continue
        k = max(1, int(round(n * test_ratio)))
        test += lst[-k:]
        train += lst[:-k] if n > k else lst
    return train, test

def build_histories(examples):
    by_user = {}
    out = []
    for uid, iid, lab, ts in examples:
        hist = [x[1] for x in by_user.get(uid, [])]
        out.append((uid, hist, iid, lab, ts))
        by_user.setdefault(uid, []).append((uid, iid, ts))
    return out

def sample_neg(n_items: int, forbid: set, rng: random.Random) -> int:
    if n_items <= len(forbid):
        for x in range(n_items):
            if x not in forbid:
                return x
        return 0
    while True:
        z = rng.randint(0, n_items - 1)
        if z not in forbid:
            return z

def to_legacy_train_samples(hist_examples):
    out = []
    for uid, hist, tgt, lab, _ in hist_examples:
        out.append((uid, hist, int(tgt), int(lab)))
    return out

def to_legacy_test_samples(hist_examples, n_items: int, seed: int = 42):
    rng = random.Random(seed)
    out = []
    for uid, hist, tgt, lab, _ in hist_examples:
        if int(lab) != 1:
            continue
        forbid = set(hist); forbid.add(int(tgt))
        neg = sample_neg(n_items, forbid, rng)
        out.append((uid, hist, [int(tgt), int(neg)]))
    return out

def to_std_din_samples(hist_examples, cate_list: list):
    out = []
    for uid, hist, tgt, lab, _ in hist_examples:
        h_cates = [cate_list[i] for i in hist] if hist else []
        out.append((uid, hist, h_cates, int(tgt), int(cate_list[int(tgt)]), int(lab)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", default=None, help="Single CSV with user_idx,item_idx,label (will internally split)")
    ap.add_argument("--proxy_jsonl", default=None, help="JSONL interactions (legacy path)")
    ap.add_argument("--pairs_csv_train", default=None, help="CSV: twotower_train_pairs.csv")
    ap.add_argument("--pairs_csv_val",   default=None, help="CSV: twotower_val_pairs.csv")

    ap.add_argument("--user_map",    required=True)
    ap.add_argument("--item_map",    required=True)
    ap.add_argument("--item_catalog", default=None)
    ap.add_argument("--cate_col",     default=None)
    ap.add_argument("--theme_col",    default=None,
                    help="Column name inside item_catalog that stores the theme identifier")

    ap.add_argument("--out",          default="./din/dataset.pkl")
    ap.add_argument("--test_ratio",   type=float, default=0.1)
    ap.add_argument("--format",       choices=["legacy","std"], default="legacy",
                    help="legacy=(train: (u,h,t,l), test: (u,h,[pos,neg])); std=DIN 6-tuple")
    args = ap.parse_args()

    user_id2idx, n_users = read_map_csv(args.user_map, "user_id", "user_idx")
    item_id2idx, n_items = read_map_csv(args.item_map, "item_id", "item_idx")
    cate_list, cate_count = build_cate_list(n_items, args.item_catalog, args.cate_col)
    theme_list, theme_count = build_theme_list(n_items, args.item_catalog, args.theme_col)

    train_rows = None
    val_rows = None

    if args.pairs_csv_train or args.pairs_csv_val:
        if not (args.pairs_csv_train and args.pairs_csv_val):
            raise SystemExit("Please provide both --pairs_csv_train and --pairs_csv_val")
        train_rows = load_pairs_csv(args.pairs_csv_train)
        val_rows   = load_pairs_csv(args.pairs_csv_val)
    elif args.pairs_csv:
        all_rows = load_pairs_csv(args.pairs_csv)
        train_rows, val_rows = split_by_user_chrono(all_rows, test_ratio=args.test_ratio)
    elif args.proxy_jsonl:
        all_rows = load_proxy_din_jsonl(args.proxy_jsonl, user_id2idx, item_id2idx)
        if not all_rows:
            raise RuntimeError("No usable rows after mapping. Check maps and jsonl.")
        train_rows, val_rows = split_by_user_chrono(all_rows, test_ratio=args.test_ratio)
    else:
        raise SystemExit("Please provide one of: --pairs_csv_train+--pairs_csv_val  OR  --pairs_csv  OR  --proxy_jsonl")

    tr_hist = build_histories(train_rows)
    te_hist = build_histories(val_rows)

    if args.format == "legacy":
        train_set = to_legacy_train_samples(tr_hist)
        test_set  = to_legacy_test_samples(te_hist, n_items=n_items)
    else:
        train_set = to_std_din_samples(tr_hist, cate_list)
        test_set  = to_std_din_samples(te_hist,  cate_list)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(train_set, f, protocol=4)
        pickle.dump(test_set,  f, protocol=4)
        pickle.dump(cate_list, f, protocol=4)
        pickle.dump(theme_list, f, protocol=4)
        pickle.dump((n_users, n_items, cate_count, theme_count), f, protocol=4)

    print(f"[write] {out_path.resolve()}")
    print(f"[stats] users={n_users}, items={n_items}, cate_count={cate_count}, "
          f"theme_count={theme_count}, train={len(train_set)}, test={len(test_set)}, format={args.format})")

if __name__ == "__main__":
    main()
