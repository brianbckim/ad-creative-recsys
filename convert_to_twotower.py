import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

def read_interactions(path: Path, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    need_cols = {"user_id", "item_id", "timestamp"}
    if not need_cols.issubset(df.columns):
        missing = need_cols - set(df.columns)
        raise ValueError(f"Missing columns in interactions: {missing}")
    if label_col not in df.columns:
        cands = sorted([c for c in df.columns if c.startswith("label")])
        raise ValueError(f"label_col '{label_col}' not found. Available: {cands}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df["label"] = (df[label_col].astype(int) > 0).astype(int)
    return df[["user_id","item_id","timestamp","label"]]

def read_items(path: Path) -> pd.DataFrame:
    it = pd.read_csv(path, low_memory=False)
    if "item_id" not in it.columns:
        raise ValueError("items CSV must contain 'item_id'.")
    keep = [c for c in [
        "item_id","text","domain","lang","sentiment_num","proxy_ctr_composite",
        "proxy_volume","proxy_eng_rate","proxy_sent_pos","proxy_recency",
        "proxy_volume_n","proxy_eng_rate_n","proxy_sent_pos_n","proxy_recency_n",
        "engagement_total","engagement_actions_total","exposure_total",
        "theme_raw"
    ] if c in it.columns]
    return it[keep].copy()

def build_index(series: pd.Series) -> pd.Series:
    uniq = pd.Index(series.unique())
    return pd.Series(range(len(uniq)), index=uniq)

def map_ids(inter: pd.DataFrame, items: pd.DataFrame):
    item_map = build_index(items["item_id"])
    items = items.copy()
    items["item_idx"] = items["item_id"].map(item_map).astype(int)
    inter = inter.merge(items[["item_id","item_idx"]], on="item_id", how="inner")
    user_map = build_index(inter["user_id"])
    inter["user_idx"] = inter["user_id"].map(user_map).astype(int)
    inter = inter[["user_idx","item_idx","timestamp","label","user_id","item_id"]]
    return inter, items, user_map, item_map

def time_split(inter: pd.DataFrame, holdout_frac=0.2, per_user=False, min_events_per_user=2, seed=42):
    rng = np.random.default_rng(seed)
    inter = inter.sort_values("timestamp").reset_index(drop=True)
    if per_user:
        train_idx, val_idx = [], []
        for _, g in inter.groupby("user_idx", sort=False):
            if len(g) < max(2, min_events_per_user):
                train_idx.append(g.index.to_numpy()); continue
            cut = int(np.floor((1.0 - holdout_frac) * len(g)))
            cut = max(1, min(cut, len(g)-1))
            train_idx.append(g.index[:cut].to_numpy())
            val_idx.append(g.index[cut:].to_numpy())
        train_idx = np.concatenate(train_idx) if len(train_idx) else np.array([], dtype=int)
        val_idx = np.concatenate(val_idx) if len(val_idx) else np.array([], dtype=int)
        train = inter.loc[train_idx].copy()
        val   = inter.loc[val_idx].copy()
    else:
        cutoff = inter["timestamp"].quantile(1.0 - holdout_frac)
        train = inter[inter["timestamp"] < cutoff].copy()
        val   = inter[inter["timestamp"] >= cutoff].copy()
        vc = train["user_idx"].value_counts()
        ok = set(vc[vc >= min_events_per_user].index.tolist())
        train = train[train["user_idx"].isin(ok)]
        val   = val[val["user_idx"].isin(ok)]
    return train.reset_index(drop=True), val.reset_index(drop=True)

def negative_sampling(train_df: pd.DataFrame, num_items: int, neg_per_pos=4, seed=42) -> pd.DataFrame:
    if neg_per_pos <= 0:
        return train_df[["user_idx","item_idx","label"]].copy()
    rng = np.random.default_rng(seed)
    pos = train_df[train_df["label"] == 1]
    user_pos = pos.groupby("user_idx")["item_idx"].apply(set).to_dict()
    rows = []
    for uid, g in train_df.groupby("user_idx", sort=False):
        pset = set(g.loc[g["label"] == 1, "item_idx"].tolist())
        if not pset:
            continue
        want = len(pset) * neg_per_pos
        if want == 0:
            continue
        picked, seen, tries = set(), user_pos.get(uid, set()), 0
        while len(picked) < want and tries < want * 50:
            cand = int(rng.integers(0, num_items)); tries += 1
            if cand in seen or cand in picked: continue
            picked.add(cand)
        if len(picked) < want:
            print(f"[warn] user {uid}: negatives {len(picked)}/{want}")
        if picked:
            rows.append(pd.DataFrame({"user_idx": uid, "item_idx": list(picked), "label": 0}))
    if rows:
        neg_df = pd.concat(rows, ignore_index=True)
        base = train_df[["user_idx","item_idx","label"]]
        out = pd.concat([base, neg_df], ignore_index=True)
        out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return out
    return train_df[["user_idx","item_idx","label"]].copy()

def save_outputs(out_dir: Path, train_pairs: pd.DataFrame, val_pairs: pd.DataFrame,
                 user_map: pd.Series, item_map: pd.Series, item_catalog: pd.DataFrame):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_pairs.to_csv(out_dir / "twotower_train_pairs.csv", index=False)
    val_pairs.to_csv(out_dir / "twotower_val_pairs.csv", index=False)
    um = user_map.reset_index(); um.columns = ["user_id","user_idx"]
    im = item_map.reset_index(); im.columns = ["item_id","item_idx"]
    um.to_csv(out_dir / "user_map.csv", index=False)
    im.to_csv(out_dir / "item_map.csv", index=False)
    cols = ["item_id","item_idx"] + [c for c in item_catalog.columns if c not in ("item_id","item_idx")]
    item_catalog[cols].to_csv(out_dir / "item_catalog.csv", index=False)

def _load_json(p: Path):
    if not p: return None
    p = Path(p)
    if not p.exists(): return None
    try:
        with open(p, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return None

def _flag_present(argv: list, names: list) -> bool:
    return any(n in argv for n in names)

def _as_float(x):
    try:
        if x is None: return None
        v = float(x)
        return None if np.isnan(v) else v
    except Exception:
        return None

def get_first(quality: dict, *keys):
    for k in keys:
        if k in quality: return quality[k]
    return None

def suggest_from_quality(quality: dict, inter_df: pd.DataFrame, n_users: int, n_items: int):
    sug = {"per_user_split": False, "min_events_per_user": 2, "neg_per_pos": 4, "time_holdout": 0.2,
           "eval_all_items": False, "eval_item_batch": 65536}
    notes = []
    if quality:
        hist_zero = _as_float(quality.get("hist_len_zero_ratio"))
        uq_users  = int(_as_float(quality.get("unique_users")) or n_users)
        uq_items  = int(_as_float(quality.get("unique_items")) or n_items)
        if (uq_users >= 200) and (hist_zero is not None) and (hist_zero < 0.60):
            sug["per_user_split"] = True
            notes.append(f"per_user_split=True (users={uq_users}, hist0={hist_zero:.2f})")
        else:
            notes.append(f"per_user_split=False (users={uq_users}, hist0={hist_zero if hist_zero is not None else 'NA'})")
        p25 = _as_float(get_first(quality,"hist_p25","hist_25%","p25","25%"))
        p50 = _as_float(get_first(quality,"hist_p50","hist_50%","p50","50%"))
        if p25 is not None and p25 >= 3: sug["min_events_per_user"] = 3
        if p50 is not None and p50 >= 5: sug["min_events_per_user"] = 5
        if p25 is not None or p50 is not None:
            notes.append(f"min_events_per_user={sug['min_events_per_user']} (p25={p25}, p50={p50})")
        if uq_items <= 50000:
            sug["eval_all_items"] = True
            sug["eval_item_batch"] = 32768
            notes.append(f"eval_all_items=True (items={uq_items})")
        else:
            notes.append(f"eval_all_items=False (items={uq_items})")
    pos_rate = float((inter_df["label"]==1).mean()) if len(inter_df) else 0.10
    if   pos_rate < 0.05: sug["neg_per_pos"]=5; notes.append(f"neg_per_pos=5 (pos={pos_rate:.3f})")
    elif pos_rate < 0.15: sug["neg_per_pos"]=4; notes.append(f"neg_per_pos=4 (pos={pos_rate:.3f})")
    elif pos_rate < 0.30: sug["neg_per_pos"]=3; notes.append(f"neg_per_pos=3 (pos={pos_rate:.3f})")
    else:                 sug["neg_per_pos"]=1; notes.append(f"neg_per_pos=1 (pos={pos_rate:.3f})")
    n_events = int(len(inter_df))
    if   n_events <   50000: sug["time_holdout"]=0.10; notes.append(f"time_holdout=0.10 (n={n_events})")
    elif n_events > 1000000: sug["time_holdout"]=0.20; notes.append(f"time_holdout=0.20 (n={n_events})")
    else:                    sug["time_holdout"]=0.15; notes.append(f"time_holdout=0.15 (n={n_events})")
    return sug, notes

def maybe_pick_label_col(default_col: str, auto_tune_report: dict) -> str:
    if not auto_tune_report: return default_col
    meta = auto_tune_report.get("final_config", {})
    c_q = meta.get("composite_label_quantile", None)
    if isinstance(c_q, (int,float)) and c_q is not None:
        return "label_ctr_proxy" if default_col != "label_ctr_proxy" else default_col
    return default_col

def apply_auto_config(ns, inter_path: Path, items_path: Path, out_dir: Path):
    quality = _load_json(ns.quality_json)
    auto_tune = _load_json(ns.auto_tune_json)
    label_col = ns.label_col
    if ns.auto_config in ("soft","hard"):
        hinted = maybe_pick_label_col(ns.label_col, auto_tune)
        if (ns.auto_config == "hard") or (ns.auto_config == "soft" and not getattr(ns, "_raw_has_label_col", False)):
            label_col = hinted
    inter_df = read_interactions(inter_path, label_col=label_col)
    items_df = read_items(items_path)
    sug, notes = suggest_from_quality(quality, inter_df, inter_df["user_id"].nunique(), items_df["item_id"].nunique())
    eff = dict(
        interactions=str(inter_path), items=str(items_path), label_col=label_col,
        time_holdout=ns.time_holdout, per_user_split=ns.per_user_split,
        min_events_per_user=ns.min_events_per_user, neg_per_pos=ns.neg_per_pos,
        seed=ns.seed, out_dir=str(out_dir), half_life_days=ns.half_life_days
    )
    def maybe_set(k,v):
        if ns.auto_config=="hard" or (ns.auto_config=="soft" and not getattr(ns, f"_raw_has_{k}", False)):
            eff[k]=v
    maybe_set("time_holdout", sug["time_holdout"])
    maybe_set("per_user_split", sug["per_user_split"])
    maybe_set("min_events_per_user", sug["min_events_per_user"])
    maybe_set("neg_per_pos", sug["neg_per_pos"])
    auto_cfg = {
        "mode": ns.auto_config,
        "inputs": {
            "interactions": str(inter_path),
            "items": str(items_path),
            "quality_json": str(ns.quality_json) if ns.quality_json else None,
            "auto_tune_json": str(ns.auto_tune_json) if ns.auto_tune_json else None
        },
        "suggestions": sug,
        "eval_suggestions": {"eval_all_items": sug["eval_all_items"], "eval_item_batch": sug["eval_item_batch"]},
        "notes": notes,
        "final_effective_args": eff
    }
    pos_rate = float((inter_df["label"] == 1).mean()) if len(inter_df) else float("nan")
    n_events = int(len(inter_df))
    auto_cfg["suggestions_meta"] = {"pos_rate": pos_rate, "n_events": n_events}
    train_suggest = {
        "batch_size": 2048, "epochs": 10, "eval_all_items": sug["eval_all_items"],
        "eval_item_batch": sug["eval_item_batch"], "early_metric": "auc",
        "notes": ["Auto-derived suggestions from convert stage.", *notes]
    }
    n_users = inter_df["user_id"].nunique()
    n_items = items_df["item_id"].nunique()
    if n_items > 200_000:
        train_suggest["eval_all_items"] = False
        train_suggest["notes"].append(f"Large catalog (n_items={n_items}) -> sampled eval.")
    if n_users < 200:
        train_suggest["notes"].append(f"Few users (n_users={n_users}) -> prefer global time split.")
    return eff, auto_cfg, train_suggest, inter_df, items_df

def _recompute_split_recency(item_catalog: pd.DataFrame, inter_df: pd.DataFrame, tmax: pd.Timestamp, half_life_days: float):
    m = inter_df.groupby("item_id")["timestamp"].max().rename("last_ts")
    cat = item_catalog.merge(m, on="item_id", how="left")
    last_ts = pd.to_datetime(cat["last_ts"], errors="coerce")
    age_days = (tmax - last_ts).dt.total_seconds() / 86400.0
    med = np.nanmedian(age_days) if np.isfinite(np.nanmedian(age_days)) else 30.0
    age_days = pd.Series(age_days).fillna(med).clip(lower=0)
    rec = np.exp(-age_days / float(half_life_days))
    r = pd.Series(rec.values, index=cat.index)
    rn = (r - r.min()) / (r.max() - r.min() + 1e-12)
    return r.astype(float).values, rn.astype(float).values

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", type=Path, required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--label_col", type=str, default="label", choices=["label","label_ctr_proxy"])
    ap.add_argument("--time_holdout", type=float, default=0.2)
    ap.add_argument("--per_user_split", action="store_true")
    ap.add_argument("--min_events_per_user", type=int, default=2)
    ap.add_argument("--neg_per_pos", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quality_json", type=Path, default=None)
    ap.add_argument("--auto_tune_json", type=Path, default=None)
    ap.add_argument("--auto_config", type=str, choices=["off","soft","hard"], default="off")
    ap.add_argument("--out_dir", type=Path, default=Path("./twotower_data"))
    ap.add_argument("--half_life_days", type=float, default=7.0)
    ns, _ = ap.parse_known_args()
    argv = sys.argv[1:]
    flags = {
        "label_col": ["--label_col"],
        "time_holdout": ["--time_holdout"],
        "per_user_split": ["--per_user_split"],
        "min_events_per_user": ["--min_events_per_user"],
        "neg_per_pos": ["--neg_per_pos"]
    }
    for k, fns in flags.items(): setattr(ns, f"_raw_has_{k}", _flag_present(argv, fns))
    return ns

def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.auto_config in ("soft","hard"):
        eff, autoconfig, train_suggest, inter_for_sug, items_for_sug = apply_auto_config(args, args.interactions, args.items, out_dir)
        with open(out_dir / "twotower_autoconfig.json","w",encoding="utf-8") as f: json.dump(autoconfig,f,ensure_ascii=False,indent=2)
        with open(out_dir / "train_suggest.json","w",encoding="utf-8") as f: json.dump(train_suggest,f,ensure_ascii=False,indent=2)
        interactions_path = Path(eff["interactions"]); items_path = Path(eff["items"])
        label_col = eff["label_col"]; time_holdout = float(eff["time_holdout"])
        per_user_split = bool(eff["per_user_split"]); min_events_per_user = int(eff["min_events_per_user"])
        neg_per_pos = int(eff["neg_per_pos"]); half_life_days = float(eff.get("half_life_days", args.half_life_days))
        inter = inter_for_sug; items = items_for_sug
        if "proxy_sent" not in items.columns and "proxy_sent_pos" in items.columns:
            items["proxy_sent"] = items["proxy_sent_pos"]
        if "proxy_sent_n" not in items.columns and "proxy_sent_pos_n" in items.columns:
            items["proxy_sent_n"] = items["proxy_sent_pos_n"]
        if ("proxy_sent" not in items.columns) and ("proxy_sent_pos" not in items.columns):
            print("[info] no sentiment proxy found (proxy_sent / proxy_sent_pos)")
    else:
        interactions_path = Path(args.interactions); items_path = Path(args.items)
        label_col = args.label_col; time_holdout = args.time_holdout
        per_user_split = args.per_user_split; min_events_per_user = args.min_events_per_user
        neg_per_pos = args.neg_per_pos; half_life_days = args.half_life_days
        inter = read_interactions(interactions_path, label_col=label_col)
        items = read_items(items_path)
        if "proxy_sent" not in items.columns and "proxy_sent_pos" in items.columns:
            items["proxy_sent"] = items["proxy_sent_pos"]
        if "proxy_sent_n" not in items.columns and "proxy_sent_pos_n" in items.columns:
            items["proxy_sent_n"] = items["proxy_sent_pos_n"]
        if ("proxy_sent" not in items.columns) and ("proxy_sent_pos" not in items.columns):
            print("[info] no sentiment proxy found (proxy_sent / proxy_sent_pos)")

    inter, items, user_map, item_map = map_ids(inter, items)
    train_df, val_df = time_split(inter, holdout_frac=time_holdout,
                                  per_user=per_user_split,
                                  min_events_per_user=min_events_per_user,
                                  seed=args.seed)

    if len(val_df) == 0 and min_events_per_user > 1:
        print("[info] val empty -> relax min_events_per_user by 1 and retry")
        train_df, val_df = time_split(inter, holdout_frac=time_holdout,
                                      per_user=per_user_split,
                                      min_events_per_user=min_events_per_user - 1,
                                      seed=args.seed)

    if len(val_df)==0:
        print("[warn] validation split is empty.")
    if len(train_df)==0:
        print("[warn] train split is empty.")

    train_pairs = train_df[["user_idx","item_idx","label"]].copy()
    val_pairs   = val_df[["user_idx","item_idx","label"]].copy()
    train_pairs = negative_sampling(train_pairs, num_items=len(item_map), neg_per_pos=neg_per_pos, seed=args.seed)

    try:
        tmax_tr = train_df["timestamp"].max() if len(train_df) else inter["timestamp"].max()
        tmax_va = val_df["timestamp"].max()   if len(val_df)   else inter["timestamp"].max()
        r_tr, rn_tr = _recompute_split_recency(items, train_df, tmax_tr, half_life_days)
        r_va, rn_va = _recompute_split_recency(items, val_df, tmax_va, half_life_days)
        items["proxy_recency_train"]   = r_tr
        items["proxy_recency_train_n"] = rn_tr
        items["proxy_recency_val"]     = r_va
        items["proxy_recency_val_n"]   = rn_va
    except Exception as e:
        print(f"[warn] recency split-safe columns not computed: {e}")

    save_outputs(out_dir, train_pairs, val_pairs, user_map, item_map, items)

    up = len(user_map); ip = len(item_map)
    print(f"Saved Two-Tower datasets to: {out_dir.resolve()}")
    print(f" users={up} items={ip} train_pairs={len(train_pairs)} val_pairs={len(val_pairs)} pos_rate_train={train_pairs['label'].mean():.4f}")
    print(" - twotower_train_pairs.csv")
    print(" - twotower_val_pairs.csv")
    print(" - user_map.csv, item_map.csv")
    print(" - item_catalog.csv")
    if args.auto_config in ("soft","hard"):
        print(" - twotower_autoconfig.json")
        print(" - train_suggest.json")

if __name__ == "__main__":
    main()