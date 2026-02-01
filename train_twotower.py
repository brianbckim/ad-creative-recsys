import argparse
from pathlib import Path, PosixPath
from torch.serialization import add_safe_globals
import json
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple, Optional

TRAINER_VERSION = "v2.1-no-userfeats"

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SK_OK = True
except Exception:
    SK_OK = False


class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        need = {"user_idx","item_idx","label"}
        if not need.issubset(df.columns):
            miss = need - set(df.columns)
            raise ValueError(f"Dataset missing columns: {miss}")
        self.u = torch.as_tensor(df["user_idx"].to_numpy(), dtype=torch.long)
        self.i = torch.as_tensor(df["item_idx"].to_numpy(), dtype=torch.long)
        self.y = torch.as_tensor(df["label"].to_numpy(), dtype=torch.float32)
    def __len__(self): return self.u.shape[0]
    def __getitem__(self, idx): return self.u[idx], self.i[idx], self.y[idx]


def read_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"user_idx","item_idx","label"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} must contain {need}")
    return df[["user_idx","item_idx","label"]].copy()


def read_map(path: Path, key: str, idx: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {key, idx}.issubset(df.columns):
        raise ValueError(f"{path} must contain '{key}','{idx}'")
    return df[[key, idx]].copy()


def load_numeric_item_features(item_catalog_path: Optional[Path], n_items: int, numeric_cols=None):
    if item_catalog_path is None:
        return None, 0, [], None, None
    items = pd.read_csv(item_catalog_path, low_memory=False)
    if "item_idx" not in items.columns:
        raise ValueError("item_catalog.csv must include 'item_idx' column.")
    if numeric_cols is None:
        candidates = [
            "proxy_ctr_composite",
            "proxy_volume","proxy_eng_rate","proxy_sent","proxy_recency",
            "proxy_volume_n","proxy_eng_rate_n","proxy_sent_n","proxy_recency_n",
            "proxy_recency_train","proxy_recency_train_n","proxy_recency_val","proxy_recency_val_n",
            "sentiment_num",
            "engagement_total","engagement_actions_total","exposure_total"
        ]
        numeric_cols = [c for c in candidates if c in items.columns]
    if not numeric_cols:
        return None, 0, [], None, None

    feats = items[["item_idx"] + numeric_cols].copy()
    means, stds = {}, {}
    for c in numeric_cols:
        col = pd.to_numeric(feats[c], errors="coerce")
        m = float(col.mean()) if np.isfinite(col.mean()) else 0.0
        s = float(col.std(ddof=0)) if np.isfinite(col.std(ddof=0)) and col.std(ddof=0) != 0 else 1.0
        feats[c] = (col.fillna(m) - m) / s
        means[c] = m; stds[c] = s

    d_in = len(numeric_cols)
    mat = np.zeros((n_items, d_in), dtype=np.float32)
    for _, r in feats.iterrows():
        idx = int(r["item_idx"])
        if 0 <= idx < n_items:
            mat[idx, :] = r[numeric_cols].to_numpy(dtype=np.float32)
    feat_tensor = torch.from_numpy(mat)
    return feat_tensor, d_in, numeric_cols, means, stds


def augment_negatives_if_needed(train_df: pd.DataFrame, n_items: int, neg_per_pos: int, seed: int = 42) -> pd.DataFrame:
    pos = train_df[train_df["label"] == 1]
    neg = train_df[train_df["label"] == 0]
    if len(neg) >= len(pos) * max(1, neg_per_pos):
        return train_df

    rng = np.random.default_rng(seed)
    user_pos = pos.groupby("user_idx")["item_idx"].apply(set).to_dict()
    rows = []
    for uid, g in pos.groupby("user_idx", sort=False):
        want = max(0, neg_per_pos * len(g))
        picked = set()
        seen = user_pos.get(uid, set())
        tries = 0
        while len(picked) < want and tries < want * 50:
            tries += 1
            cand = int(rng.integers(0, n_items))
            if cand in seen or cand in picked:
                continue
            picked.add(cand)
        if picked:
            rows.append(pd.DataFrame({"user_idx": uid, "item_idx": list(picked), "label": 0}))
    if rows:
        neg_df = pd.concat(rows, ignore_index=True)
        out = pd.concat([train_df, neg_df], ignore_index=True)
        out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return out
    return train_df


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int,
                 item_feat_tensor: Optional[torch.Tensor] = None,
                 item_feat_in: int = 0,
                 feat_hidden_item: str = "128,64",
                 combine_item: str = "sum"):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.use_item_feats = item_feat_tensor is not None and item_feat_in > 0
        self.combine_item = combine_item
        if self.use_item_feats:
            self.register_buffer("item_feat_table", item_feat_tensor, persistent=False)
            h = [int(x) for x in feat_hidden_item.split(",") if x.strip()]
            layers = []
            in_dim = item_feat_in
            for hid in h:
                layers += [nn.Linear(in_dim, hid), nn.ReLU()]
                in_dim = hid
            out_dim = dim if combine_item == "sum" else max(1, dim // 2)
            layers += [nn.Linear(in_dim, out_dim)]
            self.item_feat_mlp = nn.Sequential(*layers)
            if combine_item == "concat":
                self.item_proj = nn.Linear(dim + out_dim, dim)

    def user_vector(self, user_idx: torch.Tensor):
        return self.user_emb(user_idx)

    def item_vector(self, item_idx: torch.Tensor):
        base = self.item_emb(item_idx)
        if not self.use_item_feats:
            return base
        feats = self.item_feat_table[item_idx]
        fvec = self.item_feat_mlp(feats)
        if self.combine_item == "sum":
            return base + fvec
        x = torch.cat([base, fvec], dim=-1)
        return self.item_proj(x)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor):
        u = self.user_vector(user_idx)
        v = self.item_vector(item_idx)
        logits = (u * v).sum(dim=-1)
        return logits


@torch.no_grad()
def eval_pointwise(model, dl, device):
    model.eval()
    ys, ps = [], []
    try:
        dl_len = len(dl)
    except TypeError:
        dl_len = None

    for b_idx, (u, i, y) in enumerate(dl):
        if dl_len is not None and b_idx % 100 == 0:
            print(f"[eval-pointwise] batch {b_idx}/{dl_len}")
        u, i = u.to(device), i.to(device)
        p = torch.sigmoid(model(u, i)).cpu().numpy()
        ys.append(y.numpy()); ps.append(p)

    y_true = np.concatenate(ys) if ys else np.array([])
    y_prob = np.concatenate(ps) if ps else np.array([])
    out = {}
    if SK_OK and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"]  = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"]  = float("nan")
    return out


@torch.no_grad()
def eval_ranking(model, df_val: pd.DataFrame, device, n_items: int,
                 ks=(10,20), sample_neg_k: Optional[int]=None, all_items: bool=False, item_batch: int=65536):
    model.eval()
    n_users_val = int(df_val["user_idx"].nunique())
    n_rows_val  = int(len(df_val))
    print(f"[eval-ranking] users={n_users_val}, rows={n_rows_val}, ks={ks}, sample_neg_k={sample_neg_k}, all_items={all_items}")

    if all_items:
        item_vecs = []
        for start in range(0, n_items, item_batch):
            batch_idx = torch.arange(start, min(start+item_batch, n_items), dtype=torch.long, device=device)
            v = model.item_vector(batch_idx)
            item_vecs.append(v)
        item_vecs = torch.cat(item_vecs, dim=0)
        item_vecs_t = item_vecs.t().contiguous()

    rng = np.random.default_rng(123)
    res = {f"recall@{k}": [] for k in ks}
    res.update({f"ndcg@{k}": [] for k in ks})

    for u_idx, (uid, g) in enumerate(df_val.groupby("user_idx", sort=False), start=1):
        if u_idx % 100 == 0 or u_idx == 1:
            print(f"[eval-ranking] user {u_idx}/{n_users_val}")

        pos_items = g.loc[g["label"] == 1, "item_idx"].unique().tolist()
        if not pos_items:
            continue

        if all_items:
            u = torch.full((1,), int(uid), dtype=torch.long, device=device)
            uvec = model.user_vector(u)
            scores = torch.matmul(uvec, item_vecs_t).flatten().cpu().numpy()
            labs = np.zeros((n_items,), dtype=int)
            labs[np.array(pos_items, dtype=int)] = 1
            order = np.argsort(-scores)
            labels = labs[order]
        else:
            cand = set(pos_items)
            max_possible = n_items
            base_target = len(pos_items) + int(sample_neg_k or 1000)
            target = min(max_possible, base_target)
            while len(cand) < target:
                cand.add(int(rng.integers(0, n_items)))
                if len(cand) >= max_possible:
                    break
            cands = np.fromiter(cand, dtype=int)
            u = torch.full((len(cands),), int(uid), dtype=torch.long, device=device)
            i = torch.as_tensor(cands, dtype=torch.long, device=device)
            logits = model(u, i).cpu().numpy()
            order = np.argsort(-logits)
            labels = np.isin(cands, np.array(pos_items, dtype=int)).astype(int)[order]

        for k in ks:
            topk = labels[:k]
            hits = topk.sum()
            if hits == 0 and labels.sum() == 0:
                continue
            denom = max(1, labels.sum())
            res[f"recall@{k}"].append(hits / denom)
            gains = (2**topk - 1)
            discounts = 1 / np.log2(np.arange(2, 2+len(topk)))
            dcg = float(np.sum(gains * discounts))
            ideal = np.sort(labels)[::-1][:k]
            ideal_dcg = float(np.sum((2**ideal - 1) * (1 / np.log2(np.arange(2, 2+len(ideal))))))
            ndcg = (dcg / ideal_dcg) if ideal_dcg > 0 else 0.0
            res[f"ndcg@{k}"].append(ndcg)

    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in res.items()}



def _sanitize_args_for_meta(args_obj):
    def _coerce(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (list, tuple)):
            return [_coerce(x) for x in v]
        if isinstance(v, dict):
            return {k: _coerce(vv) for k, vv in v.items()}
        return v
    return _coerce(dict(vars(args_obj)))

def save_bundle(model, args, item_cols, item_means, item_stds, path: Path,
                n_users: int = None, n_items: int = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    m = model.module if isinstance(model, nn.DataParallel) else model
    bundle = {
        "model": m.state_dict(),
        "meta": {
            "trainer_version": TRAINER_VERSION,
            "dim": args.dim,
            "item_feat_cols": item_cols,
            "item_means": item_means,
            "item_stds": item_stds,
            "combine_item": args.combine_item,
            "n_users": n_users,
            "n_items": n_items,
            "args": _sanitize_args_for_meta(args),
        }
    }
    torch.save(bundle, path)
    print(f"[save] {path}")


def _validate_checkpoint_meta(ckpt: Dict[str,Any], args, n_users: int, n_items: int, item_cols):
    meta = ckpt.get("meta", {})
    ver = meta.get("trainer_version", "unknown")
    if ver != TRAINER_VERSION:
        print(f"[warn] checkpoint trainer_version={ver}; current={TRAINER_VERSION}. Proceeding.")

    if "dim" in meta and meta["dim"] != args.dim:
        raise ValueError(f"Checkpoint dim={meta['dim']} but args.dim={args.dim}")
    if "combine_item" in meta and meta["combine_item"] != getattr(args, "combine_item"):
        raise ValueError(f"Checkpoint combine_item={meta['combine_item']} but args.combine_item={args.combine_item}")
    if args.use_item_features:
        ck_cols = meta.get("item_feat_cols", [])
        if item_cols and ck_cols and list(item_cols) != list(ck_cols):
            raise ValueError(f"Item feature columns mismatch: ckpt={ck_cols} vs current={item_cols}")
    ck_nu = meta.get("n_users")
    ck_ni = meta.get("n_items")
    if ck_nu is not None and ck_nu != n_users:
        raise ValueError(f"User count mismatch: ckpt={ck_nu} vs maps={n_users}")
    if ck_ni is not None and ck_ni != n_items:
        raise ValueError(f"Item count mismatch: ckpt={ck_ni} vs maps={n_items}")
    return meta


@torch.no_grad()
def dump_topk_candidates(model, args, n_users: int, n_items: int, device,
                         item_id_by_idx: Optional[np.ndarray] = None,
                         user_id_by_idx: Optional[np.ndarray] = None):
    print(f"[dump] generating Top-{args.dump_candidates_k} candidates ...")
    out = Path(args.dump_candidates_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    m = model.module if isinstance(model, nn.DataParallel) else model
    m.eval()

    item_vecs = []
    for start in range(0, n_items, args.eval_item_batch):
        idx = torch.arange(start, min(start+args.eval_item_batch, n_items), dtype=torch.long, device=device)
        v = m.item_vector(idx)
        item_vecs.append(v)
    item_vecs = torch.cat(item_vecs, dim=0)
    item_vecs_t = item_vecs.t().contiguous()

    if args.dump_users_scope == "val":
        df = pd.read_csv(args.val_pairs)
        user_ids = sorted(df["user_idx"].unique().tolist())
    elif args.dump_users_scope == "train":
        df = pd.read_csv(args.train_pairs)
        user_ids = sorted(df["user_idx"].unique().tolist())
    else:
        user_ids = list(range(n_users))

    k_req = int(args.dump_candidates_k)
    k = int(min(max(1, k_req), n_items))
    if k_req > n_items:
        print(f"[warn] dump_candidates_k={k_req} > n_items={n_items} -> clamped to {k}")

    use_sigmoid = (args.dump_score_type == "sigmoid")

    with open(out, "w", encoding="utf-8") as f:
        for uid in user_ids:
            u = torch.full((1,), int(uid), dtype=torch.long, device=device)
            uvec = m.user_vector(u)
            scores = torch.matmul(uvec, item_vecs_t).flatten()
            if use_sigmoid:
                scores = torch.sigmoid(scores)
            topk = torch.topk(scores, k=k, largest=True, sorted=True)

            cand_idx = topk.indices.detach().cpu().numpy().tolist()
            cand_scores = topk.values.detach().cpu().numpy().tolist()

            rec = {
                "user_idx": int(uid),
                "user_id": (None if user_id_by_idx is None else user_id_by_idx[int(uid)]),
                "candidate_item_idx": cand_idx,
                "candidate_item_id": (None if item_id_by_idx is None else [item_id_by_idx[i] for i in cand_idx]),
                "candidate_scores": cand_scores,
                "k": k,
                "score_type": args.dump_score_type
            }
            f.write(json.dumps(rec) + "\n")
    print(f"[dump] wrote: {out.resolve()}")

def _seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        import random as _random
        _random.seed(seed)
    except Exception:
        pass


def _worker_init(seed_base: int):
    def _fn(worker_id: int):
        s = seed_base + worker_id
        np.random.seed(s)
        torch.manual_seed(s)
    return _fn


def _export_embeddings(model, args, user_map_df: pd.DataFrame, item_map_df: pd.DataFrame, device):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = model.module if isinstance(model, nn.DataParallel) else model
    m.eval()

    with torch.no_grad():
        u_idx = torch.arange(0, int(user_map_df["user_idx"].max()) + 1, dtype=torch.long, device=device)
        i_idx = torch.arange(0, int(item_map_df["item_idx"].max()) + 1, dtype=torch.long, device=device)
        u_vec = m.user_vector(u_idx).cpu().numpy()
        i_vec = m.item_vector(i_idx).cpu().numpy()

    u = user_map_df.sort_values("user_idx").reset_index(drop=True)
    i = item_map_df.sort_values("item_idx").reset_index(drop=True)
    np.save(out_dir / "user_emb.npy", u_vec)
    np.save(out_dir / "item_emb.npy", i_vec)
    u.to_csv(out_dir / "user_map.csv", index=False)
    i.to_csv(out_dir / "item_map.csv", index=False)
    print(f"[export] user_emb.npy / item_emb.npy saved in {out_dir.resolve()}")


def check_config(args):
    banned = ["use_user_features","user_features","feat_hidden_user","combine_user"]
    for k in banned:
        if hasattr(args, k) and getattr(args, k):
            raise SystemExit(f"--{k} is not supported in this trainer version ({TRAINER_VERSION}).")
    if args.eval_all_items and args.eval_item_batch < 1024:
        raise SystemExit("--eval_item_batch too small for --eval_all_items.")


def train_loop(args):
    device = ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    device = torch.device(device)
    _seed_all(args.seed)

    tr = read_pairs(args.train_pairs)
    va = read_pairs(args.val_pairs)
    user_map = read_map(args.user_map, "user_id", "user_idx")
    item_map = read_map(args.item_map, "item_id", "item_idx")
    n_users = int(user_map["user_idx"].max()) + 1
    n_items = int(item_map["item_idx"].max()) + 1

    if args.ensure_negatives > 0:
        tr = augment_negatives_if_needed(tr, n_items, neg_per_pos=args.ensure_negatives, seed=args.seed)

    item_feat_tensor, item_feat_in, item_cols, item_means, item_stds = load_numeric_item_features(
        args.item_catalog, n_items, None if args.use_item_features else []
    )

    model = TwoTower(
        n_users=n_users, n_items=n_items, dim=args.dim,
        item_feat_tensor=(item_feat_tensor.to(device) if item_feat_tensor is not None else None),
        item_feat_in=item_feat_in,
        feat_hidden_item=args.feat_hidden_item,
        combine_item=args.combine_item
    ).to(device)

    ckpt_state = None
    if args.load_checkpoint:
        p = Path(args.load_checkpoint)
        if p.exists():
            try:
                add_safe_globals([PosixPath])
            except Exception:
                pass
            try:
                ckpt_state = torch.load(p, map_location=device, weights_only=True)
            except Exception:
                ckpt_state = torch.load(p, map_location=device, weights_only=False)

            if "model" in ckpt_state:
                model.load_state_dict(ckpt_state["model"])
            else:
                model.load_state_dict(ckpt_state)
            try:
                if isinstance(ckpt_state, dict):
                    _validate_checkpoint_meta(ckpt_state, args, n_users, n_items, item_cols)
            except Exception as e:
                raise RuntimeError(f"[meta-check] {e}")
        else:
            print(f"[warn] checkpoint not found: {p}. Continuing without warm-start.")

    if (args.mode == "dump" or args.epochs == 0) and args.dump_candidates_k > 0:
        item_id_by_idx = None
        user_id_by_idx = None
        try:
            item_id_by_idx = item_map.sort_values("item_idx")["item_id"].to_numpy()
            user_id_by_idx = user_map.sort_values("user_idx")["user_id"].to_numpy()
        except Exception:
            pass
        dump_topk_candidates(
            model, args, n_users, n_items, device,
            item_id_by_idx=item_id_by_idx,
            user_id_by_idx=user_id_by_idx
        )
        if args.export_embeddings:
            _export_embeddings(model, args, user_map, item_map, device)
        return

    if torch.cuda.device_count() > 1 and args.dataparallel:
        print(f"[info] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,args.epochs))
    elif args.scheduler == "steplr":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1,args.step_size), gamma=args.gamma)
    else:
        sched = None
    loss_fn = nn.BCEWithLogitsLoss()

    ds_tr = PairDataset(tr)
    ds_va = PairDataset(va)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False,
                       worker_init_fn=_worker_init(args.seed))
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False,
                       worker_init_fn=_worker_init(args.seed+1))

    best_metric = -math.inf
    patience_ctr = 0
    metrics_rows = []

    ks = tuple(int(k) for k in args.k.split(","))
    if args.eval_all_items and n_items > args.eval_all_items_max_guard:
        print(f"[guard] n_items={n_items} > {args.eval_all_items_max_guard} -> switch to sampled eval")
        args.eval_all_items = False

    if not args.eval_all_items:
        args.eval_sample_k = int(min(args.eval_sample_k, max(1, n_items - 1)))

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        print(f"\n[epoch {epoch}] training start...")

        for b_idx, (u, i, y) in enumerate(dl_tr):
            u, i, y = u.to(device), i.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(u, i)
            loss = loss_fn(logits, y)
            if args.emb_reg > 0.0:
                m = model.module if isinstance(model, nn.DataParallel) else model
                uv = m.user_vector(u)
                iv = m.item_vector(i)
                emb_pen = (uv.pow(2).sum(dim=1).mean() + iv.pow(2).sum(dim=1).mean()) * (0.5 * args.emb_reg)
                loss = loss + emb_pen
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            run_loss += loss.item() * y.size(0)

        if sched is not None:
            sched.step()
        tr_loss = run_loss / max(1, len(ds_tr))

        t0 = time.perf_counter()
        point = eval_pointwise(model, dl_va, device)
        t1 = time.perf_counter()
        print(f"[eval] pointwise done in {t1 - t0:.2f}s")

        print(f"[eval] ranking start (ks={ks}, sample_k={None if args.eval_all_items else args.eval_sample_k}, all_items={args.eval_all_items})")
        t2 = time.perf_counter()
        rank = eval_ranking(
            model, va, device,
            n_items=n_items,
            ks=ks,
            sample_neg_k=(None if args.eval_all_items else args.eval_sample_k),
            all_items=args.eval_all_items,
            item_batch=args.eval_item_batch
        )
        t3 = time.perf_counter()
        print(f"[eval] ranking done in {t3 - t2:.2f}s")

        score = point.get("roc_auc", float("nan")) if args.early_metric.lower() == "auc" \
                else rank.get(f"ndcg@{ks[0]}", float("nan"))
        improved = np.isfinite(score) and (score > best_metric)

        rank_str = "  ".join([f"{k}={v:.4f}" for k,v in rank.items()])
        print(f"[epoch {epoch:03d}] loss={tr_loss:.4f}  val_auc={point['roc_auc']:.4f}  val_pr={point['pr_auc']:.4f}  {rank_str}")

        row = dict(epoch=epoch, loss=tr_loss, **point, **rank)
        metrics_rows.append(row)

        if improved:
            best_metric = score
            patience_ctr = 0
            if args.save_best:
                save_bundle(model, args, item_cols, item_means, item_stds,
                            path=Path(args.out_dir) / "twotower_best.pt",
                            n_users=n_users, n_items=n_items)
        else:
            patience_ctr += 1
            if args.early_stop > 0 and patience_ctr >= args.early_stop:
                print(f"[early-stop] no improvement for {args.early_stop} epoch(s).")
                break

    if args.save_metrics:
        metrics_path = Path(args.out_dir) / (args.metrics_out if args.metrics_out else "metrics.jsonl")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            for r in metrics_rows:
                f.write(json.dumps(r) + "\n")
        print(f"[metrics] wrote: {metrics_path.resolve()}")

    if args.save_final:
        save_bundle(model, args, item_cols, item_means, item_stds,
                    path=Path(args.out_dir) / "twotower_final.pt",
                    n_users=n_users, n_items=n_items)

    if args.dump_candidates_k and args.dump_candidates_k > 0:
        item_id_by_idx = item_map.sort_values("item_idx")["item_id"].to_numpy()
        user_id_by_idx = user_map.sort_values("user_idx")["user_id"].to_numpy()
        dump_topk_candidates(model, args, n_users, n_items, device,
                             item_id_by_idx=item_id_by_idx, user_id_by_idx=user_id_by_idx)

    if args.export_embeddings:
        _export_embeddings(model, args, user_map, item_map, device)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pairs", type=Path, required=True)
    ap.add_argument("--val_pairs",   type=Path, required=True)
    ap.add_argument("--user_map",    type=Path, required=True)
    ap.add_argument("--item_map",    type=Path, required=True)
    ap.add_argument("--item_catalog",type=Path, default=None)

    ap.add_argument("--use_item_features", action="store_true")
    ap.add_argument("--feat_hidden_item", type=str, default="128,64")
    ap.add_argument("--combine_item", type=str, choices=["sum","concat"], default="sum")

    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ensure_negatives", type=int, default=0)
    ap.add_argument("--emb_reg", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=0.0)
    ap.add_argument("--scheduler", type=str, default="none", choices=["none","cosine","steplr"])
    ap.add_argument("--step_size", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.5)

    ap.add_argument("--k", type=str, default="10,20")
    ap.add_argument("--eval_all_items", action="store_true")
    ap.add_argument("--eval_all_items_max_guard", type=int, default=200000)
    ap.add_argument("--eval_sample_k", type=int, default=1000)
    ap.add_argument("--eval_item_batch", type=int, default=65536)
    ap.add_argument("--early_metric", type=str, default="auc", choices=["auc","ndcg"])
    ap.add_argument("--early_stop", type=int, default=0)

    ap.add_argument("--save_best", action="store_true")
    ap.add_argument("--save_final", action="store_true")
    ap.add_argument("--save_metrics", action="store_true")
    ap.add_argument("--metrics_out", type=str, default="metrics.jsonl")
    ap.add_argument("--out_dir", type=Path, default=Path("./twotower_runs"))
    ap.add_argument("--load_checkpoint", type=Path, default=None)
    ap.add_argument("--dataparallel", action="store_true")

    ap.add_argument("--dump_candidates_k", type=int, default=0)
    ap.add_argument("--dump_candidates_out", type=Path, default=Path("./tt_candidates_topK.jsonl"))
    ap.add_argument("--dump_users_scope", type=str, default="val", choices=["val","train","all"])
    ap.add_argument("--dump_score_type", type=str, default="dot", choices=["dot","sigmoid"])

    ap.add_argument("--mode", type=str, choices=["train","dump"], default="train")

    ap.add_argument("--export_embeddings", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()
    check_config(args)
    train_loop(args)


if __name__ == "__main__":
    main()