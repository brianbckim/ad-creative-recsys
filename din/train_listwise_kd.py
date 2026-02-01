import tensorflow as _tf
tf = _tf.compat.v1
tf.disable_v2_behavior()

import argparse
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd

from input import DataInput
from model import Model


def read_map_csv(path: Path, key_col: str, idx_col: str):
    df = pd.read_csv(path)
    if key_col not in df.columns or idx_col not in df.columns:
        raise ValueError(f"{path} must have columns '{key_col}','{idx_col}'")
    return dict(zip(df[key_col].astype(str), df[idx_col].astype(int)))


def _read_meta_after_cate(f, cate_list_len: int):
    next_obj = pickle.load(f)
    if isinstance(next_obj, tuple) and 3 <= len(next_obj) <= 4:
        theme_list = [0] * cate_list_len
        meta = next_obj
    else:
        theme_list = list(next_obj)
        meta = pickle.load(f)
    return theme_list, meta


def load_dataset_payload(dataset_pkl: Path):
    with open(dataset_pkl, "rb") as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        theme_list, meta = _read_meta_after_cate(f, len(cate_list))

    if len(meta) == 4:
        user_count, item_count, cate_count, theme_count = meta
    elif len(meta) == 3:
        user_count, item_count, cate_count = meta
        theme_count = max(1, len(set(theme_list))) if theme_list else 1
    else:
        raise ValueError(f"Unsupported meta tuple length: {len(meta)}")

    if not theme_list:
        theme_list = [0] * len(cate_list)

    return (
        train_set,
        test_set,
        cate_list,
        theme_list,
        int(user_count),
        int(item_count),
        int(cate_count),
        int(theme_count),
    )


def load_candidates(tt_jsonl: Path, item_count: int, user_count: int):
    cand = {}
    with open(tt_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "user_idx" not in r:
                continue
            uid = int(r["user_idx"])
            if not (0 <= uid < user_count):
                continue
            ids = [int(x) for x in r.get("candidate_item_idx", [])]
            sco = [float(x) for x in r.get("candidate_scores", [])]
            keep = [(i, s) for (i, s) in zip(ids, sco) if 0 <= i < item_count]
            if not keep:
                continue
            ids_f, sco_f = zip(*keep)
            cand[uid] = (list(ids_f), list(sco_f))
    return cand


def load_positives_from_proxy(proxy_jsonl: Path, user2idx: dict, item2idx: dict, user_n: int, item_n: int):
    pos = {}
    with open(proxy_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if int(r.get("label", 0)) != 1:
                continue
            uid = None
            if "user_idx" in r:
                try:
                    uid = int(r["user_idx"])
                except:
                    uid = None
            if uid is None:
                uid = user2idx.get(str(r.get("user_id", "")).strip())
            if uid is None or not (0 <= uid < user_n):
                continue
            iid = None
            if "item_idx" in r:
                try:
                    iid = int(r["item_idx"])
                except:
                    iid = None
            if iid is None:
                iid = item2idx.get(str(r.get("item_id", "")).strip())
            if iid is None or not (0 <= iid < item_n):
                continue
            pos.setdefault(uid, set()).add(iid)
    return pos


def load_positives_from_dataset(train_set, user_n: int, item_n: int):
    pos = {}
    for (u, hist, tgt, lab) in train_set:
        u = int(u)
        tgt = int(tgt)
        if int(lab) == 1 and 0 <= u < user_n and 0 <= tgt < item_n:
            pos.setdefault(u, set()).add(tgt)
    return pos


def make_batches(batch_users, batch_targets, cands, pos_map, fallback, K, tau):
    def _ensure_len(ids, scores):
        if not ids:
            return list(fallback[0]), list(fallback[1])
        if len(ids) >= K:
            return list(ids[:K]), list(scores[:K])
        pad = K - len(ids)
        ids = list(ids) + [ids[-1]] * pad
        scores = list(scores) + [scores[-1]] * pad
        return ids, scores

    def _maybe_inject_target(ids, scores, target):
        if target in ids:
            return ids, scores
        ids = list(ids)
        scores = list(scores)
        ids[-1] = int(target)
        scores[-1] = max(scores)
        return ids, scores

    B = len(batch_users)
    cand_ids = np.zeros((B, K), dtype=np.int32)
    teacher_s = np.zeros((B, K), dtype=np.float32)
    y_list = np.zeros((B, K), dtype=np.float32)
    label_mask = np.zeros((B,), dtype=np.float32)

    for b, (u, tgt) in enumerate(zip(batch_users, batch_targets)):
        ids_scores = cands.get(int(u))
        if ids_scores is None:
            ids, scores = fallback
        else:
            ids, scores = ids_scores
        ids, scores = _ensure_len(ids, scores)
        ids, scores = _maybe_inject_target(ids, scores, int(tgt))

        cand_ids[b, :] = np.asarray(ids[:K], dtype=np.int32)
        sc_k = np.asarray(scores[:K], dtype=np.float32)

        t = sc_k / float(tau)
        t = t - t.max()
        q = np.exp(t)
        teacher_s[b, :] = q / (q.sum() + 1e-9)

        pos_set = pos_map.get(int(u), set())
        y = np.array([1.0 if iid in pos_set else 0.0 for iid in cand_ids[b]], dtype=np.float32)
        if y.sum() > 0:
            y_list[b, :] = y / y.sum()
            label_mask[b] = 1.0

    return cand_ids, teacher_s, y_list, label_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_pkl", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--user_map", type=Path, required=True)
    ap.add_argument("--item_map", type=Path, required=True)
    ap.add_argument("--ckpt_in", type=Path, default=None)
    ap.add_argument("--proxy_jsonl", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=Path("./din_kd_runs"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--tau", type=float, default=1.5)
    ap.add_argument("--lambda_kd", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--point_loss_weight", type=float, default=1.0,
                    help="Weight for original pointwise BCE loss (>=0)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Global seed for reproducible shuffling/initialization")
    args = ap.parse_args()

    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    (
        train_set,
        test_set,
        cate_list,
        theme_list,
        user_count,
        item_count,
        cate_count,
        theme_count,
    ) = load_dataset_payload(args.dataset_pkl)

    user2idx = read_map_csv(args.user_map, "user_id", "user_idx")
    item2idx = read_map_csv(args.item_map, "item_id", "item_idx")

    cands = load_candidates(args.candidates, item_count=item_count, user_count=user_count)
    users = sorted(cands.keys())
    if not users:
        raise RuntimeError("No users in candidates after filtering. Check tt_candidates_topK.jsonl and maps.")
    print(f"[info] users_with_candidates={len(users)} item_count={item_count} cate_count={cate_count} theme_count={theme_count}")

    if args.proxy_jsonl and args.proxy_jsonl.exists():
        pos_map = load_positives_from_proxy(args.proxy_jsonl, user2idx, item2idx, user_count, item_count)
        print(f"[info] positives(from proxy_jsonl) users={len(pos_map)}")
    else:
        pos_map = load_positives_from_dataset(train_set, user_count, item_count)
        print(f"[info] positives(from dataset.pkl) users={len(pos_map)}")

    if len(pos_map) == 0:
        print("[warn] No positives found; training will rely on KD only (no CE).")

    tf.reset_default_graph()
    try:
        tf.set_random_seed(args.seed)
    except Exception:
        pass
    with tf.Session() as sess:
        model = Model(
            user_count,
            item_count,
            cate_count,
            cate_list,
            theme_count,
            theme_list,
            predict_batch_size=args.batch_size,
            predict_ads_num=args.K,
        )

        logits_raw = model.logits_sub_raw
        pi = tf.nn.softmax(logits_raw / float(args.tau), axis=1)

        teacher_q = tf.placeholder(tf.float32, [None, None], name="teacher_q")
        y_list = tf.placeholder(tf.float32, [None, None], name="y_list")
        label_mask = tf.placeholder(tf.float32, [None], name="label_mask")

        kl = tf.reduce_sum(teacher_q * (tf.log(teacher_q + 1e-9) - tf.log(pi + 1e-9)), axis=1)
        kd_loss = tf.reduce_mean(kl)

        ce = -tf.reduce_sum(y_list * tf.log(pi + 1e-9), axis=1)
        ce_masked = tf.reduce_sum(ce * label_mask) / (tf.reduce_sum(label_mask) + 1e-9)

        kd_mix = args.lambda_kd * kd_loss + (1.0 - args.lambda_kd) * ce_masked
        total_loss = kd_mix + tf.constant(float(args.point_loss_weight), dtype=tf.float32) * model.loss

        sess.run(tf.global_variables_initializer())

        restore_vars = [v for v in tf.global_variables() if "Adam" not in v.name and "beta" not in v.name]
        saver_restore = tf.train.Saver(var_list=restore_vars)
        if args.ckpt_in and args.ckpt_in.exists():
            saver_restore.restore(sess, str(args.ckpt_in))
            print(f"[load] restored weights from {args.ckpt_in}")
        else:
            print("[load] no ckpt_in provided -> training from scratch")

        opt = tf.train.AdamOptimizer(learning_rate=args.lr)
        train_vars = tf.trainable_variables()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(total_loss, var_list=train_vars)

        new_vars = [v for v in tf.global_variables() if v not in restore_vars]
        if new_vars:
            sess.run(tf.variables_initializer(new_vars))

        saver = tf.train.Saver(max_to_keep=3)

        steps_per_epoch = (len(users) + args.batch_size - 1) // args.batch_size
        best_loss = float("inf")

        if cands:
            sample_ids, sample_scores = next(iter(cands.values()))
            fallback = (list(sample_ids), list(sample_scores))
        else:
            fallback = ([0] * args.K, [0.0] * args.K)

        for ep in range(1, args.epochs + 1):
            random.shuffle(train_set)
            run_loss = 0.0
            batches = 0

            batch_iter = DataInput(train_set, args.batch_size)
            for _, batch in batch_iter:
                u_batch, i_batch, y_batch, hist_batch, sl_batch = batch
                cand_ids, tq, yl, lm = make_batches(u_batch, i_batch, cands, pos_map, fallback, args.K, args.tau)

                u_vals = np.asarray(u_batch, dtype=np.int32)
                i_vals = np.asarray(i_batch, dtype=np.int32)
                hist_vals = np.asarray(hist_batch, dtype=np.int32)
                sl_vals = np.asarray(sl_batch, dtype=np.int32)
                y_vals = np.asarray(y_batch, dtype=np.float32)

                feed = {
                    model.u: u_vals,
                    model.i: i_vals,
                    model.j: np.zeros_like(i_vals, dtype=np.int32),
                    model.hist_i: hist_vals,
                    model.sl: sl_vals,
                    model.y: y_vals,
                    model.cand_ids: cand_ids,
                    model.use_cand_ids: True,
                    teacher_q: tq,
                    y_list: yl,
                    label_mask: lm,
                }

                loss_val, _ = sess.run([total_loss, train_op], feed_dict=feed)
                run_loss += float(loss_val)
                batches += 1

            avg = run_loss / max(1, batches)
            print(f"[epoch {ep}] total_loss={avg:.6f}")

            if avg < best_loss:
                best_loss = avg
                out_ckpt = str(args.out_dir / "din_kd_best.ckpt")
                saver.save(sess, out_ckpt)
                print(f"[save] {out_ckpt}  (best_loss={best_loss:.6f})")


if __name__ == "__main__":
    main()
