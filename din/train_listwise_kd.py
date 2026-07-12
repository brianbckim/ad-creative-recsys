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

try:
    from .input import DataInput
    from .model import MODEL_VERSION, Model
except ImportError:
    from input import DataInput
    from model import MODEL_VERSION, Model


def read_map_csv(path: Path, key_col: str, idx_col: str):
    df = pd.read_csv(path)
    if key_col not in df.columns or idx_col not in df.columns:
        raise ValueError(f"{path} must have columns '{key_col}','{idx_col}'")
    return dict(zip(df[key_col].astype(str), df[idx_col].astype(int)))


def checkpoint_exists(path: Path) -> bool:
    return path.exists() or Path(str(path) + ".index").exists()


def _jsonable_args(args):
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


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


def _ranking_metrics_for_order(candidate_ids, ordered_positions, positives, ks):
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    ordered_positions = np.asarray(ordered_positions, dtype=np.int64)
    positives = set(int(x) for x in positives)
    ordered_ids = candidate_ids[ordered_positions]
    out = {}

    for k in ks:
        top_ids = ordered_ids[: min(int(k), len(ordered_ids))]
        relevance = np.asarray(
            [1.0 if int(iid) in positives else 0.0 for iid in top_ids],
            dtype=np.float64,
        )
        recall = float(relevance.sum() / max(1, len(positives)))
        discounts = 1.0 / np.log2(np.arange(2, 2 + len(relevance)))
        dcg = float(np.sum(relevance * discounts))
        ideal_len = min(len(positives), int(k))
        ideal_dcg = float(np.sum(1.0 / np.log2(np.arange(2, 2 + ideal_len))))
        out[f"recall@{k}"] = recall
        out[f"ndcg@{k}"] = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    return out


def evaluate_teacher_and_student(sess, model, test_set, cands, K, ks, batch_size):
    """Evaluate retrieval and re-ranking on the same held-out candidate slates."""
    rows = []
    for row in test_set:
        if len(row) != 3:
            continue
        uid, hist, pair = row
        if not isinstance(pair, (list, tuple)) or len(pair) < 1:
            continue
        uid = int(uid)
        ids_scores = cands.get(uid)
        if ids_scores is None:
            continue
        ids, teacher_scores = ids_scores
        ids = list(ids[:K])
        teacher_scores = list(teacher_scores[:K])
        if not ids:
            continue
        rows.append((
            uid,
            [int(x) for x in hist],
            {int(pair[0])},
            ids,
            teacher_scores,
        ))

    if not rows:
        return {"eval_examples": 0, "eval_users": 0}

    aggregates = {
        f"{prefix}_{metric}@{k}": []
        for prefix in ("teacher", "student")
        for metric in ("recall", "ndcg")
        for k in ks
    }

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        max_hist = max(1, max(len(row[1]) for row in batch))
        max_cands = max(len(row[3]) for row in batch)

        user_ids = np.asarray([row[0] for row in batch], dtype=np.int32)
        histories = np.zeros((len(batch), max_hist), dtype=np.int32)
        history_lengths = np.zeros((len(batch),), dtype=np.int32)
        candidate_ids = np.zeros((len(batch), max_cands), dtype=np.int32)

        for b, (_, hist, _, ids, _) in enumerate(batch):
            history_lengths[b] = len(hist)
            if hist:
                histories[b, : len(hist)] = np.asarray(hist, dtype=np.int32)
            candidate_ids[b, : len(ids)] = np.asarray(ids, dtype=np.int32)
            if len(ids) < max_cands:
                candidate_ids[b, len(ids) :] = int(ids[-1])

        student_logits = sess.run(
            model.logits_sub_raw,
            feed_dict={
                model.u: user_ids,
                model.hist_i: histories,
                model.sl: history_lengths,
                model.cand_ids: candidate_ids,
                model.use_cand_ids: True,
                model.is_training: False,
            },
        )

        for b, (_, _, positives, ids, teacher_scores) in enumerate(batch):
            n = len(ids)
            teacher_order = np.argsort(-np.asarray(teacher_scores, dtype=np.float64))
            student_order = np.argsort(-np.asarray(student_logits[b, :n], dtype=np.float64))
            teacher_metrics = _ranking_metrics_for_order(
                ids, teacher_order, positives, ks
            )
            student_metrics = _ranking_metrics_for_order(
                ids, student_order, positives, ks
            )
            for k in ks:
                for metric in ("recall", "ndcg"):
                    aggregates[f"teacher_{metric}@{k}"].append(
                        teacher_metrics[f"{metric}@{k}"]
                    )
                    aggregates[f"student_{metric}@{k}"].append(
                        student_metrics[f"{metric}@{k}"]
                    )

    out = {
        "eval_examples": len(rows),
        "eval_users": len({int(row[0]) for row in rows}),
    }
    out.update({key: float(np.mean(values)) for key, values in aggregates.items()})
    for k in ks:
        out[f"delta_ndcg@{k}"] = (
            out[f"student_ndcg@{k}"] - out[f"teacher_ndcg@{k}"]
        )
    return out


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
    ap.add_argument("--eval_k", default="10,20",
                    help="Comma-separated held-out ranking cutoffs")
    ap.add_argument("--checkpoint_metric", choices=["student_ndcg", "total_loss"],
                    default="student_ndcg",
                    help="Select checkpoints by held-out student NDCG or training loss")
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

    if args.ckpt_in and checkpoint_exists(args.ckpt_in):
        prior_config_path = args.ckpt_in.parent / "run_config.json"
        if prior_config_path.exists():
            prior_checkpoint_config = json.loads(
                prior_config_path.read_text(encoding="utf-8")
            )
            prior_version = prior_checkpoint_config.get("model_version")
            if prior_version and prior_version != MODEL_VERSION:
                raise RuntimeError(
                    f"Checkpoint model_version={prior_version}, expected {MODEL_VERSION}"
                )
        else:
            print(
                "[warn] ckpt_in has no run_config.json; architecture compatibility "
                "cannot be verified"
            )

    run_config = {
        "model_version": MODEL_VERSION,
        "args": _jsonable_args(args),
        "counts": {
            "users": user_count,
            "items": item_count,
            "categories": cate_count,
            "themes": theme_count,
            "train_examples": len(train_set),
            "test_examples": len(test_set),
            "users_with_candidates": len(users),
        },
    }
    with open(args.out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

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

        user_grad = tf.gradients(total_loss, model.user_emb_w)[0]
        if user_grad is None:
            raise RuntimeError("user_emb_w is disconnected from the total DIN KD loss")
        user_grad_values = user_grad.values if isinstance(user_grad, tf.IndexedSlices) else user_grad
        user_grad_norm = tf.norm(user_grad_values)

        restore_vars = [
            v for v in tf.global_variables()
            if "/Adam" not in v.name
            and "beta1_power" not in v.name
            and "beta2_power" not in v.name
        ]
        saver_restore = tf.train.Saver(var_list=restore_vars)

        opt = tf.train.AdamOptimizer(learning_rate=args.lr)
        train_vars = tf.trainable_variables()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(total_loss, var_list=train_vars)

        saver = tf.train.Saver(max_to_keep=3)
        sess.run(tf.global_variables_initializer())

        if args.ckpt_in and checkpoint_exists(args.ckpt_in):
            saver_restore.restore(sess, str(args.ckpt_in))
            print(f"[load] restored weights from {args.ckpt_in}")
        else:
            print("[load] no ckpt_in provided -> training from scratch")

        eval_ks = tuple(sorted({int(k) for k in args.eval_k.split(",") if k.strip()}))
        if not eval_ks:
            raise ValueError("--eval_k must contain at least one positive integer")
        if any(k <= 0 for k in eval_ks):
            raise ValueError("--eval_k values must be positive")

        best_value = -float("inf") if args.checkpoint_metric == "student_ndcg" else float("inf")
        metrics_path = args.out_dir / "metrics.jsonl"
        metrics_path.write_text("", encoding="utf-8")

        audit_ids = sorted({int(row[0]) for row in train_set})[:1024]
        audit_tensor = tf.gather(
            model.user_emb_w,
            tf.constant(audit_ids, dtype=tf.int32),
        )
        audit_before = sess.run(audit_tensor)

        if cands:
            sample_ids, sample_scores = next(iter(cands.values()))
            fallback = (list(sample_ids), list(sample_scores))
        else:
            fallback = ([0] * args.K, [0.0] * args.K)

        for ep in range(1, args.epochs + 1):
            random.shuffle(train_set)
            run_loss = 0.0
            run_user_grad_norm = 0.0
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

                loss_val, grad_norm_val, _ = sess.run(
                    [total_loss, user_grad_norm, train_op], feed_dict=feed
                )
                run_loss += float(loss_val)
                run_user_grad_norm += float(grad_norm_val)
                batches += 1

            avg = run_loss / max(1, batches)
            avg_user_grad_norm = run_user_grad_norm / max(1, batches)
            audit_after = sess.run(audit_tensor)
            user_emb_delta = float(np.mean(np.abs(audit_after - audit_before)))
            audit_before = audit_after

            rank_metrics = evaluate_teacher_and_student(
                sess,
                model,
                test_set,
                cands,
                K=args.K,
                ks=eval_ks,
                batch_size=args.batch_size,
            )

            metric_parts = []
            for k in eval_ks:
                metric_parts.append(
                    f"student_ndcg@{k}="
                    f"{rank_metrics.get(f'student_ndcg@{k}', float('nan')):.6f}"
                )
                metric_parts.append(
                    f"delta_ndcg@{k}="
                    f"{rank_metrics.get(f'delta_ndcg@{k}', float('nan')):.6f}"
                )
            metric_text = " ".join(metric_parts)
            print(
                f"[epoch {ep}] total_loss={avg:.6f} "
                f"user_grad_norm={avg_user_grad_norm:.6f} "
                f"user_emb_delta={user_emb_delta:.8f} "
                f"eval_users={rank_metrics.get('eval_users', 0)} {metric_text}"
            )

            row = {
                "epoch": ep,
                "total_loss": avg,
                "user_grad_norm": avg_user_grad_norm,
                "user_emb_delta": user_emb_delta,
                **rank_metrics,
            }
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            if user_emb_delta <= 1e-12:
                raise RuntimeError(
                    "user_emb_w did not update; aborting to avoid exporting random user embeddings"
                )

            if args.checkpoint_metric == "student_ndcg":
                metric_key = f"student_ndcg@{eval_ks[0]}"
                current_value = rank_metrics.get(metric_key, float("nan"))
                if not np.isfinite(current_value):
                    print(
                        f"[warn] {metric_key} unavailable; falling back to minimum total_loss"
                    )
                    current_value = -avg
                improved = current_value > best_value
            else:
                current_value = avg
                improved = current_value < best_value

            if improved:
                best_value = current_value
                out_ckpt = str(args.out_dir / "din_kd_best.ckpt")
                saver.save(sess, out_ckpt)
                print(
                    f"[save] {out_ckpt} "
                    f"({args.checkpoint_metric}={best_value:.6f})"
                )

        print(f"[metrics] wrote: {metrics_path}")


if __name__ == "__main__":
    main()
