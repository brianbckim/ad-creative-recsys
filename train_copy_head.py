import argparse
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from copy_head_model import build_copy_head

tf.config.experimental.set_visible_devices([], "GPU")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_emb_csv", default="user_embeddings.csv")
    ap.add_argument("--theme_emb_csv", default="theme_embeddings.csv")
    ap.add_argument("--copy_emb_csv", default="copy_embeddings.csv")
    ap.add_argument("--train_samples_csv", required=True,
                    help="CSV with columns: user_id, theme_id, copy_id, label")
    ap.add_argument("--sample_weight_col", default="sample_weight",
                    help="Optional column name for sample_weight")
    ap.add_argument("--weight_transform", choices=["none", "log1p", "sqrt"], default="log1p",
                    help="Transform to apply to raw sample weights before training")
    ap.add_argument("--disable_weight_normalize", action="store_true",
                    help="Skip normalization of transformed sample weights")
    ap.add_argument("--weight_eps", type=float, default=1e-6,
                    help="Numerical epsilon used during weight normalization")
    ap.add_argument("--out_model", default="copy_head.keras")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42,
                    help="Global seed for reproducible training (Python/NumPy/TF/Keras)")
    args = ap.parse_args()

    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        tf.keras.utils.set_random_seed(args.seed)
    except Exception:
        tf.random.set_seed(args.seed)

    user_df = pd.read_csv(args.user_emb_csv)
    theme_df = pd.read_csv(args.theme_emb_csv)
    copy_df = pd.read_csv(args.copy_emb_csv)

    user_df = user_df.sort_values("user_id").reset_index(drop=True)
    theme_df = theme_df.sort_values("theme_id").reset_index(drop=True)
    copy_df = copy_df.sort_values("copy_id").reset_index(drop=True)

    user_vecs = user_df.drop(columns=["user_id"]).values.astype("float32")
    theme_vecs = theme_df.drop(columns=["theme_id"]).values.astype("float32")
    copy_vecs = copy_df.drop(columns=["copy_id"]).values.astype("float32")

    user_dim = user_vecs.shape[1]
    theme_dim = theme_vecs.shape[1]
    assert theme_dim == copy_vecs.shape[1], "Theme and copy embedding dimensions must match."

    samples = pd.read_csv(args.train_samples_csv)
    for col in ["user_id", "theme_id", "copy_id", "label"]:
        if col not in samples.columns:
            raise SystemExit(f"Input CSV {args.train_samples_csv} must contain column {col}.")

    u_idx = samples["user_id"].values.astype("int64")
    t_idx = samples["theme_id"].values.astype("int64")
    c_idx = samples["copy_id"].values.astype("int64")
    y = samples["label"].values.astype("float32")

    def _check_bounds(name: str, idx: np.ndarray, limit: int):
        if len(idx) == 0:
            return
        max_idx = int(idx.max())
        if max_idx >= limit:
            raise SystemExit(
                f"{name} index {max_idx} exceeds available embeddings (size={limit}). "
                "Ensure copy/user/theme IDs align with the embedding tables."
            )

    _check_bounds("user_id", u_idx, user_vecs.shape[0])
    _check_bounds("theme_id", t_idx, theme_vecs.shape[0])
    _check_bounds("copy_id", c_idx, copy_vecs.shape[0])

    u_mat = user_vecs[u_idx]
    t_mat = theme_vecs[t_idx]
    c_mat = copy_vecs[c_idx]

    weights = None
    if args.sample_weight_col and args.sample_weight_col in samples.columns:
        weights = samples[args.sample_weight_col].astype("float32").values
        if args.weight_transform == "log1p":
            weights = np.log1p(np.maximum(weights, 0.0))
        elif args.weight_transform == "sqrt":
            weights = np.sqrt(np.maximum(weights, 0.0))
        if not args.disable_weight_normalize and weights.size > 0:
            weights = weights / (np.mean(weights) + args.weight_eps)

    model = build_copy_head(user_dim=user_dim, theme_dim=theme_dim)

    model.fit(
        {"user_emb": u_mat, "theme_emb": t_mat, "copy_emb": c_mat},
        y,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.1,
        shuffle=True,
        sample_weight=weights,
    )

    model.save(args.out_model)
    print(f"[save] copy head model -> {args.out_model}")


if __name__ == "__main__":
    main()
