import argparse
import pickle
import numpy as np
import pandas as pd
import tensorflow as _tf

tf = _tf.compat.v1
tf.disable_v2_behavior()

from model import Model


def export_embeddings(dataset_pkl: str, ckpt: str,
                      out_user_csv: str, out_theme_csv: str) -> None:
    with open(dataset_pkl, "rb") as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        try:
            theme_list = pickle.load(f)
            user_count, item_count, cate_count, theme_count = pickle.load(f)
        except Exception:
            theme_list = [0] * len(cate_list)
            user_count, item_count, cate_count = pickle.load(f)
            theme_count = 1

    predict_batch_size = 1
    predict_ads_num = 1

    model = Model(user_count, item_count,
                  cate_count, cate_list,
                  theme_count, theme_list,
                  predict_batch_size, predict_ads_num)

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver.restore(sess, ckpt)

        user_emb = sess.run(model.user_emb_w)
        theme_emb = sess.run(model.theme_emb_w)

        u_ids = np.arange(user_emb.shape[0], dtype=np.int64)
        u_cols = [f"emb_{i}" for i in range(user_emb.shape[1])]
        user_df = pd.DataFrame(user_emb, columns=u_cols)
        user_df.insert(0, "user_id", u_ids)
        user_df.to_csv(out_user_csv, index=False)

        t_ids = np.arange(theme_emb.shape[0], dtype=np.int64)
        t_cols = [f"emb_{i}" for i in range(theme_emb.shape[1])]
        theme_df = pd.DataFrame(theme_emb, columns=t_cols)
        theme_df.insert(0, "theme_id", t_ids)
        theme_df.to_csv(out_theme_csv, index=False)

        print(f"[write] {out_user_csv}  (users={len(user_df)})")
        print(f"[write] {out_theme_csv} (themes={len(theme_df)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_pkl", default="din/dataset.pkl")
    ap.add_argument("--ckpt", required=True,
                    help="Path to the DIN checkpoint (e.g., din_base.ckpt)")
    ap.add_argument("--out_user_csv", default="user_embeddings.csv")
    ap.add_argument("--out_theme_csv", default="theme_embeddings.csv")
    args = ap.parse_args()

    export_embeddings(
        dataset_pkl=args.dataset_pkl,
        ckpt=args.ckpt,
        out_user_csv=args.out_user_csv,
        out_theme_csv=args.out_theme_csv,
    )


if __name__ == "__main__":
    main()
