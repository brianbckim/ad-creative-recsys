from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

tf.config.experimental.set_visible_devices([], "GPU")


def load_embedding_table(path: Path, id_col: str) -> Tuple[Dict[int, np.ndarray], int]:
    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise SystemExit(f"{path} must contain column '{id_col}'")
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise SystemExit(f"{path} must contain at least one column prefixed with 'emb_'")
    emb = df[emb_cols].values.astype("float32")
    mapping: Dict[int, np.ndarray] = {}
    for idx, eid in enumerate(df[id_col].astype("int64")):
        mapping[int(eid)] = emb[idx]
    return mapping, emb.shape[1]


def load_copy_catalog(copy_catalog: Path, id_col: str, text_col: str) -> Dict[int, str]:
    df = pd.read_csv(copy_catalog)
    need = {id_col, text_col}
    if not need.issubset(df.columns):
        raise SystemExit(f"{copy_catalog} must contain columns {need}")
    df = df.sort_values(id_col).reset_index(drop=True)
    return {int(row[id_col]): str(row[text_col]) for _, row in df.iterrows()}


def build_theme_contexts(
    samples: pd.DataFrame,
    user_embs: Dict[int, np.ndarray],
    weight_col: str,
    min_positives: int,
    temperature: float,
    uniform_mix: float,
) -> Dict[int, Dict[str, np.ndarray]]:
    if "label" not in samples.columns:
        raise SystemExit("copy_head_train_samples.csv must contain a 'label' column")
    if "user_id" not in samples.columns or "theme_id" not in samples.columns:
        raise SystemExit("Training samples must include user_id and theme_id columns")

    positives = samples[samples["label"] >= 1.0].copy()
    if positives.empty:
        raise RuntimeError("No positive samples available to build ranking contexts.")

    if weight_col and weight_col in positives.columns:
        weights = positives[weight_col].astype("float32").values
    else:
        weights = np.ones(len(positives), dtype="float32")

    sum_vec: Dict[int, np.ndarray] = {}
    sum_weight: Dict[int, float] = {}
    count_map: Dict[int, int] = {}

    for (user_id, theme_id, w) in zip(
        positives["user_id"].astype("int64"),
        positives["theme_id"].astype("int64"),
        weights,
    ):
        if user_id not in user_embs:
            continue
        vec = user_embs[user_id]
        if theme_id not in sum_vec:
            sum_vec[theme_id] = np.zeros_like(vec)
            sum_weight[theme_id] = 0.0
        sum_vec[theme_id] += vec * w
        sum_weight[theme_id] += float(w)
        count_map[theme_id] = count_map.get(theme_id, 0) + 1

    if not sum_weight:
        raise RuntimeError("No positive samples matched the available user embeddings.")

    total_weight = sum(sum_weight.values())
    contexts: Dict[int, Dict[str, np.ndarray]] = {}
    for theme_id, w in sum_weight.items():
        contexts[theme_id] = {
            "user_vec": sum_vec[theme_id] / max(w, 1e-9),
            "raw_weight": w,
            "count": count_map.get(theme_id, 0),
        }

    usable_ids = [tid for tid, ctx in contexts.items() if ctx["count"] >= min_positives]
    if not usable_ids:
        usable_ids = list(contexts.keys())

    weights = np.array([contexts[tid]["raw_weight"] for tid in usable_ids], dtype="float32")
    weights = np.maximum(weights, 1e-9)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    weights = np.power(weights, temperature)
    weights /= weights.sum()
    if uniform_mix > 0:
        uniform = np.full_like(weights, 1.0 / len(weights))
        mix = min(1.0, max(0.0, uniform_mix))
        weights = (1.0 - mix) * weights + mix * uniform

    balanced = {}
    for tid, w in zip(usable_ids, weights):
        ctx = contexts[tid]
        ctx["weight"] = float(w)
        balanced[tid] = ctx
    return balanced


def prepare_copy_matrix(
    copy_text_map: Dict[int, str],
    copy_embs: Dict[int, np.ndarray],
    copy_dim: int,
) -> Tuple[List[int], np.ndarray, List[str]]:
    ids: List[int] = []
    mat: List[np.ndarray] = []
    texts: List[str] = []
    missing: List[int] = []
    for cid, text in copy_text_map.items():
        if cid not in copy_embs:
            missing.append(cid)
            continue
        vec = copy_embs[cid]
        if vec.shape[0] != copy_dim:
            raise RuntimeError("copy embedding dimension mismatch")
        ids.append(cid)
        texts.append(text)
        mat.append(vec)

    if missing:
        missing_str = ", ".join(map(str, missing))
        raise RuntimeError(
            "The following copy_id values are missing from copy_embeddings.csv: "
            f"{missing_str}. Re-run build_copy_embeddings.py using the same copy catalog."
        )

    if not ids:
        raise RuntimeError("No copy_ids were usable for ranking.")

    return ids, np.vstack(mat).astype("float32"), texts


def score_copies(
    model,
    copy_ids: List[int],
    copy_matrix: np.ndarray,
    copy_texts: List[str],
    contexts: Dict[int, Dict[str, np.ndarray]],
    theme_embs: Dict[int, np.ndarray],
    top_k: int,
    out_csv: Path | None,
    summary_k: int,
):
    scores = np.zeros(len(copy_ids), dtype="float32")

    for theme_id in sorted(contexts.keys()):
        ctx = contexts[theme_id]
        if theme_id not in theme_embs:
            continue
        user_batch = np.repeat(ctx["user_vec"][None, :], len(copy_ids), axis=0)
        theme_batch = np.repeat(theme_embs[theme_id][None, :], len(copy_ids), axis=0)
        preds = model.predict(
            {
                "user_emb": user_batch,
                "theme_emb": theme_batch,
                "copy_emb": copy_matrix,
            },
            verbose=0,
            batch_size=min(1024, len(copy_ids)),
        ).reshape(-1)
        scores += ctx["weight"] * preds

    copy_id_arr = np.asarray(copy_ids, dtype="int64")
    order = np.lexsort((copy_id_arr, -scores))
    ranked = pd.DataFrame(
        {
            "rank": np.arange(1, len(copy_ids) + 1),
            "copy_id": copy_id_arr[order],
            "score": scores[order],
            "copy_text": np.array(copy_texts)[order],
        }
    )

    if out_csv:
        ranked.to_csv(out_csv, index=False)
        print(f"[write] ranking -> {out_csv}")

    if contexts:
        rows = sorted(
            [(tid, ctx["weight"], ctx.get("count", 0)) for tid, ctx in contexts.items()],
            key=lambda x: (-float(x[1]), int(x[0])),
        )
        print("[context] top theme weights", min(summary_k, len(rows)))
        for tid, weight, cnt in rows[:summary_k]:
            print(f"  - theme_id={tid}	weight={weight:.4f}	positives={cnt}")

    display_top = min(top_k, len(ranked))
    print("=== Global copy ranking ===")
    for _, row in ranked.head(display_top).iterrows():
        print(
            f"rank={int(row['rank'])}\tcopy_id={int(row['copy_id'])}\t"
            f"score={row['score']:.4f}\ttext={row['copy_text']}"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy_catalog_csv", type=Path, default=Path("copy_catalog.csv"))
    ap.add_argument("--copy_catalog_id_col", default="copy_id")
    ap.add_argument("--copy_catalog_text_col", default="copy_text")
    ap.add_argument("--copy_embeddings_csv", type=Path, default=Path("copy_embeddings.csv"))
    ap.add_argument("--user_emb_csv", type=Path, default=Path("din_user_embeddings.csv"))
    ap.add_argument("--theme_emb_csv", type=Path, default=Path("din_theme_embeddings.csv"))
    ap.add_argument("--copy_head_model", type=Path, default=Path("copy_head_model.keras"))
    ap.add_argument(
        "--train_samples_csv",
        type=Path,
        default=Path("copy_head_train_samples.csv"),
    )
    ap.add_argument("--weight_column", default="sample_weight")
    ap.add_argument("--theme_min_positives", type=int, default=10,
                    help="Themes below this positive-count threshold are dropped unless no themes qualify")
    ap.add_argument(
        "--exclude_theme_id",
        type=int,
        default=None,
        help="Exclude this theme_id from ranking contexts (useful for dropping an 'NA/unknown' dominant theme)",
    )
    ap.add_argument("--theme_weight_temperature", type=float, default=0.5,
                    help="Temperature smoothing (0<temperature<=1) applied to theme weights")
    ap.add_argument("--theme_uniform_mix", type=float, default=0.1,
                    help="Additional uniform prior mass blended into theme weights")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--out_csv", type=Path, default=None)
    ap.add_argument("--context_summary_k", type=int, default=5)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    user_embs, user_dim = load_embedding_table(args.user_emb_csv, "user_id")
    theme_embs, theme_dim = load_embedding_table(args.theme_emb_csv, "theme_id")
    copy_embs, copy_dim = load_embedding_table(args.copy_embeddings_csv, "copy_id")

    if theme_dim != copy_dim:
        raise RuntimeError("Theme and copy embedding dimensions must match.")

    copy_texts = load_copy_catalog(
        args.copy_catalog_csv,
        args.copy_catalog_id_col,
        args.copy_catalog_text_col,
    )
    copy_ids, copy_matrix, copy_text_list = prepare_copy_matrix(
        copy_texts, copy_embs, copy_dim
    )

    samples = pd.read_csv(args.train_samples_csv)
    contexts = build_theme_contexts(
        samples,
        user_embs,
        args.weight_column,
        args.theme_min_positives,
        args.theme_weight_temperature,
        args.theme_uniform_mix,
    )
    if args.exclude_theme_id is not None:
        contexts = {tid: ctx for tid, ctx in contexts.items() if int(tid) != int(args.exclude_theme_id)}
    contexts = {
        tid: ctx for tid, ctx in contexts.items() if tid in theme_embs
    }
    if not contexts:
        raise RuntimeError("No usable themes were found in the embeddings table.")

    model = keras.models.load_model(args.copy_head_model)

    score_copies(
        model=model,
        copy_ids=copy_ids,
        copy_matrix=copy_matrix,
        copy_texts=copy_text_list,
        contexts=contexts,
        theme_embs=theme_embs,
        top_k=args.top_k,
        out_csv=args.out_csv,
        summary_k=args.context_summary_k,
    )


if __name__ == "__main__":
    main()
