import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9']+", text)


def _fmt_eta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60.0
    return f"{hours:.2f}h"


def build_vocab(
    texts,
    min_freq: int = 1,
    progress_every: int = 0,
) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    t0 = time.perf_counter()
    n = len(texts)
    for t in texts:
        for tok in simple_tokenize(t or ""):
            freq[tok] = freq.get(tok, 0) + 1

        if progress_every and (len(freq) % (progress_every * 5) == 0):
            elapsed = time.perf_counter() - t0
            print(f"[vocab] building... elapsed={_fmt_eta(elapsed)} current_terms={len(freq)}")

    vocab = {}
    for w, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        if c < min_freq:
            continue
        vocab[w] = len(vocab)
    return vocab


def build_tfidf_matrix(
    texts: List[str],
    vocab: Dict[str, int],
    existing_idf: np.ndarray | None = None,
    progress_every: int = 0,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    tokens = [simple_tokenize(t or "") for t in texts]
    if not vocab:
        return sparse.csr_matrix((len(texts), 0), dtype="float32"), np.zeros((0,), dtype="float32")

    term_to_idx = {term: idx for term, idx in vocab.items()}
    indptr = [0]
    indices: List[int] = []
    data: List[float] = []
    doc_freq = np.zeros(len(vocab), dtype="float32") if existing_idf is None else None

    t0 = time.perf_counter()
    n = len(tokens)

    for row, toks in enumerate(tokens):
        counts = Counter(tok for tok in toks if tok in term_to_idx)
        if not counts:
            indptr.append(len(indices))
            continue
        total = float(sum(counts.values()))
        seen_idxs: List[int] = []
        for tok, cnt in counts.items():
            idx = term_to_idx[tok]
            tf = cnt / max(total, 1.0)
            indices.append(int(idx))
            data.append(float(tf))
            seen_idxs.append(int(idx))
            if doc_freq is not None:
                doc_freq[idx] += 1.0

        indptr.append(len(indices))

        if progress_every and (row + 1) % progress_every == 0:
            elapsed = time.perf_counter() - t0
            rate = (row + 1) / max(elapsed, 1e-9)
            remaining = (n - (row + 1)) / max(rate, 1e-9)
            print(
                f"[tfidf] rows={row+1}/{n} ({rate:.1f}/s) elapsed={_fmt_eta(elapsed)} eta={_fmt_eta(remaining)}"
            )

    if len(texts) == 0:
        return sparse.csr_matrix((0, len(vocab)), dtype="float32"), np.zeros((len(vocab),), dtype="float32")

    if existing_idf is None:
        idf = np.log((1.0 + len(texts)) / (1.0 + doc_freq)) + 1.0
    else:
        if len(existing_idf) != len(vocab):
            raise ValueError("Loaded IDF vector dimension does not match vocab size")
        idf = existing_idf.astype("float32")

    mat = sparse.csr_matrix((np.asarray(data, dtype="float32"), np.asarray(indices, dtype="int32"), np.asarray(indptr, dtype="int32")),
                            shape=(len(texts), len(vocab)))
    mat = mat.multiply(idf)
    return mat, idf


def compute_svd(
    tfidf: sparse.csr_matrix,
    emb_dim: int,
    n_iter: int = 7,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if tfidf.shape[0] == 0 or tfidf.shape[1] == 0 or tfidf.nnz == 0:
        return (
            np.zeros((0, tfidf.shape[1]), dtype="float32"),
            np.zeros((tfidf.shape[0], emb_dim), dtype="float32"),
            0.0,
        )

    k = min(emb_dim, min(tfidf.shape))
    if k == 0:
        return (
            np.zeros((0, tfidf.shape[1]), dtype="float32"),
            np.zeros((tfidf.shape[0], 0), dtype="float32"),
            0.0,
        )

    svd = TruncatedSVD(n_components=int(k), n_iter=int(n_iter), random_state=int(random_state))
    reduced = svd.fit_transform(tfidf).astype("float32")
    components = svd.components_.astype("float32")
    evr_sum = float(np.sum(svd.explained_variance_ratio_)) if getattr(svd, "explained_variance_ratio_", None) is not None else 0.0
    return components, reduced, evr_sum


def project_with_components(tfidf: sparse.csr_matrix, components: np.ndarray) -> np.ndarray:
    if tfidf.shape[0] == 0 or tfidf.shape[1] == 0 or tfidf.nnz == 0 or components.size == 0:
        return np.zeros((tfidf.shape[0], components.shape[0]), dtype="float32")
    return tfidf.dot(components.T).astype("float32")


def artifact_paths(prefix: Path) -> Tuple[Path, Path]:
    vocab_path = prefix.with_suffix(".vocab.json")
    svd_path = prefix.with_suffix(".svd.npz")
    return vocab_path, svd_path


def save_artifacts(prefix: Path, vocab: Dict[str, int], idf: np.ndarray, components: np.ndarray) -> None:
    vocab_path, svd_path = artifact_paths(prefix)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "vocab": vocab,
        "idf": idf.tolist(),
        "emb_dim": int(components.shape[0]),
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    np.savez(svd_path, components=components)
    print(f"[write] artifacts -> {vocab_path}, {svd_path}")


def load_artifacts(prefix: Path) -> Tuple[Dict[str, int], np.ndarray, np.ndarray]:
    vocab_path, svd_path = artifact_paths(prefix)
    if not vocab_path.exists() or not svd_path.exists():
        raise SystemExit(
            f"Artifact files not found for prefix '{prefix}'. Run with --mode fit first."
        )
    with open(vocab_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    vocab = {str(k): int(v) for k, v in meta["vocab"].items()}
    idf = np.asarray(meta["idf"], dtype="float32")
    components = np.load(svd_path)["components"].astype("float32")
    return vocab, idf, components


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--catalog_csv",
        default="copy_catalog.csv",
        help="CSV containing the text and ID columns (e.g., copy_catalog.csv or item_catalog.csv)",
    )
    ap.add_argument("--id_col", default="copy_id",
                    help="Name of the column containing the copy ID")
    ap.add_argument("--text_col", default="copy_text",
                    help="Name of the column that stores the copy text")
    ap.add_argument("--out_copy_emb", default="copy_embeddings.csv")
    ap.add_argument("--emb_dim", type=int, default=64, help="Latent dimension when running in fit mode")
    ap.add_argument("--min_freq", type=int, default=1)
    ap.add_argument("--mode", choices=["fit", "transform"], default="fit")
    ap.add_argument("--artifact_prefix", default="copy_embedding_artifacts",
                    help="Path prefix to store or load the vocab/IDF/SVD artifacts")
    ap.add_argument("--progress_every", type=int, default=5000,
                    help="Print progress every N rows during TF-IDF build (0 disables)")
    ap.add_argument("--svd_n_iter", type=int, default=7,
                    help="Number of power iterations for randomized TruncatedSVD")
    ap.add_argument("--svd_random_state", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.catalog_csv, low_memory=False)
    if args.id_col not in df.columns or args.text_col not in df.columns:
        raise SystemExit(
            f"{args.catalog_csv} must contain columns '{args.id_col}' and '{args.text_col}'"
        )

    texts = df[args.text_col].astype(str).tolist()

    prefix = Path(args.artifact_prefix)

    if args.mode == "fit":
        t0 = time.perf_counter()
        vocab = build_vocab(texts, min_freq=args.min_freq, progress_every=args.progress_every)
        print(f"[fit] vocab_size={len(vocab)} elapsed={_fmt_eta(time.perf_counter() - t0)}")

        t1 = time.perf_counter()
        tfidf, idf = build_tfidf_matrix(texts, vocab, progress_every=args.progress_every)
        print(
            f"[fit] tfidf_shape={tuple(tfidf.shape)} nnz={int(tfidf.nnz)} elapsed={_fmt_eta(time.perf_counter() - t1)}"
        )

        t2 = time.perf_counter()
        print(
            f"[svd] starting... emb_dim={args.emb_dim} tfidf_shape={tuple(tfidf.shape)} nnz={int(tfidf.nnz)} "
            f"n_iter={args.svd_n_iter}"
        )
        components, doc_emb, evr_sum = compute_svd(
            tfidf,
            args.emb_dim,
            n_iter=args.svd_n_iter,
            random_state=args.svd_random_state,
        )
        print(
            f"[svd] done. elapsed={_fmt_eta(time.perf_counter() - t2)} "
            f"explained_variance_ratio_sum={evr_sum:.4f}"
        )

        save_artifacts(prefix, vocab, idf, components)
    else:
        vocab, idf, components = load_artifacts(prefix)
        tfidf, _ = build_tfidf_matrix(texts, vocab, existing_idf=idf, progress_every=args.progress_every)
        doc_emb = project_with_components(tfidf, components)

    cols = [f"emb_{i}" for i in range(doc_emb.shape[1])]
    emb_df = pd.DataFrame(doc_emb, columns=cols)
    emb_df.insert(0, args.id_col, df[args.id_col].values)
    emb_df.to_csv(args.out_copy_emb, index=False)

    print(f"[write] {args.out_copy_emb}  (copies={len(emb_df)})")


if __name__ == "__main__":
    main()
