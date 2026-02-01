import os, re, json, hashlib
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
import argparse
import numpy as np
import pandas as pd

def snake(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"__+", "_", s)
    return s.strip("_").lower()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [snake(c) for c in df.columns]
    return df

def safe_parse_dt(x):
    if pd.isna(x): return pd.NaT
    try: return pd.to_datetime(x, errors="coerce", utc=True)
    except Exception: return pd.NaT

def extract_domain(u: str) -> str:
    if not isinstance(u, str) or not u.strip(): return ""
    try: return urlparse(u).netloc or ""
    except Exception: return ""

def load_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    p = str(path).lower()
    if p.endswith(".csv"):
        df = pd.read_csv(path, low_memory=False)
    elif p.endswith(".xlsx"):
        df = pd.read_excel(path, sheet_name=0)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return df

def to_float(v):
    try: return float(str(v).replace(",", "").strip())
    except Exception: return np.nan

def to_int_na(v):
    try:
        s = str(v).replace(",", "").strip()
        if s == "" or s.lower() == "nan":
            return np.nan
        return int(float(s))
    except Exception:
        return np.nan

def _is_exposure_metric(col: str) -> bool:
    c = (col or "").lower()
    if "impression" in c or "impressions" in c:
        return True
    if "reach" in c:
        return True
    if "_views" in c or c.endswith("views") or "video_views" in c:
        return True
    if c.endswith("_view") or "url_view" in c:
        return True
    return False

def _minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - lo) / (hi - lo)

def _renorm_weighted_sum(parts: dict, weights: dict) -> pd.Series:
    active = {}
    aw = {}
    for k, s in parts.items():
        if s is None:
            continue
        if isinstance(s, pd.Series):
            if s.notna().any():
                m = s.mean()
                active[k] = s.fillna(m if np.isfinite(m) else 0.5)
                aw[k] = float(weights.get(k, 0.0))
    if not active:
        idx = next(iter(parts.values())).index if parts else pd.RangeIndex(0)
        return pd.Series(np.full(len(idx), 0.5), index=idx)
    wsum = sum(max(0.0, w) for w in aw.values())
    if wsum <= 0:
        wsum = float(len(aw))
        aw = {k: 1.0 for k in aw}
    out = None
    for k, s in active.items():
        w = aw[k] / wsum
        out = s * w if out is None else out + s * w
    return out

def _normalize_text_for_key(txt: str) -> str:
    txt = (txt or "").strip()
    txt = re.sub(r"\s+", " ", txt)
    txt = re.sub(r"[^\w\s\-]", "", txt)
    return txt[:256].lower()

def stable_item_id(url: str, domain: str, text: str, ts: pd.Timestamp) -> str:
    u = (url or "").strip()
    if u:
        key = u
    else:
        normt = _normalize_text_for_key(text)
        dstr = (ts.date().isoformat() if isinstance(ts, pd.Timestamp) and pd.notna(ts) else "na")
        key = f"{domain or 'nodomain'}|{normt}|{dstr}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]

_SENT_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0, "pos": 1.0, "neu": 0.0, "neg": -1.0}
def map_sentiment(v):
    if isinstance(v, (int, float)) and not pd.isna(v):
        fv = float(v)
        if fv > 0:
            return 1.0
        if fv < 0:
            return -1.0
        return 0.0
    if isinstance(v, str) and v.strip():
        return float(_SENT_MAP.get(v.strip().lower(), np.nan))
    return np.nan

def pick_base_topic(df: pd.DataFrame) -> pd.Series:
    for k in ["cluster_id", "tags_customer", "tags_internal", "entity_urls", "domain"]:
        if k in df.columns:
            s = df[k].fillna("").astype(str)
            return s.where(s != "", other="all")
    return pd.Series(["all"] * len(df), index=df.index)

def build_topic_key(df: pd.DataFrame) -> pd.Series:
    base = pick_base_topic(df)
    return base.astype(str)

def make_cohort_key(topic: pd.Series, domain: pd.Series, post_type: pd.Series, lang: pd.Series,
                    use_post_type: bool, use_lang: bool):
    parts = [topic, domain]
    if use_post_type: parts.append(post_type)
    if use_lang: parts.append(lang)
    arr = pd.concat(parts, axis=1).fillna("").astype(str).agg("|".join, axis=1)
    return arr.replace(r"(\|)+", "|", regex=True).str.strip("|").replace("", "all")

def densify_cohorts(cohort_series: pd.Series, counts: pd.Series,
                    topic: pd.Series, domain: pd.Series, post_type: pd.Series, lang: pd.Series,
                    min_items_per_cohort: int):
    coh = cohort_series.copy()
    need_merge = counts[coh.values].to_numpy() < int(min_items_per_cohort)
    if not need_merge.any(): return coh

    tmp = np.where(need_merge, (topic + "|" + domain + "|" + post_type), coh)

    tmp2 = []
    for i, c in enumerate(tmp):
        if need_merge[i]: tmp2.append(f"{topic.iloc[i]}|{domain.iloc[i]}")
        else: tmp2.append(c)
    tmp2 = pd.Series(tmp2, index=coh.index)

    tmp3 = []
    for i, c in enumerate(tmp2):
        if need_merge[i]: tmp3.append(f"{topic.iloc[i]}")
        else: tmp3.append(c)
    tmp3 = pd.Series(tmp3, index=coh.index)

    tmp4 = []
    for i, c in enumerate(tmp3):
        if need_merge[i]: tmp4.append("all")
        else: tmp4.append(c)
    return pd.Series(tmp4, index=coh.index)

def compute_adaptive_k(counts: pd.Series, k_max: int, target_hist: int) -> dict:
    out = {}
    for coh, n in counts.items():
        n = int(n)
        if n <= 0: out[coh] = 1
        else: out[coh] = int(np.clip(round(n / max(1, target_hist)), 1, int(k_max)))
    return out

def build_session_users(df_slice: pd.DataFrame, gap_min: int) -> pd.Series:
    base = df_slice.copy()
    base["ts_pref"] = base["published_ts"].combine_first(base["indexed_ts"])
    parts = []
    for coh, g in base.groupby("cohort_key", sort=False):
        g = g.sort_values("ts_pref")
        dt = g["ts_pref"].diff().dt.total_seconds().fillna(0) / 60.0
        sess = (dt > gap_min).cumsum()
        s = pd.Series([f"ses:{coh}:s{int(x)}" for x in sess], index=g.index)
        parts.append(s)
    if parts:
        out = pd.concat(parts).sort_index()
        return out.loc[df_slice.index].astype(str)
    else:
        return pd.Series([], index=df_slice.index, dtype=str)

def pseudo_users_assign(df: pd.DataFrame,
                        pseudo_user_strategy: str,
                        session_gap_minutes: int,
                        k_by_cohort: dict,
                        session_domains: set):
    df["cohort_key"] = df["cohort_key"].astype(str)
    domain = df["domain"].astype(str)

    def _cohort_hash_row(row):
        coh = row["cohort_key"]
        k = max(1, int(k_by_cohort.get(coh, 1)))
        h = int(hashlib.sha1(f"{coh}|{row['item_id']}".encode()).hexdigest(), 16)
        bucket = h % k
        return f"coh:{coh}:u{bucket:02d}"

    if pseudo_user_strategy == "cohort":
        return df.apply(_cohort_hash_row, axis=1)

    if pseudo_user_strategy == "session":
        return build_session_users(df, session_gap_minutes)

    mask_session = domain.isin(session_domains) if session_domains else pd.Series(False, index=df.index)
    uid = pd.Series(index=df.index, dtype=object)
    if mask_session.any():
        ses_ids = build_session_users(df.loc[mask_session], session_gap_minutes)
        uid.loc[mask_session] = ses_ids
    if (~mask_session).any():
        uid.loc[~mask_session] = df.loc[~mask_session].apply(_cohort_hash_row, axis=1)
    return uid.astype(str)

def make_label_by_group(df, label_quantile: float|None, min_pos: int, groupby: str):
    if groupby == "none" or groupby not in df.columns:
        if label_quantile is not None:
            base = df["engagement_total"].astype(float)
            nz = base[base > 0]
            thr = nz.quantile(label_quantile) if len(nz) else 0.0
            return ((base > 0) & (base >= thr)).astype(int)
        return (df["engagement_total"] >= float(min_pos)).astype(int)

    if label_quantile is not None:
        base = df["engagement_total"].astype(float)
        thr = df.groupby(groupby)["engagement_total"].transform(
            lambda s: (s[s > 0].astype(float).quantile(label_quantile) if (s > 0).any() else 0.0)
        )
        return ((base > 0) & (base >= thr)).astype(int)
    else:
        return (df["engagement_total"] >= float(min_pos)).astype(int)

def auto_tune_hparams(df: pd.DataFrame, cur: dict, decision_log: list) -> dict:
    out = cur.copy()
    n = len(df)

    if "engagement_total" not in df.columns:
        eng_cols = [c for c in df.columns if c.startswith("article_extended_attributes")] + \
                   [c for c in ["facebook_shares","facebook_reactions_total","facebook_likes",
                                "facebook_reactions_haha","facebook_reactions_angry","facebook_reactions_sad",
                                "facebook_reactions_love","facebook_reactions_wow","twitter_retweets",
                                "article_extended_attributes_url_views","url_views",
                                "pinterest_likes","pinterest_pins","pinterest_repins","youtube_views",
                                "impressions","reach"] if c in df.columns]
        tmp = df[eng_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0) if eng_cols else pd.DataFrame(index=df.index)
        df["engagement_total"] = (tmp.sum(axis=1) if not tmp.empty else 0.0)

    views_cols = [c for c in ["url_views","article_extended_attributes_url_views","impressions","reach"] if c in df.columns]
    views_col = views_cols[0] if views_cols else None
    has_views = views_col is not None
    views_cov = (df[views_col].notna().mean() if has_views else 0.0)
    views_zero_rate = (df[views_col].fillna(0).astype(float) == 0).mean() if has_views else 1.0

    zero_rate = (df["engagement_total"].fillna(0) == 0).mean()
    if zero_rate >= 0.7:
        out["label_quantile"] = 0.85
        decision_log.append(f"[labeling] High sparsity detected (zero-engagement rate={zero_rate:.2f} >= 0.70). "
                            f"Using quantile-based labeling with label_quantile=0.85 to avoid class collapse.")
    elif zero_rate >= 0.4:
        out["label_quantile"] = 0.75
        decision_log.append(f"[labeling] Moderate sparsity (zero-engagement rate={zero_rate:.2f} >= 0.40). "
                            f"Using label_quantile=0.75 for more stable positives.")
    else:
        out["label_quantile"] = out.get("label_quantile", None)
        decision_log.append(f"[labeling] Engagement density looks reasonable (zero-engagement rate={zero_rate:.2f}). "
                            f"Keeping threshold-based labeling (min_engagement_for_pos={out.get('min_engagement_for_pos')}).")

    candidate_keys = [k for k in ["domain","source_type","lang"] if k in df.columns]
    def _group_var(k):
        try:
            m = df.groupby(k)["engagement_total"].median()
            return float(m.var()) if len(m) >= 2 else 0.0
        except Exception:
            return 0.0
    best_key = max(candidate_keys, key=_group_var) if candidate_keys else None
    if best_key and _group_var(best_key) > 0:
        out["label_groupby"] = best_key
        decision_log.append(f"[labeling] Selected label_groupby='{best_key}' due to higher between-group variance "
                            f"({_group_var(best_key):.4f}).")
    else:
        out["label_groupby"] = "none"
        decision_log.append("[labeling] No strong group-wise variance signal; using label_groupby='none'.")

    if not has_views:
        out["on_missing_views"] = "median_by_group" if ("source_type" in df.columns) else "identity"
        decision_log.append(f"[views] No views column present. Setting on_missing_views='{out['on_missing_views']}' "
                            f"to normalize engagement by typical level per source (if available).")
    else:
        if views_cov < 0.5 or views_zero_rate > 0.8:
            out["on_missing_views"] = "median_by_group" if ("source_type" in df.columns) else "identity"
            decision_log.append(f"[views] Views coverage is weak (coverage={views_cov:.2f}, zero-rate={views_zero_rate:.2f}). "
                                f"Using on_missing_views='{out['on_missing_views']}' to avoid unstable ratios.")
        else:
            out["on_missing_views"] = "drop_component"
            decision_log.append(f"[views] Views coverage looks fine (coverage={views_cov:.2f}, zero-rate={views_zero_rate:.2f}). "
                                f"Using on_missing_views='drop_component' (standard CTR-like component).")

    min_items = int(np.clip(max(30, round(n * 0.01)), 30, 200))
    out["min_items_per_cohort"] = min_items
    decision_log.append(f"[cohort] Auto-set min_items_per_cohort={min_items} based on dataset size n={n} (~1%).")

    def _would_fragment(col):
        if col not in df.columns: return True
        k = int(df[col].nunique(dropna=True))
        top_ratio = float(df[col].value_counts(normalize=True, dropna=True).head(1).sum()) if k > 0 else 1.0
        return (k > 10) or (top_ratio < 0.5)

    upl = ("post_type" in df.columns) and (not _would_fragment("post_type"))
    ull = ("lang" in df.columns) and (not _would_fragment("lang"))
    out["use_post_type_in_cohort"] = upl
    out["use_lang_in_cohort"] = ull
    decision_log.append(f"[cohort] use_post_type_in_cohort={upl}, use_lang_in_cohort={ull} "
                        f"(guarding against over-fragmentation).")

    hint = {"twitter.com","x.com","facebook.com","instagram.com","youtube.com","tiktok.com",
            "reddit.com","cnn.com","bbc.co.uk","nytimes.com","washingtonpost.com"}
    if "domain" in df.columns:
        dom_ok = set(df["domain"].dropna().astype(str).str.lower().value_counts().head(20).index.tolist())
    else:
        dom_ok = set()
    pick = sorted(hint.intersection(dom_ok))
    out["session_domains"] = ",".join(pick)
    if pick:
        decision_log.append(f"[session] Selected session_domains={pick} (frequent + hinted domains).")
    else:
        decision_log.append("[session] No hinted frequent domains detected for session split.")

    if ("published_ts" in df.columns) or ("indexed_ts" in df.columns):
        try:
            ts = df["published_ts"].combine_first(df.get("indexed_ts"))
            med_per_dom = (df
                           .assign(_ts=ts)
                           .dropna(subset=["_ts"])
                           .sort_values(["domain","_ts"])
                           .groupby("domain")["_ts"].apply(lambda s: s.diff().dt.total_seconds().dropna()/60.0))
            g = float(med_per_dom.median()) if len(med_per_dom) else 360
            gap_min = int(np.clip(g, 60, 720))
        except Exception:
            gap_min = 360
    else:
        gap_min = 360
    out["session_gap_minutes"] = gap_min
    decision_log.append(f"[session] Set session_gap_minutes={gap_min} (median inter-arrival, clamped to [60,720]).")

    target_hist = int(np.clip(
        round(max(8, min(20, np.sqrt(max(1, n / 50))))),
        8, 20
    ))
    out["pseudo_user_target_hist"] = target_hist
    decision_log.append(f"[pseudo-user] Set pseudo_user_target_hist={target_hist} from dataset scale.")

    dup_rate = 0.0
    try:
        if "item_id" in df.columns:
            dup_rate = 1.0 - float(df["item_id"].nunique())/max(1, len(df["item_id"]))
    except Exception:
        dup_rate = 0.0
    eng_var = float(df["engagement_total"].var()) if n > 1 else 0.0
    if dup_rate > 0.05 and eng_var > 0:
        out["dedup_keep"] = "max_eng"
        decision_log.append(f"[dedup] High duplicate ratio (dup_rate={dup_rate:.2f}) and nonzero variance "
                            f"(eng_var={eng_var:.2f}). Using dedup_keep='max_eng'.")
    else:
        out["dedup_keep"] = "earliest"
        decision_log.append(f"[dedup] Low duplicate ratio or low variance (dup_rate={dup_rate:.2f}, eng_var={eng_var:.2f}). "
                            f"Using dedup_keep='earliest'.")

    parts = {}
    try: parts["volume"] = pd.to_numeric(df.get("proxy_volume"), errors="coerce")
    except: parts["volume"] = None
    try: parts["eng_rate"] = pd.to_numeric(df.get("proxy_eng_rate"), errors="coerce")
    except: parts["eng_rate"] = None
    try: parts["sent_pos"] = pd.to_numeric(df.get("sentiment_num"), errors="coerce")
    except: parts["sent_pos"] = None
    try:
        ts = df["published_ts"].combine_first(df.get("indexed_ts"))
        parts["recency"] = (ts.view("int64") if ts.notna().any() else None)
    except: parts["recency"] = None

    raw_w = dict(zip(["volume","eng_rate","sent_pos","recency"], out.get("weights", (0.25,0.25,0.25,0.25))))
    live = []
    for k, v in parts.items():
        ok = (v is not None) and pd.api.types.is_numeric_dtype(v)
        std_ok = (float(pd.Series(v).std(skipna=True)) > 1e-9) if ok else False
        if ok and std_ok and raw_w.get(k,0.0) > 0:
            live.append(k)
    if live:
        eq = 1.0 / len(live)
        out["weights"] = tuple(eq if k in live else 0.0 for k in ["volume","eng_rate","sent_pos","recency"])
        decision_log.append(f"[weights] Active components={live}. Re-normalized weights evenly across them.")
    else:
        out["weights"] = (0.25,0.25,0.25,0.25)
        decision_log.append("[weights] No active components detected. Falling back to uniform weights.")

    if out["label_groupby"] != "none":
        gsz = df[out["label_groupby"]].value_counts()
        if len(gsz) and int(gsz.min()) < 50:
            decision_log.append(f"[labeling] label_groupby='{out['label_groupby']}' has tiny groups (min={int(gsz.min())}). "
                                f"Reverting to 'none' to avoid unstable per-group thresholds.")
            out["label_groupby"] = "none"

    return out

def build_proxy(
    src_path: Path,
    out_dir: Path,
    min_engagement_for_pos: int = 1,
    max_history_len: int = 20,
    label_quantile: float | None = None,
    half_life_days: float = 7.0,
    weights=(0.25, 0.25, 0.25, 0.25),
    composite_label_quantile: float = 0.7,
    pseudo_user_mode: str = "cohort",
    pseudo_user_k_max: int = 32,
    pseudo_user_target_hist: int = 12,
    session_gap_minutes: int = 360,
    session_domains: str = "",
    label_groupby: str = "none",
    on_missing_views: str = "drop_component",
    dedup_keep: str = "earliest",
    use_post_type_in_cohort: bool = False,
    use_lang_in_cohort: bool = False,
    min_items_per_cohort: int = 50,
    save_quality_report: bool = True,
    auto_tune: bool = False,
    auto_report_out: Path | None = None
):
    decision_log = []

    raw = load_file(src_path)
    df = normalize_columns(raw)

    title_col = "title" if "title" in df.columns else None
    content_col = "content" if "content" in df.columns else None
    title_snip_col = "title_snippet" if "title_snippet" in df.columns else None
    content_snip_col = "content_snippet" if "content_snippet" in df.columns else None
    url_col = "url" if "url" in df.columns else ("display_url" if "display_url" in df.columns else None)
    published_col = "published" if "published" in df.columns else None
    indexed_col = "indexed" if "indexed" in df.columns else None
    lang_col = "lang" if "lang" in df.columns else None
    nsfw_col = "nsfw_level" if "nsfw_level" in df.columns else None
    sentiment_col = "sentiment" if "sentiment" in df.columns else None
    source_type_col = "source_type" if "source_type" in df.columns else None
    post_type_col = "post_type" if "post_type" in df.columns else None

    df["_title"] = df.get(title_col, "")
    df["_content"] = df.get(content_col, "")
    df["_title_snip"] = df.get(title_snip_col, "")
    df["_content_snip"] = df.get(content_snip_col, "")
    df["text"] = (
        df["_title"].fillna("").astype(str).str.strip() + " | " +
        df["_content"].fillna("").astype(str).str.strip()
    ).str.strip(" |")
    mask_empty = df["text"].str.len() == 0
    df.loc[mask_empty, "text"] = (
        df["_title_snip"].fillna("").astype(str).str.strip() + " | " +
        df["_content_snip"].fillna("").astype(str).str.strip()
    ).str.strip(" |")

    df["published_ts"] = df.get(published_col).apply(safe_parse_dt) if published_col else pd.NaT
    df["indexed_ts"] = df.get(indexed_col).apply(safe_parse_dt) if indexed_col else pd.NaT
    ts_pref_src = df["published_ts"].combine_first(df["indexed_ts"])
    tmax = ts_pref_src.max() if ts_pref_src.notna().any() else pd.Timestamp("1970-01-01", tz="UTC")

    df["url_canonical"] = df.get(url_col, "").fillna("").astype(str)
    df["domain"] = df["url_canonical"].apply(extract_domain)
    df["item_id"] = [
        stable_item_id(u, d, txt, ts)
        for u, d, txt, ts in zip(df["url_canonical"], df["domain"], df["text"], ts_pref_src)
    ]

    df["lang"] = df.get(lang_col, "")
    df["sentiment_num"] = df.get(sentiment_col).apply(map_sentiment) if sentiment_col else np.nan
    df["nsfw_level_num"] = df.get(nsfw_col).apply(to_float) if nsfw_col else np.nan
    df["source_type"] = df.get(source_type_col, "")
    df["post_type"] = df.get(post_type_col, "")

    theme_cols_pref = ["main_theme", "theme", "category", "topic", "cluster_id"]
    theme_src_col = None
    for c in theme_cols_pref:
        if c in df.columns:
            theme_src_col = c
            break
    if theme_src_col is not None:
        df["theme_raw"] = df[theme_src_col].astype(str).fillna("")
    else:
        df["theme_raw"] = ""

    eng_cols = [c for c in df.columns if c.startswith("article_extended_attributes")]
    for c in [
        "facebook_shares","facebook_reactions_total","facebook_likes",
        "facebook_reactions_haha","facebook_reactions_angry","facebook_reactions_sad",
        "facebook_reactions_love","facebook_reactions_wow","twitter_retweets",
        "url_views","article_extended_attributes_url_views",
        "pinterest_likes","pinterest_pins","pinterest_repins","youtube_views",
        "impressions","reach"
    ]:
        if c in df.columns and c not in eng_cols: eng_cols.append(c)
    for c in eng_cols:
        df[c] = df[c].apply(to_int_na)

    exposure_cols = [c for c in eng_cols if _is_exposure_metric(c)]
    action_cols = [c for c in eng_cols if c not in exposure_cols]

    df["engagement_actions_total"] = (
        df[action_cols].sum(axis=1, skipna=True).fillna(0.0) if action_cols else 0.0
    )
    df["exposure_total"] = (
        df[exposure_cols].sum(axis=1, skipna=True).fillna(0.0) if exposure_cols else 0.0
    )

    df["engagement_total"] = df["engagement_actions_total"].astype(float)

    cfg = dict(
        min_engagement_for_pos=min_engagement_for_pos,
        label_quantile=label_quantile,
        label_groupby=label_groupby,
        on_missing_views=on_missing_views,
        min_items_per_cohort=min_items_per_cohort,
        use_post_type_in_cohort=use_post_type_in_cohort,
        use_lang_in_cohort=use_lang_in_cohort,
        session_domains=session_domains,
        session_gap_minutes=session_gap_minutes,
        pseudo_user_target_hist=pseudo_user_target_hist,
        dedup_keep=dedup_keep,
        weights=weights
    )
    if auto_tune:
        cfg = auto_tune_hparams(df.copy(), cfg, decision_log)

    min_engagement_for_pos = cfg["min_engagement_for_pos"]
    label_quantile = cfg["label_quantile"]
    label_groupby = cfg["label_groupby"]
    on_missing_views = cfg["on_missing_views"]
    min_items_per_cohort = cfg["min_items_per_cohort"]
    use_post_type_in_cohort = cfg["use_post_type_in_cohort"]
    use_lang_in_cohort = cfg["use_lang_in_cohort"]
    session_domains = cfg["session_domains"]
    session_gap_minutes = cfg["session_gap_minutes"]
    pseudo_user_target_hist = cfg["pseudo_user_target_hist"]
    dedup_keep = cfg["dedup_keep"]
    weights = cfg["weights"]

    if label_quantile is None:
        try:
            pos_rate_thr = float((df["engagement_total"] >= float(min_engagement_for_pos)).mean())
        except Exception:
            pos_rate_thr = 0.0
        if pos_rate_thr > 0.50:
            label_quantile = 0.90
            decision_log.append(
                f"[labeling] Threshold-based labeling would yield pos_rate={pos_rate_thr:.2f} (>0.50). "
                "Switching to quantile-based labeling with label_quantile=0.90 for stability."
            )

    groupby_key = label_groupby if label_groupby in df.columns else "none"
    if label_quantile is not None and groupby_key == "none" and "source_type" in df.columns:
        groupby_key = "source_type"
        decision_log.append("[labeling] Using label_groupby='source_type' for quantile labeling (industry-standard normalization).")

    df["label"] = make_label_by_group(
        df,
        label_quantile=label_quantile,
        min_pos=min_engagement_for_pos,
        groupby=groupby_key,
    ).astype(int)

    df["text_len"] = df["text"].fillna("").str.len()
    df["has_image"] = (df.get("images_url","").fillna("").astype(str).str.len() > 0).astype(int) if "images_url" in df.columns else 0
    df["has_video"] = (df.get("videos_url","").fillna("").astype(str).str.len() > 0).astype(int) if "videos_url" in df.columns else 0

    df["topic_key"] = build_topic_key(df)

    df["proxy_volume"] = df.groupby("topic_key")["item_id"].transform("count").astype(float)

    has_exposure = ("exposure_total" in df.columns) and (df["exposure_total"].notna().any())
    if has_exposure and on_missing_views != "identity":
        denom = df["exposure_total"].fillna(0).astype(float) + 1.0
        df["proxy_eng_rate"] = df["engagement_total"].astype(float) / denom
    else:
        if on_missing_views == "median_by_group":
            by = df.get("source_type", pd.Series("na", index=df.index))
            med = df.groupby(by)["engagement_total"].transform(lambda s: max(s.median(), 1.0))
            df["proxy_eng_rate"] = df["engagement_total"] / med
        elif on_missing_views == "identity":
            df["proxy_eng_rate"] = df["engagement_total"].astype(float)
        else:
            df["proxy_eng_rate"] = np.nan

    sent = df.get("sentiment_num", pd.Series(np.nan, index=df.index)).astype(float)
    df["proxy_sent_pos"] = sent

    ts_pref = df["published_ts"].combine_first(df["indexed_ts"])
    age_days = (tmax - ts_pref).dt.total_seconds() / 86400.0
    age_days = age_days.fillna(age_days.median() if np.isfinite(age_days.median()) else 30.0)
    df["proxy_recency"] = np.exp(-age_days / float(half_life_days))

    df["proxy_volume_n"] = _minmax(df["proxy_volume"])
    df["proxy_eng_rate_n"] = _minmax(df["proxy_eng_rate"]) if df["proxy_eng_rate"].notna().any() else None
    df["proxy_sent_pos_n"] = _minmax(df["proxy_sent_pos"])
    df["proxy_recency_n"] = _minmax(df["proxy_recency"])

    vw, ew, sw, rw = weights
    parts = {
        "volume": df["proxy_volume_n"],
        "eng_rate": df["proxy_eng_rate_n"],
        "sent_pos": df["proxy_sent_pos_n"],
        "recency": df["proxy_recency_n"],
    }
    wmap = {"volume": vw, "eng_rate": ew, "sent_pos": sw, "recency": rw}

    df["proxy_ctr_composite"] = _renorm_weighted_sum(parts, wmap)

    thr = df["proxy_ctr_composite"].quantile(composite_label_quantile)
    df["label_ctr_proxy"] = (df["proxy_ctr_composite"] >= thr).astype(int)

    ts = ts_pref.fillna(pd.to_datetime("1970-01-01", utc=True))
    if dedup_keep == "earliest":
        order = ts.argsort(kind="mergesort")
        keep = df.iloc[order].drop_duplicates(subset=["item_id"], keep="first").index
    elif dedup_keep == "max_eng":
        order = df["engagement_total"].astype(float).fillna(-1e18).argsort(kind="mergesort")[::-1]
        keep = df.iloc[order].drop_duplicates(subset=["item_id"], keep="first").index
    else:
        order = df["text_len"].astype(int).fillna(0).argsort(kind="mergesort")[::-1]
        keep = df.iloc[order].drop_duplicates(subset=["item_id"], keep="first").index

    items_cols = [
        "item_id","url_canonical","domain","text","text_len","lang","sentiment_num",
        "nsfw_level_num","has_image","has_video","source_type","post_type","theme_raw",
        "engagement_total","engagement_actions_total","exposure_total"
    ]
    items = df.loc[sorted(keep), [c for c in items_cols if c in df.columns]].copy().reset_index(drop=True)

    topic = df["topic_key"].astype(str)
    domain = df["domain"].astype(str)
    post_type = df.get("post_type", pd.Series("", index=df.index)).astype(str)
    lang = df.get("lang", pd.Series("", index=df.index)).astype(str)

    base_cohort = make_cohort_key(topic, domain, post_type, lang,
                                  use_post_type_in_cohort, use_lang_in_cohort)
    counts = base_cohort.value_counts()
    cohort_key = densify_cohorts(base_cohort, counts, topic, domain, post_type, lang, min_items_per_cohort)
    df["cohort_key"] = cohort_key

    ct_counts = df["cohort_key"].value_counts()
    k_by_cohort = compute_adaptive_k(ct_counts, k_max=int(pseudo_user_k_max), target_hist=int(pseudo_user_target_hist))

    sess_domains = set([d.strip().lower() for d in (session_domains or "").split(",") if d.strip()])
    if sess_domains:
        decision_log.append(f"[pseudo-user] Mixed/session mode will use these session domains: {sorted(sess_domains)}")

    df["user_id"] = pseudo_users_assign(
        df,
        pseudo_user_strategy=pseudo_user_mode,
        session_gap_minutes=int(session_gap_minutes),
        k_by_cohort=k_by_cohort,
        session_domains=sess_domains
    )

    ts_clean = ts_pref.fillna(pd.to_datetime("1970-01-01", utc=True))
    interactions = (
        pd.DataFrame({
            "user_id": df["user_id"].astype(str),
            "item_id": df["item_id"].astype(str),
            "timestamp": ts_clean.dt.tz_convert("UTC").dt.tz_localize(None),
            "label": df["label"].astype(int),
            "engagement_total": df["engagement_total"].astype(float),
        })
        .sort_values(["user_id","timestamp"])
        .reset_index(drop=True)
    )

    rows = []
    for uid, g in interactions.groupby("user_id", sort=False):
        past = []
        for _, r in g.iterrows():
            rows.append([
                uid, r["item_id"], r["timestamp"], int(r["label"]), int(r["engagement_total"]),
                past[-max_history_len:].copy(), len(past[-max_history_len:])
            ])
            past.append(r["item_id"])
    din = pd.DataFrame(rows, columns=["user_id","item_id","timestamp","label","engagement_total","hist_item_ids","hist_len"])

    agg_cols = ["item_id","proxy_volume","proxy_eng_rate","proxy_sent_pos","proxy_recency",
                "proxy_volume_n","proxy_eng_rate_n","proxy_sent_pos_n","proxy_recency_n",
                "proxy_ctr_composite","label_ctr_proxy"]
    agg = df[agg_cols].groupby("item_id", as_index=False).mean(numeric_only=True)
    items = items.merge(agg, on="item_id", how="left")
    interactions = interactions.merge(
        agg[["item_id","label_ctr_proxy","proxy_ctr_composite"]],
        on="item_id",
        how="left"
    )
    if "theme_raw" in items.columns:
        interactions = interactions.merge(
            items[["item_id","theme_raw"]],
            on="item_id",
            how="left"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    p_items = out_dir / f"proxy_items_{tag}.csv"
    p_inter = out_dir / f"proxy_interactions_{tag}.csv"
    p_din = out_dir / f"proxy_din_interactions_{tag}.jsonl"
    items.to_csv(p_items, index=False)
    interactions.to_csv(p_inter, index=False)
    with open(p_din, "w", encoding="utf-8") as f:
        for _, r in din.iterrows():
            f.write(json.dumps({
                "user_id": r["user_id"],
                "item_id": r["item_id"],
                "timestamp": pd.to_datetime(r["timestamp"]).isoformat(),
                "label": int(r["label"]),
                "engagement_total": int(r["engagement_total"]),
                "hist_item_ids": list(r["hist_item_ids"]),
                "hist_len": int(r["hist_len"])
            }, ensure_ascii=False) + "\n")

    if save_quality_report:
        q = {}
        q["rows"] = int(len(df))
        q["unique_items"] = int(df["item_id"].nunique())
        q["unique_users"] = int(interactions["user_id"].nunique())
        hl = din["hist_len"].astype(int)
        for name, fn in [("mean", np.mean),
                         ("p25", lambda x: np.percentile(x, 25)),
                         ("p50", lambda x: np.percentile(x, 50)),
                         ("p75", lambda x: np.percentile(x, 75)),
                         ("p90", lambda x: np.percentile(x, 90))]:
            try: q[f"hist_{name}"] = float(fn(hl)) if len(hl) else float("nan")
            except: q[f"hist_{name}"] = float("nan")
        q["hist_len_zero_ratio"] = float((hl == 0).mean()) if len(hl) else float("nan")
        ck = df["cohort_key"].value_counts().describe()
        for k in ["min","25%","50%","75%","max","mean"]:
            if k in ck: q[f"cohort_items_{k}"] = float(ck[k])
        p_q = out_dir / f"quality_{tag}.json"
        with open(p_q, "w", encoding="utf-8") as f:
            json.dump(q, f, ensure_ascii=False, indent=2)

    if auto_tune:
        report = {
            "timestamp": tag,
            "decisions": decision_log,
            "final_config": {
                "min_engagement_for_pos": min_engagement_for_pos,
                "label_quantile": label_quantile,
                "label_groupby": label_groupby,
                "on_missing_views": on_missing_views,
                "min_items_per_cohort": min_items_per_cohort,
                "use_post_type_in_cohort": use_post_type_in_cohort,
                "use_lang_in_cohort": use_lang_in_cohort,
                "session_domains": session_domains,
                "session_gap_minutes": session_gap_minutes,
                "pseudo_user_mode": pseudo_user_mode,
                "pseudo_user_k_max": pseudo_user_k_max,
                "pseudo_user_target_hist": pseudo_user_target_hist,
                "dedup_keep": dedup_keep,
                "weights": weights,
                "composite_label_quantile": composite_label_quantile,
                "half_life_days": half_life_days
            }
        }
        print("\n[AUTO-TUNE EXPLANATIONS]")
        for line in decision_log:
            print(" - " + line)
        out_path = (auto_report_out if auto_report_out else (out_dir / f"auto_tune_report_{tag}.json"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[auto-tune] wrote report: {Path(out_path).resolve()}")

    return str(p_items), str(p_inter), str(p_din)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("./proxy"))
    ap.add_argument("--min_engagement_for_pos", type=int, default=1)
    ap.add_argument("--max_history_len", type=int, default=20)
    ap.add_argument("--label_quantile", type=float, default=None)
    ap.add_argument("--half_life_days", type=float, default=7.0)
    ap.add_argument("--weights", type=str, default="0.25,0.25,0.25,0.25")
    ap.add_argument("--composite_label_quantile", type=float, default=0.7)

    ap.add_argument("--pseudo_user_mode", type=str, choices=["cohort","session","mixed"], default="cohort")
    ap.add_argument("--pseudo_user_k_max", type=int, default=32)
    ap.add_argument("--pseudo_user_target_hist", type=int, default=12)
    ap.add_argument("--session_gap_minutes", type=int, default=360)
    ap.add_argument("--session_domains", type=str, default="")
    ap.add_argument("--label_groupby", type=str, choices=["none","source_type","domain","lang"], default="none")
    ap.add_argument("--on_missing_views", type=str, choices=["drop_component","median_by_group","identity"], default="drop_component")
    ap.add_argument("--dedup_keep", type=str, choices=["earliest","max_eng","textlen"], default="earliest")
    ap.add_argument("--use_post_type_in_cohort", action="store_true")
    ap.add_argument("--use_lang_in_cohort", action="store_true")
    ap.add_argument("--min_items_per_cohort", type=int, default=50)
    ap.add_argument("--no_quality_report", action="store_true")

    ap.add_argument("--auto_tune", action="store_true")
    ap.add_argument("--auto_report_out", type=Path, default=None)

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    weights = tuple(float(x) for x in args.weights.split(","))
    paths = build_proxy(
        src_path=args.src,
        out_dir=args.out,
        min_engagement_for_pos=args.min_engagement_for_pos,
        max_history_len=args.max_history_len,
        label_quantile=args.label_quantile,
        half_life_days=args.half_life_days,
        weights=weights,
        composite_label_quantile=args.composite_label_quantile,
        pseudo_user_mode=args.pseudo_user_mode,
        pseudo_user_k_max=args.pseudo_user_k_max,
        pseudo_user_target_hist=args.pseudo_user_target_hist,
        session_gap_minutes=args.session_gap_minutes,
        session_domains=args.session_domains,
        label_groupby=args.label_groupby,
        on_missing_views=args.on_missing_views,
        dedup_keep=args.dedup_keep,
        use_post_type_in_cohort=args.use_post_type_in_cohort,
        use_lang_in_cohort=args.use_lang_in_cohort,
        min_items_per_cohort=args.min_items_per_cohort,
        save_quality_report=(not args.no_quality_report),
        auto_tune=args.auto_tune,
        auto_report_out=args.auto_report_out
    )
    print("\n".join(paths))