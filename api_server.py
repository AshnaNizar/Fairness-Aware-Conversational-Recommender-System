import os, io, json, base64, random, re
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Single source of truth: import the real pipeline, don't re-implement it ────
from fa_crs_core import (
    ExposureTracker,
    diversified_injection,
    infer_profile,
    rule_based_cot,
    llm_cot,
    call_llm,
)
try:
    import torch_geometric  # noqa: F401
except ImportError:
    import sys, types
    _pyg = types.ModuleType("torch_geometric")
    _nn = types.ModuleType("torch_geometric.nn")
    _ut = types.ModuleType("torch_geometric.utils")
    _nn.LightGCN = object
    _ut.structured_negative_sampling = None
    _pyg.nn, _pyg.utils = _nn, _ut
    sys.modules.update({"torch_geometric": _pyg,
                        "torch_geometric.nn": _nn,
                        "torch_geometric.utils": _ut})

from fair_rerank import (
    _joint_rerank,
    precision_recall_ndcg,
    compute_fairness_metrics,
)
from metrics_extended import evaluate_extended, gini_exposure, catalog_coverage

# ── Config ────────────────────────────────────────────────────────────────────
KG_OUTPUT_DIR = os.environ.get("KG_OUTPUT_DIR", "outputs/kg")
FAIR_DIR      = os.environ.get("FAIR_DIR", "outputs/fair")
TOP_K         = 10
CANDIDATE_K   = 50
N_INJECT      = 10
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# Ollama (local, free fallback — the backend this project originally shipped)
OLLAMA_URL    = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_TAGS   = OLLAMA_URL.replace("/api/chat", "/api/tags")
OLLAMA_MODEL  = os.environ.get("OLLAMA_MODEL", "llama3")

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def json_response(data):
    return JSONResponse(content=json.loads(json.dumps(data, cls=NumpyEncoder)))


def _find(*names):
    for n in names:
        if os.path.exists(n):
            return n
    return None


# ── Ollama (optional local backend) ───────────────────────────────────────────
def ollama_available():
    """True iff a local Ollama server is reachable. Cheap, 2s timeout."""
    if not REQUESTS_OK:
        return False
    try:
        return requests.get(OLLAMA_TAGS, timeout=2).status_code == 200
    except Exception:
        return False


def call_ollama(prompt, timeout=45):

    if not REQUESTS_OK:
        return None
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False, "options": {"temperature": 0.3},
        }, timeout=timeout)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as e:
        print(f"[ollama] call failed: {e}")
        return None


# ── Offline run configuration (p, alpha) comes from fair_results.json ─────────
def load_run_config():

    path = _find(os.path.join(FAIR_DIR, "fair_results.json"), "fair_results.json")
    cfg = {"primary_p": 0.30, "fair_alpha": 0.15, "results": None, "path": path}
    if path:
        try:
            data = json.load(open(path))
            cfg["primary_p"]  = float(data.get("primary_p", 0.30))
            cfg["fair_alpha"] = float(data.get("fair_alpha", 0.15))
            cfg["results"]    = data
        except Exception as e:
            print(f"[config] could not read {path}: {e}")
    return cfg


RUN_CFG    = load_run_config()
P_GENDER   = RUN_CFG["primary_p"]
P_REGION   = RUN_CFG["primary_p"]
FAIR_ALPHA = RUN_CFG["fair_alpha"]


def reference_block():

    r = RUN_CFG["results"]
    if not r:
        return None
    row = next((c for c in r.get("fut_curve", []) if c["p"] == RUN_CFG["primary_p"]), None)
    return {
        "source": os.path.basename(RUN_CFG["path"] or ""),
        "primary_p": RUN_CFG["primary_p"],
        "fair_alpha": RUN_CFG["fair_alpha"],
        "no_rerank": r.get("norerank_metrics"),
        "fair": row,
        "comparison_table": r.get("comparison_table"),
        "fut_curve": r.get("fut_curve"),
    }


REFERENCE = reference_block()


# ── Data / embeddings: load exactly what LightGCN was trained on ──────────────
READY = False
LOAD_ERROR = None
n_users = n_movies = 0
movies = movies_indexed = None
movie_gender = movie_region = {}
train_df = test_df = None
train_seen = {}
all_scores = None
female_movies = nonwestern_movies = set()
CANDIDATE_CACHE = {}
EXPOSURE_TRACKER = ExposureTracker()   # shared across ALL requests, as offline
LLM_CLIENT = None
BACKEND = "rule_based"   # resolved at startup: "anthropic" | "ollama" | "rule_based"

try:
    print("Loading trained subset (same indices as the embeddings)...")
    movies_p = _find(os.path.join(KG_OUTPUT_DIR, "movies_subset.csv"), "movies_subset.csv")
    train_p  = _find(os.path.join(KG_OUTPUT_DIR, "train_subset.csv"),  "train_subset.csv")
    test_p   = _find(os.path.join(KG_OUTPUT_DIR, "test_subset.csv"),   "test_subset.csv")
    emb_p    = _find(os.path.join(KG_OUTPUT_DIR, "user_movie_emb.pt"), "user_movie_emb.pt")
    if not all([movies_p, train_p, emb_p]):
        raise FileNotFoundError(
            "Need movies_subset.csv, train_subset.csv and user_movie_emb.pt "
            f"(looked in {KG_OUTPUT_DIR}/ and cwd). Run lightgcn_pyg.py first."
        )

    movies = pd.read_csv(movies_p).dropna(subset=["movie_idx"]).copy()
    movies["movie_idx"] = movies["movie_idx"].astype(int)
    for col, d in [("director", "Unknown Director"), ("director_gender", "unknown"),
                   ("region", "unknown"), ("genres", "Unknown")]:
        if col in movies.columns:
            movies[col] = movies[col].fillna(d)

    train_df = pd.read_csv(train_p)
    test_df  = pd.read_csv(test_p) if test_p else None

    emb = torch.load(emb_p, map_location="cpu")
    user_emb  = emb["user_emb"].cpu().numpy()
    movie_emb = emb["movie_emb"].cpu().numpy()
    n_users, n_movies = user_emb.shape[0], movie_emb.shape[0]
    print(f"  embeddings: users={user_emb.shape}, movies={movie_emb.shape}")

    movies_indexed = movies.set_index("movie_idx")
    movie_gender = movies_indexed["director_gender"].to_dict()
    movie_region = movies_indexed["region"].to_dict()
    female_movies     = {m for m, g in movie_gender.items() if g == "female"}
    nonwestern_movies = {m for m, r in movie_region.items() if r == "non-western"}

    train_seen = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    USER_EMB, MOVIE_EMB = user_emb, movie_emb

    SUPPLY = {
        "pct_female_catalog": round(len(female_movies) / max(len(movies), 1) * 100, 2),
        "pct_nonwestern_catalog": round(len(nonwestern_movies) / max(len(movies), 1) * 100, 2),
    }

    try:
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        LLM_CLIENT = anthropic.Anthropic(api_key=key) if key else None
    except Exception:
        LLM_CLIENT = None

    if LLM_CLIENT is not None:
        BACKEND = "anthropic"
    elif ollama_available():
        BACKEND = "ollama"
    else:
        BACKEND = "rule_based"

    READY = True
    backend_label = {
        "anthropic": f"Anthropic ({ANTHROPIC_MODEL})",
        "ollama": f"Ollama ({OLLAMA_MODEL} @ {OLLAMA_URL})",
        "rule_based": "rule-based (no LLM)",
    }[BACKEND]
    print(f"Ready. {n_users} users, {n_movies} movies. "
          f"p={P_GENDER}, alpha={FAIR_ALPHA}, backend={backend_label}")
    print(f"Catalogue supply: {SUPPLY}")
except Exception as e:
    import traceback
    LOAD_ERROR = f"{e}"
    print(f"Load error: {e}\n{traceback.format_exc()}")


def user_scores(uid):
    s = (USER_EMB[uid] @ MOVIE_EMB.T).astype(np.float64)
    for m in train_seen.get(uid, set()):
        if m < len(s):
            s[m] = -np.inf
    return s


def build_candidates(uid):
    s = user_scores(uid)
    seen_u = train_seen.get(uid, set())
    k = min(CANDIDATE_K, len(s))
    top = np.argpartition(s, -k)[-k:]
    top = top[np.argsort(s[top])[::-1]]
    pool = set(int(m) for m in top)

    female_scores     = {m: float(s[m]) for m in female_movies if m < len(s)}
    nonwestern_scores = {m: float(s[m]) for m in nonwestern_movies if m < len(s)}
    pool.update(diversified_injection(female_scores, seen_u, female_movies,
                                      n_inject=N_INJECT, tracker=EXPOSURE_TRACKER,
                                      sample_from_top=30, seed=uid))
    pool.update(diversified_injection(nonwestern_scores, seen_u, nonwestern_movies,
                                      n_inject=N_INJECT, tracker=EXPOSURE_TRACKER,
                                      sample_from_top=30, seed=uid))

    pl = sorted(pool, key=lambda m: s[m] if s[m] > -1e8 else -1e9, reverse=True)
    return [(int(m), float(s[m])) for m in pl]


def candidates_for(uid):
    if uid not in CANDIDATE_CACHE:
        CANDIDATE_CACHE[uid] = build_candidates(uid)
    return CANDIDATE_CACHE[uid]


def apply_genre_filters(cands, excluded_genres, include_genres, require_all=False):
    out = cands
    if include_genres:
        il = [g.lower() for g in include_genres]
        match = all if require_all else any
        out = [(m, s) for m, s in out
               if m in movies_indexed.index
               and match(g in str(movies_indexed.loc[m, "genres"]).lower() for g in il)]
    if excluded_genres:
        el = [g.lower() for g in excluded_genres]
        out = [(m, s) for m, s in out
               if m not in movies_indexed.index
               or not any(g in str(movies_indexed.loc[m, "genres"]).lower() for g in el)]
    return out


def get_recs(uid, excl=None, incl=None, require_all=False):
    cands = candidates_for(uid)
    filtered = apply_genre_filters(cands, excl, incl, require_all)
    baseline = [m for m, _ in filtered[:TOP_K]]
    if not filtered:
        return [], [], []
    fair, flags = _joint_rerank(filtered, movie_gender, movie_region,
                                P_GENDER, P_REGION, k=TOP_K, alpha=FAIR_ALPHA)
    return baseline, fair, flags



SESSION_COHORT = {"fair": {}, "base": {}}
SEED_COHORT    = {"fair": {}, "base": {}}
COHORT_MIN_USERS = 2   # below this, cohort metrics are reported as None
ACCURACY_WARMUP_USERS = int(os.environ.get("ACCURACY_WARMUP_USERS", "200"))


def _merged_cohort(which):
    """Seed sample first, then live clicks override (a user clicked live is
    scored on their live list, not the seed one)."""
    merged = dict(SEED_COHORT[which])
    merged.update(SESSION_COHORT[which])
    return merged


def list_spd(rec_list, movie_group, protected_val, unprotected_val):
    """Same convention as fair_rerank.compute_fairness_metrics, one list."""
    a = sum(1 for m in rec_list if movie_group.get(m) == protected_val)
    b = sum(1 for m in rec_list if movie_group.get(m) == unprotected_val)
    total = a + b
    return 0.0 if total == 0 else a / total - b / total


def _gt_for(uid):
    if test_df is None:
        return None
    gt = test_df[test_df["user_idx"] == uid]
    return None if gt.empty else gt


def list_accuracy(uid, rec_list):
    """Real Precision@10 / Recall@10 / NDCG@10 for THIS list against the
    held-out test split, via fair_rerank.precision_recall_ndcg."""
    gt = _gt_for(uid)
    if gt is None or not rec_list:
        return None, None, None
    prec, rec, ndcg = precision_recall_ndcg({uid: rec_list}, gt)
    return float(prec), float(rec), float(ndcg)


def list_eod(uid, rec_list, attribute_col, group_a, group_b):
    """Equal Opportunity Difference for THIS list, via the same
    fair_rerank.compute_fairness_metrics used offline. Returns None when the
    user's test set contains no relevant item from one of the two groups —
    TPR is undefined there, and the offline code skips those users too."""
    gt = _gt_for(uid)
    if gt is None or not rec_list:
        return None
    movie_group = movies_indexed[attribute_col].to_dict()
    relevant = set(gt["movie_idx"])
    rel_a = sum(1 for m in relevant if movie_group.get(m) == group_a)
    rel_b = sum(1 for m in relevant if movie_group.get(m) == group_b)
    if rel_a == 0 or rel_b == 0:
        return None
    _, eod = compute_fairness_metrics({uid: rec_list}, gt, movies,
                                      attribute_col, group_a, group_b)
    return float(eod)


def list_exposure_gap(rec_list, movie_group, protected_val):
    """Rank-discounted exposure gap for one list, same 1/log2(rank+1) discount
    as metrics_extended.exposure_gap."""
    ep = eu = 0.0
    n_p = n_u = 0
    for rank, m in enumerate(rec_list[:TOP_K], start=1):
        e = 1.0 / np.log2(rank + 1)
        if movie_group.get(m) == protected_val:
            ep += e; n_p += 1
        else:
            eu += e; n_u += 1
    ap = ep / n_p if n_p else 0.0
    au = eu / n_u if n_u else 0.0
    return ap - au


def _r(v, nd=4):
    return None if v is None else round(float(v), nd)


def per_list_metrics(uid, rec_list, accuracy_valid=True):

    if not rec_list:
        return {}
    if accuracy_valid:
        prec, rec, ndcg = list_accuracy(uid, rec_list)
    else:
        prec = rec = ndcg = None
    ext_g = evaluate_extended({uid: rec_list}, movies, "director_gender",
                              "female", n_movies, k=TOP_K)
    ext_r = evaluate_extended({uid: rec_list}, movies, "region",
                              "non-western", n_movies, k=TOP_K)
    n = max(len(rec_list), 1)
    return {
        "ndcg_at_10":          _r(ndcg),
        "precision_at_10":     _r(prec),
        "recall_at_10":        _r(rec),
        "gender_spd":          _r(list_spd(rec_list, movie_gender, "female", "male")),
        "gender_eod":          _r(list_eod(uid, rec_list, "director_gender", "female", "male")),
        "region_spd":          _r(list_spd(rec_list, movie_region, "non-western", "western")),
        "region_eod":          _r(list_eod(uid, rec_list, "region", "non-western", "western")),
        "gender_rND":          _r(ext_g["rND"]),
        "gender_exposure_gap": _r(list_exposure_gap(rec_list, movie_gender, "female")),
        "region_rND":          _r(ext_r["rND"]),
        "region_exposure_gap": _r(list_exposure_gap(rec_list, movie_region, "non-western")),
        "pct_female":          round(sum(1 for m in rec_list if movie_gender.get(m) == "female") / n * 100, 1),
        "pct_nonwestern":      round(sum(1 for m in rec_list if movie_region.get(m) == "non-western") / n * 100, 1),
    }


def cohort_metrics(which):
    recs = _merged_cohort(which)
    n = len(recs)
    n_live = len(SESSION_COHORT[which])
    n_seed = n - n_live

    # session-average accuracy against the held-out test split
    acc = {"ndcg_at_10": None, "precision_at_10": None, "recall_at_10": None,
           "n_scored_users": 0}
    if test_df is not None and n >= 1:
        scored = {u: r for u, r in recs.items()
                  if not test_df[test_df["user_idx"] == u].empty}
        if scored:
            gt = test_df[test_df["user_idx"].isin(scored.keys())]
            prec, rec, ndcg = precision_recall_ndcg(scored, gt)
            acc = {"ndcg_at_10": _r(ndcg), "precision_at_10": _r(prec),
                   "recall_at_10": _r(rec), "n_scored_users": len(scored)}

    if n < COHORT_MIN_USERS:
        return {"n_users": n, "n_live": n_live, "n_seed": n_seed,
                "gender_collapse_rate": None,
                "region_collapse_rate": None, "gini_exposure": None,
                "catalog_coverage": None, **acc}
    ext_g = evaluate_extended(recs, movies, "director_gender", "female", n_movies, k=TOP_K)
    ext_r = evaluate_extended(recs, movies, "region", "non-western", n_movies, k=TOP_K)
    return {
        "n_users":              n,
        "n_live":               n_live,
        "n_seed":               n_seed,
        "gender_collapse_rate": _r(ext_g["collapse_rate"]),
        "region_collapse_rate": _r(ext_r["collapse_rate"]),
        "gender_collapse_top":  [[int(m), int(c)] for m, c in ext_g["collapse_top_items"]],
        "region_collapse_top":  [[int(m), int(c)] for m, c in ext_r["collapse_top_items"]],
        "gini_exposure":        _r(gini_exposure(recs, k=TOP_K)),
        "catalog_coverage":     _r(catalog_coverage(recs, n_movies, k=TOP_K)),
        **acc,
    }



TABLE_ROWS = [
    ("Gender SPD",           "gender_spd",          "per_list"),
    ("Gender EOD",           "gender_eod",          "per_list"),
    ("Region SPD",           "region_spd",          "per_list"),
    ("Region EOD",           "region_eod",          "per_list"),
    ("Gender rND",           "gender_rND",          "per_list"),
    ("Gender Exposure Gap",  "gender_exposure_gap", "per_list"),
    ("Region rND",           "region_rND",          "per_list"),
    ("Region Exposure Gap",  "region_exposure_gap", "per_list"),
    ("NDCG@10 (avg)",        "ndcg_at_10",          "cohort"),
    ("Precision@10 (avg)",   "precision_at_10",     "cohort"),
    ("Recall@10 (avg)",      "recall_at_10",        "cohort"),
    ("Gender Collapse Rate", "gender_collapse_rate", "cohort"),
    ("Region Collapse Rate", "region_collapse_rate", "cohort"),
    ("Gini Exposure",        "gini_exposure",       "cohort"),
    ("Catalog Coverage",     "catalog_coverage",    "cohort"),
]


def build_table(fair_pl, base_pl, fair_co, base_co):
    """The full 15-metric No-Rerank / FA*IR / Delta table, live."""
    rows = []
    for label, key, scope in TABLE_ROWS:
        src_f, src_b = (fair_co, base_co) if scope == "cohort" else (fair_pl, base_pl)
        fv, bv = src_f.get(key), src_b.get(key)
        delta = None if (fv is None or bv is None) else round(fv - bv, 4)
        rows.append({"metric": label, "no_rerank": bv, "fair": fv,
                     "delta": delta, "scope": scope})
    return rows


def compute_live_metrics(uid, fair_list, base_list, genre_filtered=False):
    # record this user's lists in the running cohort before computing cohort stats
    if fair_list:
        SESSION_COHORT["fair"][uid] = list(fair_list)
    if base_list:
        SESSION_COHORT["base"][uid] = list(base_list)

    acc_valid = not genre_filtered
    fair_pl = per_list_metrics(uid, fair_list, accuracy_valid=acc_valid)
    base_pl = per_list_metrics(uid, base_list, accuracy_valid=acc_valid)
    fair_co = cohort_metrics("fair")
    base_co = cohort_metrics("base")

    def _panel(v):
        return "n/a" if v is None else v

    notes = {
        "cohort_metrics": ("NDCG/Precision/Recall, Collapse Rate, Gini Exposure and "
                           "Catalog Coverage are cross-user quantities, computed over "
                           f"{fair_co['n_users']} users "
                           f"({fair_co.get('n_seed', 0)} startup warm-up + "
                           f"{fair_co.get('n_live', 0)} live this session; accuracy "
                           f"over {fair_co.get('n_scored_users', 0)} with test ground "
                           "truth). Per-user NDCG is near-binary and is reported "
                           "separately as a snapshot only."),
        "eod": ("EOD is null when the user's test set has no relevant item from "
                "one of the two groups — TPR is undefined; the offline eval "
                "skips those users identically."),
    }
    if genre_filtered:
        notes["accuracy"] = ("This list is genre-filtered, so the per-user "
                             "NDCG/Precision/Recall snapshot is n/a: it measures "
                             "agreement with the user's actual (cross-genre) held-out "
                             "history, which a single-genre request overrides. Session-"
                             "average accuracy still reflects unfiltered requests.")

    return {
        "spd":  _panel(fair_pl.get("gender_spd")),
        "oead": _panel(fair_pl.get("gender_exposure_gap")),
        "ndcg": _panel(fair_co.get("ndcg_at_10")),
        "ndcg_user": _panel(fair_pl.get("ndcg_at_10")),   # per-user snapshot

        # the full 15-row table, live, in results_table.txt order
        "table": build_table(fair_pl, base_pl, fair_co, base_co),

        "per_list":       {"fair": fair_pl, "no_rerank": base_pl},
        "cohort":         {"fair": fair_co, "no_rerank": base_co},
        "genre_filtered": genre_filtered,
        "targets": {
            "p_gender": P_GENDER, "p_region": P_REGION, "alpha": FAIR_ALPHA,
            **SUPPLY,
        },
        "notes": notes,
        # offline aggregate context (the numbers in results_table.txt), labelled
        "reference": REFERENCE,
    }


def format_metrics_summary(metrics):
    t = metrics["targets"]
    pl = metrics["per_list"]["fair"]
    co = metrics["cohort"]["fair"]

    def fmt(v):
        return "  n/a  " if v is None else f"{v:+.4f}"

    lines = [
        f"Live evaluation of this list — FA*IR p={t['p_gender']:g}, alpha={t['alpha']:g}",
        f"(catalogue supply: {t['pct_female_catalog']}% female-directed, "
        f"{t['pct_nonwestern_catalog']}% non-western)",
        "",
        f"{'Metric':<22}{'No-Rerank':>11}{'FA*IR':>11}{'Delta':>11}",
        "-" * 55,
    ]
    for row in metrics["table"]:
        mark = "*" if row["scope"] == "cohort" else " "
        lines.append(f"{row['metric']+mark:<22}{fmt(row['no_rerank']):>11}"
                     f"{fmt(row['fair']):>11}{fmt(row['delta']):>11}")
    lines += [
        "-" * 55,
        f"* cohort metric, averaged over {co['n_users']} list(s) served this "
        f"session (accuracy over {co.get('n_scored_users', 0)} with test "
        f"ground truth). Per-user NDCG is near-binary, so accuracy is shown as "
        f"a session average — it converges to the offline ~0.011 as more users load.",
        f"This list: {pl.get('pct_female', 0)}% female-directed, "
        f"{pl.get('pct_nonwestern', 0)}% non-western.",
    ]
    if metrics.get("genre_filtered"):
        lines.append("Note: genre-filtered request — the per-user accuracy snapshot "
                     "is n/a, but session-average accuracy reflects unfiltered lists.")
    return "\n".join(lines)


# ── Chart: bars drawn from the live lists, nothing pre-set ────────────────────
def make_chart_b64(base_list, fair_list):
    BG, PANEL = "#191919", "#1C1C1C"
    C_BASE, C_FAIR = "#3b82f6", "#7c3aed"
    plt.style.use("dark_background")

    def pcts(lst):
        n = max(len(lst), 1)
        return [sum(1 for m in lst if movie_gender.get(m) == "female") / n * 100,
                sum(1 for m in lst if movie_region.get(m) == "non-western") / n * 100]

    bp, fp = pcts(base_list), pcts(fair_list)
    base_spd = abs(list_spd(base_list, movie_gender, "female", "male"))
    fair_spd = abs(list_spd(fair_list, movie_gender, "female", "male"))

    fig = plt.figure(figsize=(11, 4), facecolor=BG)
    gs = fig.add_gridspec(1, 2, wspace=0.35, left=0.08, right=0.96, top=0.82, bottom=0.18)
    ax_bar, ax_spd = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    for ax in (ax_bar, ax_spd):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2d2d2d")

    x, w = np.arange(2), 0.3
    for vals, col, lbl, off in [(bp, C_BASE, "No rerank", -w / 2),
                                (fp, C_FAIR, "FA\u2605IR", +w / 2)]:
        bars = ax_bar.bar(x + off, vals, w, color=col, alpha=0.85, label=lbl)
        for b, v in zip(bars, vals):
            ax_bar.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=8, color="#94a3b8")
    ax_bar.axhline(P_GENDER * 100, color="#22c55e", lw=1.1, ls="--", alpha=0.7,
                   label=f"target p={P_GENDER:g}")
    ax_bar.axhline(SUPPLY["pct_female_catalog"], color="#ef4444", lw=1.0, ls=":", alpha=0.7,
                   label=f"female supply {SUPPLY['pct_female_catalog']:.1f}%")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["% Female Dir", "% Non-Western"], color="#94a3b8", fontsize=8)
    ax_bar.set_title("Diversity of this list", color="#f1f5f9", fontsize=10, fontweight="bold")
    ax_bar.legend(fontsize=6, framealpha=0.1, labelcolor="#94a3b8")
    ax_bar.tick_params(colors="#4a5568")

    ax_spd.barh(["No rerank |SPD|", "FA\u2605IR |SPD|"], [base_spd, fair_spd],
                color=[C_BASE, C_FAIR], alpha=0.85, height=0.4)
    for y, val in enumerate([base_spd, fair_spd]):
        ax_spd.text(val + 0.01, y, f"{val:.2f}", va="center", color="#94a3b8", fontsize=9)
    ax_spd.set_xlim(0, 1.1)
    ax_spd.set_xlabel("|Gender SPD| (lower = fairer)", color="#94a3b8", fontsize=8)
    ax_spd.set_title("Gender SPD, this list", color="#f1f5f9", fontsize=10, fontweight="bold")
    ax_spd.tick_params(colors="#94a3b8")

    fig.suptitle("FA\u2605IR Fairness Analytics (live)", color="#f1f5f9",
                 fontsize=12, fontweight="bold", y=0.97)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Helpers ───────────────────────────────────────────────────────────────────
def movie_to_dict(m, flag):
    if movies_indexed is not None and m in movies_indexed.index:
        row = movies_indexed.loc[m]
        return {
            "id": int(m),
            "title": str(row.get("title", f"Movie {m}")),
            "director": str(row.get("director", "Unknown")),
            "gender": str(row.get("director_gender", "unknown")),
            "region": str(row.get("region", "unknown")).title(),
            "year": "",
            "genres": str(row.get("genres", "")).replace("|", ", "),
            "flag": flag,
        }
    return {"id": int(m), "title": f"Movie {m}", "director": "Unknown",
            "gender": "unknown", "region": "Unknown", "year": "", "genres": "", "flag": flag}


def user_stats(uid):
    seen = train_seen.get(uid, set())
    gc = {}
    for m in seen:
        if m in movies_indexed.index:
            for g in str(movies_indexed.loc[m, "genres"]).split("|"):
                g = g.strip()
                if g and g != "Unknown":
                    gc[g] = gc.get(g, 0) + 1
    top = ", ".join(k for k, _ in sorted(gc.items(), key=lambda x: -x[1])[:3]) or "—"
    prof = infer_profile(uid, train_df, movies)
    return {"id": int(uid), "n_rated": int(len(seen)), "top_genres": top,
            "diversity_appetite": prof["diversity"], "avg_rating": None}


def cand_dicts_for(uid, fair_list, flags):
    """Build the candidate dicts fa_crs_core's CoT functions expect."""
    score_of = dict(candidates_for(uid))
    out = []
    for m, f in zip(fair_list, flags):
        row = movies_indexed.loc[m] if m in movies_indexed.index else {}
        out.append({
            "movie_idx": int(m),
            "title": str(row.get("title", f"Movie {m}")),
            "genres": str(row.get("genres", "")),
            "director": str(row.get("director", "Unknown")),
            "director_gender": str(row.get("director_gender", "unknown")),
            "region": str(row.get("region", "unknown")),
            "score": float(score_of.get(m, 0.0)),
            "fairness_flag": f,
        })
    return out


# ── Intent parsing: LLM when available, rule-based fallback ───────────────────
KNOWN_GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
                "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]

GENRE_SYNONYMS = [
    (["rom com", "rom-com", "romcom", "romantic comedy"], ["Romance", "Comedy"]),
    (["sci fi", "sci-fi", "scifi", "science fiction"],     ["Sci-Fi"]),
    (["scary", "spooky", "slasher"],                        ["Horror"]),
    (["animated", "cartoon", "anime"],                      ["Animation"]),
    (["kids", "family", "children's"],                      ["Children"]),
    (["docu", "documentaries"],                             ["Documentary"]),
    (["noir"],                                              ["Film-Noir"]),
    (["musicals"],                                          ["Musical"]),
    (["thrillers", "suspense"],                             ["Thriller"]),
    (["funny", "comedies"],                                 ["Comedy"]),
    (["romantic", "romances", "love story"],               ["Romance"]),
    (["westerns", "cowboy"],                                ["Western"]),
    (["war film", "war movie"],                             ["War"]),
    (["superhero", "action-packed"],                        ["Action"]),
]


def extract_genres(msg):
    msg = msg.lower()
    genres, require_all = [], False
    for phrases, tags in GENRE_SYNONYMS:
        if any(p in msg for p in phrases):
            genres.extend(tags)
            if len(tags) > 1:
                require_all = True
    # literal single-word genre names ("comedy", "horror", ...)
    for g in KNOWN_GENRES:
        gl = g.lower()
        if gl in msg and g not in genres:
            genres.append(g)
    # dedupe, preserve order
    seen, out = set(), []
    for g in genres:
        if g not in seen:
            seen.add(g); out.append(g)
    return out, require_all

INTENT_PROMPT = """You are the intent parser for a fairness-aware movie recommender.
Reply with ONE valid JSON object and nothing else.

Schemas:
  {{"intent":"recommend","include_genres":["Genre"],"require_all":false,"reason":"one sentence"}}
  {{"intent":"filter","exclude_genres":["Genre"],"reason":"one sentence"}}
  {{"intent":"explain","movie_title":"exact title"}}
  {{"intent":"reset"}}
  {{"intent":"question","answer":"2-3 plain sentences"}}

Valid genres: {genres}
For compound genres like "rom com" (Romance + Comedy) or "sci-fi horror", list
every genre in include_genres and set "require_all": true.

Current recommendations:
{context}

User message: {message}"""


def rule_based_intent(message):
    msg = message.lower()
    if "reset" in msg or "clear" in msg:
        return {"intent": "reset"}
    mentioned, require_all = extract_genres(message)
    if any(w in msg for w in ["no ", "without", "exclude", "don't want", "dont want",
                              "hate", "avoid"]) and mentioned:
        return {"intent": "filter", "exclude_genres": mentioned,
                "reason": f"Filtering out {', '.join(mentioned)}."}
    if "why" in msg or "explain" in msg:
        m = re.search(r'"([^"]+)"', message)
        return {"intent": "explain", "movie_title": m.group(1) if m else ""}
    if mentioned or any(w in msg for w in ["recommend", "show me", "suggest", "want", "more"]):
        label = " + ".join(mentioned) if require_all else ", ".join(mentioned)
        return {"intent": "recommend", "include_genres": mentioned,
                "require_all": require_all,
                "reason": (f"Showing {label} films." if mentioned
                           else "Refreshing your recommendations.")}
    return {"intent": "question",
            "answer": "I can recommend by genre, filter genres out, or explain a pick."}


def _coerce_intent(raw, message):
    """Parse an LLM's JSON reply, then backfill compound-genre info from the
    rule-based synonym layer so 'rom com' -> require_all even if the model
    returned a single genre or missed the AND semantics."""
    try:
        parsed = json.loads(re.sub(r"```json|```", "", raw).strip())
    except Exception:
        return rule_based_intent(message)
    if not isinstance(parsed, dict) or "intent" not in parsed:
        return rule_based_intent(message)
    if parsed.get("intent") == "recommend":
        syn_genres, syn_all = extract_genres(message)
        if syn_all and syn_genres:
            # trust the synonym layer for compound phrases like "rom com"
            parsed["include_genres"] = syn_genres
            parsed["require_all"] = True
        else:
            parsed.setdefault("require_all", False)
    return parsed


def parse_intent(message, fair_list, fair_flags):
    """Anthropic -> Ollama -> rule-based, decided by BACKEND at startup.
    Any LLM failure falls through to the rule-based parser."""
    if BACKEND == "rule_based":
        return rule_based_intent(message)

    ctx = "\n".join(
        f"  {i}. {movies_indexed.loc[m,'title']} [{fair_flags[i-1] if i-1 < len(fair_flags) else '?'}]"
        for i, m in enumerate(fair_list[:TOP_K], 1) if m in movies_indexed.index
    ) or "  (none yet)"
    prompt = INTENT_PROMPT.format(genres=", ".join(KNOWN_GENRES),
                                  context=ctx, message=message)

    if BACKEND == "anthropic":
        raw = call_llm(prompt, LLM_CLIENT, model=ANTHROPIC_MODEL, max_tokens=400)
    else:  # ollama
        raw = call_ollama(prompt)

    if raw is None:
        return rule_based_intent(message)
    return _coerce_intent(raw, message)


def _rank_facts(uid, movie_idx, fair_list, flags):

    profile = infer_profile(uid, train_df, movies)
    # baseline order = score-sorted candidate pool (this IS relevance-only ranking)
    pool = candidates_for(uid)
    base_order = [m for m, _ in pool]
    scores = [s for _, s in pool]
    score_of = dict(pool)

    fair_rank = (fair_list.index(movie_idx) + 1) if movie_idx in fair_list else None
    baseline_rank = (base_order.index(movie_idx) + 1) if movie_idx in base_order else None
    flag = flags[fair_list.index(movie_idx)] if (fair_rank and len(flags) >= fair_rank) else "relevance"

    row = movies_indexed.loc[movie_idx] if movie_idx in movies_indexed.index else {}
    gender = str(row.get("director_gender", "unknown"))
    region = str(row.get("region", "unknown"))
    liked = set(profile["liked"])
    genre_match = sorted({g.strip() for g in str(row.get("genres", "")).split("|")} & liked)

    moved_up = (baseline_rank - fair_rank) if (baseline_rank and fair_rank) else None
    score = float(score_of.get(movie_idx, 0.0))

    # how strong is this relevance score relative to the whole candidate pool?
    n_pool = max(len(scores), 1)
    n_below = sum(1 for s in scores if s <= score)
    score_percentile = round(100.0 * n_below / n_pool, 1)

    # rank-discounted exposure this film receives (same discount as exposure_gap)
    exposure_weight = round(1.0 / np.log2((fair_rank or 1) + 1), 3)

    # this list's fairness composition, and this film's contribution to it
    n = max(len(fair_list), 1)
    fem_in_list = [m for m in fair_list if movie_gender.get(m) == "female"]
    nw_in_list = [m for m in fair_list if movie_region.get(m) == "non-western"]
    is_fem = gender == "female"
    is_nw = region == "non-western"
    prot_group = fem_in_list if is_fem else (nw_in_list if is_nw else [])
    protected_share = (round(100.0 / len(prot_group), 1) if prot_group else 0.0)

    return {
        "profile": profile,
        "title": str(row.get("title", f"Movie {movie_idx}")),
        "director": str(row.get("director", "Unknown")),
        "gender": gender, "region": region,
        "genres": str(row.get("genres", "")),
        "score": score,
        "score_percentile": score_percentile,
        "pool_size": n_pool,
        "exposure_weight": exposure_weight,
        "fair_rank": fair_rank,
        "baseline_rank": baseline_rank,
        "moved_up": moved_up,
        "flag": flag,
        "genre_match": genre_match,
        "is_protected": (is_fem or is_nw),
        "list_pct_female": round(len(fem_in_list) / n * 100, 1),
        "list_pct_nonwestern": round(len(nw_in_list) / n * 100, 1),
        "n_female_in_list": len(fem_in_list),
        "n_nonwestern_in_list": len(nw_in_list),
        "protected_share": protected_share,
    }


def _list_spd_facts(fair_list, base_list):

    g_base = list_spd(base_list, movie_gender, "female", "male")
    g_fair = list_spd(fair_list, movie_gender, "female", "male")
    r_base = list_spd(base_list, movie_region, "non-western", "western")
    r_fair = list_spd(fair_list, movie_region, "non-western", "western")
    return {"gender_spd_base": g_base, "gender_spd_fair": g_fair,
            "region_spd_base": r_base, "region_spd_fair": r_fair}


def _grounded_prompt(f, spd):

    axis = "gender (female-directed)" if f["gender"] == "female" else "regional (non-western)"

    if f["flag"] != "relevance" and f["moved_up"] and f["moved_up"] > 0:
        key = "gender" if f["flag"] == "gender" else "region"
        d = spd[f"{key}_spd_fair"] - spd[f"{key}_spd_base"]
        promo = (f"FORCE-PROMOTED by the FA*IR reranker to meet the {axis} quota "
                 f"(p={P_GENDER:g}). Moved up {f['moved_up']} place(s): rank "
                 f"{f['baseline_rank']} on relevance alone -> rank {f['fair_rank']} "
                 f"in the fair list. Effect on this list: {key.title()} SPD "
                 f"{spd[f'{key}_spd_base']:+.3f} -> {spd[f'{key}_spd_fair']:+.3f} "
                 f"({d:+.3f}, closer to 0 is fairer). At rank {f['fair_rank']} it "
                 f"receives exposure weight {f['exposure_weight']} (1/log2(rank+1)).")
    elif f["is_protected"]:
        grp = "female-directed" if f["gender"] == "female" else "non-western"
        n_grp = f["n_female_in_list"] if f["gender"] == "female" else f["n_nonwestern_in_list"]
        pct = f["list_pct_female"] if f["gender"] == "female" else f["list_pct_nonwestern"]
        promo = (f"NOT promoted — it earned rank {f['fair_rank']} on relevance alone "
                 f"(score {f['score']:.3f}, in the {f['score_percentile']:.0f}th "
                 f"percentile of the {f['pool_size']}-film candidate pool). It is "
                 f"{grp}, so it counts toward fairness organically: it is one of "
                 f"{n_grp} {grp} film(s) in this top-10 ({pct:.0f}% of the list), "
                 f"contributing {f['protected_share']:.0f}% of that protected group. "
                 f"At rank {f['fair_rank']} its exposure weight is "
                 f"{f['exposure_weight']} (1/log2(rank+1)).")
    else:
        promo = (f"Relevance pick at rank {f['fair_rank']} (rank {f['baseline_rank']} "
                 f"before reranking; the reranker left it essentially unchanged). "
                 f"Relevance score {f['score']:.3f} sits in the "
                 f"{f['score_percentile']:.0f}th percentile of the {f['pool_size']}-film "
                 f"pool; exposure weight at this rank is {f['exposure_weight']}.")

    gm = (", ".join(f["genre_match"]) if f["genre_match"]
          else "no direct overlap with their top genres")
    return (f"You are explaining a fairness-aware movie recommender's decision. Use "
            f"ONLY these facts; do NOT invent any numbers. Ground the explanation in "
            f"the metrics below, not genre alone.\n\n"
            f"User likes: {', '.join(f['profile']['liked'])} "
            f"(diversity appetite: {f['profile']['diversity']}).\n"
            f"Film: {f['title']} | genres: {f['genres']} | {f['director']} "
            f"({f['gender']}-directed) | {f['region']} | relevance score {f['score']:.3f}.\n"
            f"Genre overlap with user: {gm}.\n"
            f"Ranking & fairness: {promo}\n\n"
            f"In 2-4 sentences, explain to the user why this film appears where it does. "
            f"Cite the actual rank, the relevance score/percentile, and the fairness "
            f"numbers (SPD change if promoted, or its contribution to the list's "
            f"protected representation if not). Reference the exposure weight if it "
            f"helps. No preamble, no invented figures.")


def _grounded_rule_based(f, spd):

    lines = []
    gm = (", ".join(f["genre_match"]) if f["genre_match"] else None)
    basis = (f"matches your taste for {gm}" if gm
             else "scored highly on the learned preference model")
    lines.append(f"Rank {f['fair_rank']}: {f['title']} — relevance score "
                 f"{f['score']:.3f} ({f['score_percentile']:.0f}th percentile of the "
                 f"{f['pool_size']}-film candidate pool); {basis}.")

    if f["flag"] != "relevance" and f["moved_up"] and f["moved_up"] > 0:
        axis_word = "female-directed" if f["flag"] == "gender" else "non-western"
        key = "gender" if f["flag"] == "gender" else "region"
        d = spd[f"{key}_spd_fair"] - spd[f"{key}_spd_base"]
        lines.append(f"Promoted by FA*IR ({axis_word}) to meet the p={P_GENDER:g} "
                     f"diversity quota: moved up {f['moved_up']} place(s), from rank "
                     f"{f['baseline_rank']} (relevance only) to rank {f['fair_rank']}.")
        lines.append(f"Effect on this list: {key.title()} SPD {spd[f'{key}_spd_base']:+.3f} "
                     f"\u2192 {spd[f'{key}_spd_fair']:+.3f} ({d:+.3f}, closer to 0 is fairer); "
                     f"exposure weight at rank {f['fair_rank']} is {f['exposure_weight']}.")
    elif f["is_protected"]:
        axis_word = "female-directed" if f["gender"] == "female" else "non-western"
        n_grp = f["n_female_in_list"] if f["gender"] == "female" else f["n_nonwestern_in_list"]
        pct = f["list_pct_female"] if f["gender"] == "female" else f["list_pct_nonwestern"]
        lines.append(f"Not promoted — reached rank {f['fair_rank']} on relevance alone. "
                     f"As a {axis_word} film it aids fairness organically: 1 of {n_grp} "
                     f"{axis_word} title(s) in this top-10 ({pct:.0f}% of the list), "
                     f"exposure weight {f['exposure_weight']} at rank {f['fair_rank']}.")
    else:
        if f["baseline_rank"] == f["fair_rank"]:
            lines.append(f"Relevance pick; the reranker kept it at rank {f['fair_rank']} "
                         f"(unchanged from the relevance-only ordering); exposure weight "
                         f"{f['exposure_weight']}.")
        else:
            lines.append(f"Relevance pick; sits at rank {f['fair_rank']} "
                         f"(rank {f['baseline_rank']} before reranking); exposure weight "
                         f"{f['exposure_weight']}.")
    return lines


def explain_pick(uid, movie_idx, fair_list, flags, base_list=None):
    if movie_idx not in fair_list:
        prof = infer_profile(uid, train_df, movies)
        return "That film isn't in the current list.", [], prof

    f = _rank_facts(uid, movie_idx, fair_list, flags)
    if base_list is None:
        # baseline = relevance-only top-10 from the same pool
        base_list = [m for m, _ in candidates_for(uid)[:TOP_K]]
    spd = _list_spd_facts(fair_list, base_list)

    text = None
    if BACKEND == "anthropic":
        text = call_llm(_grounded_prompt(f, spd), LLM_CLIENT,
                        model=ANTHROPIC_MODEL, max_tokens=220)
    elif BACKEND == "ollama":
        text = call_ollama(_grounded_prompt(f, spd))

    if text:
        steps = [ln.strip() for ln in text.split("\n") if ln.strip()]
    else:
        steps = _grounded_rule_based(f, spd)   # deterministic, rank-consistent

    focus = steps[0] if steps else "No reasoning produced."
    return focus, steps, f["profile"]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ready": READY, "error": LOAD_ERROR, "n_users": n_users,
            "n_movies": n_movies, "p": P_GENDER, "alpha": FAIR_ALPHA,
            "backend": BACKEND,
            "llm": BACKEND != "rule_based",
            "ollama_model": OLLAMA_MODEL if BACKEND == "ollama" else None,
            "anthropic_model": ANTHROPIC_MODEL if BACKEND == "anthropic" else None}


@app.get("/random_user")
def random_user():
    if not READY:
        return {"error": LOAD_ERROR or "Model not loaded"}
    uid = random.randint(0, n_users - 1)
    bl, fl, ff = get_recs(uid)
    return json_response({
        "user_idx": uid,
        "stats": user_stats(uid),
        "fair_movies": [movie_to_dict(m, f) for m, f in zip(fl, ff)],
        "base_movies": [movie_to_dict(m, "relevance") for m in bl],
        "fair_list": [int(m) for m in fl],
        "fair_flags": ff,
        "metrics": compute_live_metrics(uid, fl, bl),
        "chart_b64": make_chart_b64(bl, fl),
    })


class ChatRequest(BaseModel):
    message: str
    user_idx: int
    history: List[dict] = []
    excluded_genres: List[str] = []
    include_genres: List[str] = []
    fair_list: List[int] = []
    fair_flags: List[str] = []


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    if not READY:
        return {"error": LOAD_ERROR or "Model not loaded"}

    uid, excl, incl = req.user_idx, list(req.excluded_genres), list(req.include_genres)
    parsed = parse_intent(req.message, req.fair_list, req.fair_flags)
    intent = parsed.get("intent", "question")

    if intent == "recommend":
        new_incl = parsed.get("include_genres", []) or incl
        require_all = bool(parsed.get("require_all", False))
        bl, fl, ff = get_recs(uid, excl, new_incl, require_all)
        if not fl and require_all and len(new_incl) > 1:
            # compound genre too scarce in this pool — relax AND to OR
            bl, fl, ff = get_recs(uid, excl, new_incl, require_all=False)
            if fl:
                incl = new_incl
                reply = (f"No films tagged {' + '.join(new_incl)} together in this "
                         f"user's pool, so here are {', '.join(new_incl)} films.")
            else:
                reply = f"No {', '.join(new_incl)} films in this user's candidate pool."
                bl, fl, ff = get_recs(uid, excl, incl)
        elif not fl:
            reply = f"No {', '.join(new_incl)} films in this user's candidate pool. Try another genre."
            bl, fl, ff = get_recs(uid, excl, incl)
        else:
            incl = new_incl
            reply = parsed.get("reason", "Here are your picks.")
    elif intent == "filter":
        excl = sorted(set(excl + parsed.get("exclude_genres", [])))
        bl, fl, ff = get_recs(uid, excl, incl)
        reply = parsed.get("reason", "Filter applied.") + f" Excluding: {', '.join(excl)}."
    elif intent == "explain":
        bl, fl, ff = get_recs(uid, excl, incl)
        tq = (parsed.get("movie_title") or "").lower()
        matched = next((m for m in fl if m in movies_indexed.index and tq
                        and tq in str(movies_indexed.loc[m, "title"]).lower()), None)
        if matched is None:
            reply = "I couldn't match that title to the current list. Name one of the shown films."
        else:
            reply, _, _ = explain_pick(uid, matched, fl, ff, base_list=bl)
    elif intent == "reset":
        excl, incl = [], []
        bl, fl, ff = get_recs(uid, excl, incl)
        reply = "Filters cleared."
    else:
        bl, fl, ff = get_recs(uid, excl, incl)
        reply = parsed.get("answer", "I can recommend, filter, or explain a pick.")

    # Every chat turn reports the fairness metrics of the list it just returned.
    # A genre-filtered list invalidates accuracy metrics (see compute_live_metrics).
    metrics = compute_live_metrics(uid, fl, bl, genre_filtered=bool(incl))
    reply += "\n\n" + format_metrics_summary(metrics)

    return json_response({
        "reply": reply,
        "intent": intent,
        "fair_movies": [movie_to_dict(m, f) for m, f in zip(fl, ff)],
        "base_movies": [movie_to_dict(m, "relevance") for m in bl],
        "fair_list": [int(m) for m in fl],
        "fair_flags": ff,
        "excluded_genres": excl,
        "include_genres": incl,
        "metrics": metrics,
        "chart_b64": make_chart_b64(bl, fl),
    })


class CotRequest(BaseModel):
    user_idx: int
    movie_idx: int
    fair_list: List[int] = []
    fair_flags: List[str] = []
    excluded_genres: List[str] = []
    include_genres: List[str] = []


@app.post("/cot_explain")
def cot_explain(req: CotRequest):
    if not READY:
        return {"error": LOAD_ERROR or "Model not loaded"}

    fl, ff = list(req.fair_list), list(req.fair_flags)
    # Recompute the canonical list+flags+baseline for this user/filter so the
    # explanation is internally consistent even if the client sent partial state.
    bl, canon_fl, canon_ff = get_recs(req.user_idx, req.excluded_genres, req.include_genres)
    if not fl:
        fl, ff = canon_fl, canon_ff
    if not ff or len(ff) != len(fl):
        # derive flags for the client-supplied order from the canonical mapping
        flag_of = {m: canon_ff[i] for i, m in enumerate(canon_fl)} if canon_fl else {}
        ff = [flag_of.get(m, "relevance") for m in fl]

    focus, steps, profile = explain_pick(req.user_idx, req.movie_idx, fl, ff, base_list=bl)
    return json_response({
        "movie_idx": req.movie_idx,
        "focus": focus,
        "steps": steps,
        "profile": profile,
        "backend": BACKEND,
    })


@app.get("/reference_metrics")
def reference_metrics():
    """Offline aggregate numbers exactly as written by fair_rerank.py."""
    if REFERENCE is None:
        return {"error": "fair_results.json not found"}
    return json_response(REFERENCE)


@app.get("/metrics_table")
def metrics_table(user_idx: Optional[int] = None, n_users_eval: int = 0):
    """The full 15-metric results_table.txt, recomputed live.

    - user_idx given      -> table for that one user's list (cohort columns
                             still span every list served this session)
    - n_users_eval > 0    -> run the pipeline over N random users and report
                             the cohort table, i.e. the offline eval in
                             miniature. This is how you reproduce the shape of
                             results_table.txt without rerunning fair_rerank.py.
    """
    if not READY:
        return {"error": LOAD_ERROR or "Model not loaded"}

    if n_users_eval > 0:
        uids = random.sample(range(n_users), min(n_users_eval, n_users))
        fair_recs, base_recs = {}, {}
        for u in uids:
            bl, fl, _ = get_recs(u)
            if fl:
                fair_recs[u] = fl
            if bl:
                base_recs[u] = bl
        gt = test_df[test_df["user_idx"].isin(fair_recs.keys())] if test_df is not None else None

        def agg(recs):
            if gt is None or not recs:
                return {}
            prec, rec, ndcg = precision_recall_ndcg(recs, gt)
            spd_g, eod_g = compute_fairness_metrics(recs, gt, movies, "director_gender", "female", "male")
            spd_r, eod_r = compute_fairness_metrics(recs, gt, movies, "region", "non-western", "western")
            eg = evaluate_extended(recs, movies, "director_gender", "female", n_movies, k=TOP_K)
            er = evaluate_extended(recs, movies, "region", "non-western", n_movies, k=TOP_K)
            return {
                "ndcg_at_10": _r(ndcg), "precision_at_10": _r(prec), "recall_at_10": _r(rec),
                "gender_spd": _r(spd_g), "gender_eod": _r(eod_g),
                "region_spd": _r(spd_r), "region_eod": _r(eod_r),
                "gender_rND": _r(eg["rND"]), "gender_exposure_gap": _r(eg["exposure_gap"]),
                "gender_collapse_rate": _r(eg["collapse_rate"]),
                "region_rND": _r(er["rND"]), "region_exposure_gap": _r(er["exposure_gap"]),
                "region_collapse_rate": _r(er["collapse_rate"]),
                "gini_exposure": _r(eg["gini_exposure"]),
                "catalog_coverage": _r(eg["catalog_coverage"]),
            }

        f_agg, b_agg = agg(fair_recs), agg(base_recs)
        rows = []
        for label, key, _scope in TABLE_ROWS:
            fv, bv = f_agg.get(key), b_agg.get(key)
            rows.append({"metric": label, "no_rerank": bv, "fair": fv,
                         "delta": None if (fv is None or bv is None) else round(fv - bv, 4),
                         "scope": "cohort"})
        return json_response({
            "mode": "cohort_eval", "n_users": len(fair_recs),
            "p": P_GENDER, "alpha": FAIR_ALPHA, "table": rows,
            "reference": REFERENCE,
        })

    uid = random.randint(0, n_users - 1) if user_idx is None else user_idx
    bl, fl, _ = get_recs(uid)
    m = compute_live_metrics(uid, fl, bl)
    return json_response({"mode": "single_user", "user_idx": uid,
                          "p": P_GENDER, "alpha": FAIR_ALPHA,
                          "table": m["table"], "targets": m["targets"],
                          "notes": m["notes"], "reference": REFERENCE})


def warmup_accuracy_cohort(n_sample=None):
    if not READY:
        return
    n_sample = ACCURACY_WARMUP_USERS if n_sample is None else n_sample
    if n_sample <= 0:
        print("[warmup] disabled (ACCURACY_WARMUP_USERS=0)")
        return
    n_sample = min(n_sample, n_users)
    print(f"[warmup] seeding accuracy cohort over {n_sample} users...")
    import time
    t0 = time.time()

    _tracker_snapshot = dict(EXPOSURE_TRACKER.counts)
    _cache_keys_before = set(CANDIDATE_CACHE.keys())

    uids = random.sample(range(n_users), n_sample)
    done = 0
    for uid in uids:
        try:
            bl, fl, _ = get_recs(uid)
            if fl:
                SEED_COHORT["fair"][uid] = list(fl)
            if bl:
                SEED_COHORT["base"][uid] = list(bl)
            done += 1
        except Exception as e:
            print(f"[warmup] user {uid} failed: {e}")

    # restore tracker + drop cache entries created during warm-up
    EXPOSURE_TRACKER.counts = _tracker_snapshot
    for k in list(CANDIDATE_CACHE.keys()):
        if k not in _cache_keys_before:
            del CANDIDATE_CACHE[k]

    # report the seeded accuracy so it's visible in logs
    co = cohort_metrics("fair")
    print(f"[warmup] seeded {done} users in {time.time()-t0:.1f}s | "
          f"NDCG@10={co['ndcg_at_10']} Precision@10={co['precision_at_10']} "
          f"Recall@10={co['recall_at_10']} (scored {co['n_scored_users']})")


# Run warm-up once at import, after all functions/endpoints are defined.
warmup_accuracy_cohort()


if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")