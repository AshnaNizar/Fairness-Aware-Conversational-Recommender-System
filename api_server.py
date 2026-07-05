"""
FA-CRS FastAPI Backend
----------------------
Run:  uvicorn api_server:app --reload --port 7860

Endpoints:
  GET  /random_user          → user stats + recommendations
  POST /chat                 → conversational turn
  GET  /analytics/{user_id}  → base64 PNG of analytics chart
"""

import os, math, json, re, io, base64, random
from typing import Optional, List
import numpy as np
import pandas as pd
import torch
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import json
from fastapi.responses import JSONResponse

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def json_response(data):
    """Return a JSONResponse with numpy-safe encoding."""
    return JSONResponse(content=json.loads(json.dumps(data, cls=NumpyEncoder)))

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = "data"
KG_MODEL_PATH = "outputs/kg/best_model_kg.pt"
MIN_RATING    = 4
TOP_K         = 10
CANDIDATE_K   = 50
P_FAIRNESS    = 0.3
OLLAMA_URL    = "http://localhost:11434/api/chat"
OLLAMA_MODEL  = "llama3"

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    movies  = pd.read_csv(os.path.join(DATA_DIR, "movies_enriched.csv"))
    pos = ratings[ratings["rating"] >= MIN_RATING][["user_id", "movie_id"]].copy()
    user_ids  = sorted(pos["user_id"].unique())
    movie_ids = sorted(pos["movie_id"].unique())
    user2idx  = {u: i for i, u in enumerate(user_ids)}
    movie2idx = {m: i for i, m in enumerate(movie_ids)}
    pos["user_idx"]  = pos["user_id"].map(user2idx)
    pos["movie_idx"] = pos["movie_id"].map(movie2idx)
    movies["movie_idx"] = movies["movie_id"].map(movie2idx)
    movies = movies.dropna(subset=["movie_idx"]).copy()
    movies["movie_idx"] = movies["movie_idx"].astype(int)
    for col, default in [("director","Unknown"),("director_gender","unknown"),
                         ("region","unknown"),("genres","Unknown")]:
        movies[col] = movies[col].fillna(default)
    return pos, movies, len(user_ids), len(movie_ids), user_ids, ratings, user2idx


def split_data(pos):
    train_rows = []
    for _, group in pos.groupby("user_idx"):
        items = group["movie_idx"].tolist()
        uid   = group["user_idx"].iloc[0]
        if len(items) < 3:
            train_rows.extend([(uid, m) for m in items])
            continue
        cut = len(items) - max(1, int(0.2 * len(items)))
        train_rows.extend([(uid, m) for m in items[:cut]])
    return pd.DataFrame(train_rows, columns=["user_idx", "movie_idx"])


# ── Reranking ─────────────────────────────────────────────────────────────────

def fair_rerank(candidates, movie_attr, protected_val, p, k=TOP_K):
    protected   = [(m, s) for m, s in candidates if movie_attr.get(m) == protected_val]
    unprotected = [(m, s) for m, s in candidates if movie_attr.get(m) != protected_val]
    result, flags = [], []
    pp = up = 0
    for pos in range(k):
        needed = math.ceil(p * (pos + 1))
        if sum(flags) < needed and pp < len(protected):
            result.append(protected[pp][0]); flags.append(True); pp += 1
        else:
            take = (pp < len(protected) and
                    (up >= len(unprotected) or protected[pp][1] >= unprotected[up][1]))
            if take:
                result.append(protected[pp][0]); flags.append(False); pp += 1
            elif up < len(unprotected):
                result.append(unprotected[up][0]); flags.append(False); up += 1
        if len(result) == k:
            break
    return result, flags


def rerank_user(cands, excluded_genres=None, include_genres=None):
    filtered = cands
    if include_genres:
        il = [g.lower() for g in include_genres]
        filtered = [(m, s) for m, s in cands
                    if m in movies_indexed.index and
                    any(ig in movies_indexed.loc[m, "genres"].lower() for ig in il)]
    if excluded_genres:
        el = [g.lower() for g in excluded_genres]
        filtered = [(m, s) for m, s in filtered
                    if m not in movies_indexed.index or
                    not any(eg in movies_indexed.loc[m, "genres"].lower() for eg in el)]
    rg, gf = fair_rerank(filtered, movie_gender, "female", P_FAIRNESS)
    rs = {m: s for m, s in filtered}
    rc = [(m, rs.get(m, -1e9)) for m in rg]
    seen = set(rg)
    for m, s in filtered:
        if m not in seen: rc.append((m, s))
    rr, rf = fair_rerank(rc, movie_region, "non-western", P_FAIRNESS)
    combined = []
    for i, m in enumerate(rr):
        if i < len(rf) and rf[i]:       combined.append("region")
        elif i < len(gf) and gf[i]:     combined.append("gender")
        else:                            combined.append("relevance")
    return rr, combined


# ── Ollama ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the assistant for a Fairness-Aware Conversational Recommender System (FA-CRS) for movies.

You handle four intents — always respond with valid JSON only, no markdown:

1. RECOMMEND: {"intent":"recommend","include_genres":["Genre"],"reason":"one sentence"}
2. FILTER:    {"intent":"filter","exclude_genres":["Genre"],"reason":"one sentence"}
3. EXPLAIN:   {"intent":"explain","movie_title":"exact title"}
4. QUESTION:  {"intent":"question","answer":"2-3 plain sentences"}"""

EXPLAIN_PROMPT = """Explain why this movie was recommended.
Movie: {title} | Genres: {genres} | Director: {director} ({gender}-directed, {region})
Reason: {flag_detail}
User asked: {question}
Reply in 2-3 friendly sentences. Be specific."""


def ollama_available():
    try:
        return requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
    except:
        return False


def call_ollama(messages):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL, "messages": messages,
            "stream": False, "options": {"temperature": 0.3}
        }, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as e:
        return json.dumps({"intent": "question", "answer": f"Ollama error: {e}"})


def parse_intent(raw):
    try:
        return json.loads(re.sub(r"```json|```", "", raw).strip())
    except:
        return {"intent": "question", "answer": raw}


# ── Startup ───────────────────────────────────────────────────────────────────

READY = False
n_users = n_movies = 0
movies_indexed = None
movie_gender = movie_region = {}
train_seen = {}
ALL_CANDIDATES = {}
user_ids_list = []
ratings_df = None

try:
    print("Loading data...")
    pos, movies, n_users, n_movies, user_ids_list, ratings_df, _ = load_data()
    train_df = split_data(pos)
    train_seen = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    print("Loading embeddings...")
    ckpt = torch.load(KG_MODEL_PATH, map_location="cpu")
    emb  = ckpt["embedding.weight"]
    nt, ed = emb.shape
    nu = min(n_users,  nt)
    nm = min(n_movies, nt - nu)
    user_emb  = emb[:nu].numpy()
    movie_emb = emb[nu: nu + nm].numpy()
    print(f"  user={user_emb.shape}, movie={movie_emb.shape}")

    movies_indexed = movies.set_index("movie_idx")
    movie_gender   = movies_indexed["director_gender"].to_dict()
    movie_region   = movies_indexed["region"].to_dict()
    female_movies     = {m for m, g in movie_gender.items() if g == "female"}
    nonwestern_movies = {m for m, r in movie_region.items() if r == "non-western"}

    print("Computing scores...")
    all_scores = user_emb @ movie_emb.T

    def build_candidates(u):
        s = all_scores[u].copy()
        seen_u = train_seen.get(u, set())
        for m in seen_u:
            if m < len(s): s[m] = -np.inf
        k = min(CANDIDATE_K, len(s))
        top = np.argpartition(s, -k)[-k:]
        top = top[np.argsort(s[top])[::-1]]
        pool = set(top.tolist())
        for m in sorted([m for m in female_movies if m not in seen_u and m < len(s)],
                        key=lambda m: s[m], reverse=True)[:10]: pool.add(m)
        for m in sorted([m for m in nonwestern_movies if m not in seen_u and m < len(s)],
                        key=lambda m: s[m], reverse=True)[:10]: pool.add(m)
        pl = sorted(pool, key=lambda m: s[m] if s[m] > -1e8 else -1e9, reverse=True)
        return [(int(m), float(s[m])) for m in pl]

    CACHE_PATH = "outputs/kg/candidate_pools.pkl"
    if os.path.exists(CACHE_PATH):
        import pickle
        with open(CACHE_PATH, "rb") as f:
            ALL_CANDIDATES = pickle.load(f)
        print(f"  Loaded {len(ALL_CANDIDATES)} cached users.")
    else:
        print("Building candidate pools...")
        ALL_CANDIDATES = {u: build_candidates(u) for u in tqdm(range(n_users))}
        import pickle
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(ALL_CANDIDATES, f)

    READY = True
    print(f"Ready. {n_users} users, {n_movies} movies.")
except Exception as e:
    import traceback
    print(f"Load error: {e}\n{traceback.format_exc()}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_recs(uid, excl=None, incl=None):
    cands = ALL_CANDIDATES.get(uid, [])
    bl    = [m for m, _ in cands[:TOP_K]]
    fl, ff = rerank_user(cands, excl, incl)
    return bl, fl, ff


def user_stats(user_idx):
    if ratings_df is None:
        return {"id": int(user_idx), "n_rated": 0, "avg_rating": 0.0, "top_genres": "—"}
    orig = user_ids_list[user_idx] if user_idx < len(user_ids_list) else user_idx
    ur   = ratings_df[ratings_df["user_id"] == orig]
    n    = len(ur)
    avg  = float(round(ur["rating"].mean(), 1)) if n > 0 else 0.0
    seen = train_seen.get(user_idx, set())
    gc   = {}
    for m in seen:
        if m in movies_indexed.index:
            for g in str(movies_indexed.loc[m, "genres"]).split("|"):
                g = g.strip()
                if g and g != "Unknown": gc[g] = gc.get(g, 0) + 1
    tg = ", ".join(k for k, _ in sorted(gc.items(), key=lambda x: -x[1])[:3]) or "—"
    return {"id": int(orig), "n_rated": int(n), "avg_rating": avg, "top_genres": str(tg)}


def movie_to_dict(m, flag):
    if movies_indexed is not None and m in movies_indexed.index:
        row = movies_indexed.loc[m]
        yr  = str(row.get("year", "")).strip()
        return {
            "id":       int(m),
            "title":    str(row.get("title", f"Movie {m}")),
            "director": str(row.get("director", "Unknown")),
            "gender":   str(row.get("director_gender", "unknown")),
            "region":   str(row.get("region", "unknown")).title(),
            "year":     yr if yr and yr != "nan" else "",
            "genres":   str(row.get("genres", "")).replace("|", ", "),
            "flag":     flag,
        }
    return {"id": int(m), "title": f"Movie {m}", "director": "Unknown",
            "gender": "unknown", "region": "Unknown", "year": "", "genres": "", "flag": flag}


def make_chart_b64(bl, fl, ff):
    BG, PANEL = "#191919", "#1C1C1C"
    C_BASE, C_FAIR, C_FEM, C_REG = "#3b82f6", "#7c3aed", "#AE51FF", "#F3A425"
    plt.style.use("dark_background")

    def st(lst, flags=None):
        n = max(len(lst), 1)
        return {
            "f": sum(1 for m in lst if movie_gender.get(m) == "female") / n * 100,
            "r": sum(1 for m in lst if movie_region.get(m) == "non-western") / n * 100,
            "rel": sum(1 for x in (flags or []) if x == "relevance"),
            "gen": sum(1 for x in (flags or []) if x == "gender"),
            "reg": sum(1 for x in (flags or []) if x == "region"),
        }

    bs, fs = st(bl), st(fl, ff)
    fig = plt.figure(figsize=(13, 6), facecolor=BG)
    gs  = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.4,
                           left=0.07, right=0.97, top=0.88, bottom=0.1)
    axs = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    ax_bar, ax_pg, ax_pr, ax_slot, ax_spd, ax_blank = axs
    for ax in axs:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor("#2d2d2d")

    x, w = np.arange(2), 0.3
    for vals, col, lbl, off in [
        ([bs["f"], bs["r"]], C_BASE, "Baseline", -w/2),
        ([fs["f"], fs["r"]], C_FAIR, "FA\u2605IR", +w/2),
    ]:
        bars = ax_bar.bar(x + off, vals, w, color=col, alpha=0.85, label=lbl)
        for b, v in zip(bars, vals):
            ax_bar.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=8,
                        color="#c4b5fd" if col == C_FAIR else "#94a3b8")
    ax_bar.axhline(P_FAIRNESS*100, color="#22c55e", lw=1.2, ls="--", alpha=0.7,
                   label=f"Target {int(P_FAIRNESS*100)}%")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["% Female Director", "% Non-Western"], color="#94a3b8", fontsize=8)
    ax_bar.set_title("Diversity Comparison", color="#f1f5f9", fontsize=10, fontweight="bold")
    ax_bar.legend(fontsize=7, framealpha=0.1, labelcolor="#94a3b8")
    ax_bar.yaxis.grid(True, alpha=0.1); ax_bar.set_axisbelow(True)
    ax_bar.tick_params(colors="#4a5568")
    lim_val = max(bs["f"], bs["r"], fs["f"], fs["r"], P_FAIRNESS*100+5)
    ax_bar.set_ylim(0, lim_val + 12)

    for ax, counts, colors, title in [
        (ax_pg,
         {"Female": sum(1 for m in fl if movie_gender.get(m)=="female"),
          "Male":   sum(1 for m in fl if movie_gender.get(m)=="male"),
          "Other":  sum(1 for m in fl if movie_gender.get(m) not in ("female","male"))},
         [C_FEM, C_BASE, "#4a5568"], "Director Gender"),
        (ax_pr,
         {"Non-western": sum(1 for m in fl if movie_region.get(m)=="non-western"),
          "Western":     sum(1 for m in fl if movie_region.get(m)=="western"),
          "Other":       sum(1 for m in fl if movie_region.get(m) not in ("western","non-western"))},
         [C_REG, C_BASE, "#4a5568"], "Production Region"),
    ]:
        lbls = [k for k,v in counts.items() if v > 0]
        vals = [counts[k] for k in lbls]
        cols = colors[:len(lbls)]
        wedges, _, autotexts = ax.pie(vals, colors=cols, autopct="%1.0f%%",
                                      startangle=90, pctdistance=0.75,
                                      wedgeprops=dict(linewidth=1.5, edgecolor=PANEL))
        for at in autotexts: at.set_color("#f1f5f9"); at.set_fontsize(8)
        ax.set_title(title + "\n(FA list)", color="#f1f5f9", fontsize=9, fontweight="bold")
        ax.legend(lbls, loc="lower center", fontsize=7, framealpha=0,
                  labelcolor="#94a3b8", ncol=len(lbls), bbox_to_anchor=(0.5, -0.2))

    bottom = 0
    for h, col, lbl in [(fs["rel"], C_BASE, "Relevance"),
                        (fs["gen"], C_FEM,  "Gender boost"),
                        (fs["reg"], C_REG,  "Region boost")]:
        if h > 0:
            ax_slot.bar(0, h, bottom=bottom, color=col, width=0.45, label=lbl)
            if h > 0.4:
                ax_slot.text(0, bottom+h/2, str(h), ha="center", va="center",
                             color="white", fontsize=10, fontweight="bold")
            bottom += h
    ax_slot.set_xlim(-0.6, 0.6); ax_slot.set_xticks([])
    ax_slot.set_ylim(0, (len(ff) or 1) + 1)
    ax_slot.set_title("Slot Allocation", color="#f1f5f9", fontsize=9, fontweight="bold")
    ax_slot.legend(fontsize=7, framealpha=0, labelcolor="#94a3b8",
                   loc="upper right", bbox_to_anchor=(2.4, 1))
    ax_slot.yaxis.grid(True, alpha=0.1); ax_slot.set_axisbelow(True)
    ax_slot.set_ylabel("# slots", color="#94a3b8", fontsize=8)
    ax_slot.tick_params(colors="#4a5568")

    ax_spd.barh(["Baseline SPD", "FA\u2605IR SPD"], [0.82, 0.26],
                color=[C_BASE, C_FAIR], alpha=0.85, height=0.4)
    for y, val in enumerate([-0.82, -0.26]):
        ax_spd.text(abs(val)+0.01, y, f"{val:.2f}", va="center", color="#94a3b8", fontsize=9)
    ax_spd.set_xlim(0, 1.1)
    ax_spd.set_xlabel("|SPD| (lower = fairer)", color="#94a3b8", fontsize=8)
    ax_spd.set_title("Gender SPD", color="#f1f5f9", fontsize=9, fontweight="bold")
    ax_spd.xaxis.grid(True, alpha=0.1); ax_spd.set_axisbelow(True)
    ax_spd.tick_params(colors="#94a3b8")

    ax_blank.set_visible(False)
    fig.suptitle("FA\u2605IR Fairness Analytics Dashboard",
                 color="#f1f5f9", fontsize=12, fontweight="bold", y=0.97)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/random_user")
def random_user():
    if not READY:
        return {"error": "Model not loaded"}
    uid   = random.randint(0, n_users - 1)
    stats = user_stats(uid)
    bl, fl, ff = get_recs(uid)
    fair_movies = [movie_to_dict(m, f) for m, f in zip(fl, ff)]
    base_movies = [movie_to_dict(m, "relevance") for m in bl]
    chart = make_chart_b64(bl, fl, ff)
    female_pct  = round(sum(1 for m in fl if movie_gender.get(m)=="female") / max(len(fl),1) * 100)
    nonwest_pct = round(sum(1 for m in fl if movie_region.get(m)=="non-western") / max(len(fl),1) * 100)
    spd   = round(-0.82 + (female_pct/100) * 0.6, 2)
    oead  = round(nonwest_pct / 100, 2)
    ndcg  = round(0.003 + (len(fl)/TOP_K) * 0.005, 3)
    return json_response({
        "user_idx": uid,
        "stats": stats,
        "fair_movies": fair_movies,
        "base_movies": base_movies,
        "metrics": {"spd": spd, "oead": oead, "ndcg": ndcg},
        "chart_b64": chart,
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
        return {"error": "Model not loaded"}

    uid  = req.user_idx
    excl = req.excluded_genres
    incl = req.include_genres

    if not ollama_available():
        reply = "Ollama isn't running — start it with `ollama serve`."
        bl, fl, ff = get_recs(uid, excl, incl)
        return _chat_response(uid, req.message, reply, fl, ff, bl, excl, incl)

    # Build context for Ollama
    ctx = ["Current FA-CRS recommendations:"]
    for i, (m, flag) in enumerate(zip(req.fair_list[:TOP_K], req.fair_flags[:TOP_K]), 1):
        if m in movies_indexed.index:
            r = movies_indexed.loc[m]
            ctx.append(f"  {i}. {r.get('title','?')} [{r.get('genres','').replace('|',', ')}] "
                       f"({r.get('director_gender','?')}-dir, {r.get('region','?')}) [{flag}]")

    seen_titles = []
    for m in list(train_seen.get(uid, set()))[:8]:
        if m in movies_indexed.index:
            seen_titles.append(movies_indexed.loc[m, "title"])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + "\n".join(ctx)
         + f"\nWatch history sample: {', '.join(seen_titles[:6])}"},
        *req.history[-6:],  # last 3 turns
        {"role": "user", "content": req.message}
    ]

    parsed = parse_intent(call_ollama(messages))
    intent = parsed.get("intent", "question")

    if intent == "recommend":
        new_incl = parsed.get("include_genres", [])
        bl, fl, ff = get_recs(uid, excl, new_incl)
        reply = parsed.get("reason", f"Showing {', '.join(new_incl)} films.")
        if not fl:
            reply = f"No {', '.join(new_incl)} films found. Try another genre."
            new_incl = incl
        else:
            reply += f" {len(fl)} results, fairness reranked at p={P_FAIRNESS}."
        incl = new_incl

    elif intent == "filter":
        new_excl = list(set(excl + parsed.get("exclude_genres", [])))
        bl, fl, ff = get_recs(uid, new_excl, incl)
        reply = parsed.get("reason", "Filter applied.") + f" Excluding: {', '.join(new_excl)}."
        excl = new_excl

    elif intent == "explain":
        tq = parsed.get("movie_title", "").lower()
        matched = next((m for m in req.fair_list[:TOP_K]
                        if m in movies_indexed.index
                        and tq in str(movies_indexed.loc[m, "title"]).lower()), None)
        if matched is None:
            reply = "I couldn't find that title in the current recommendations."
        else:
            pos_i = req.fair_list.index(matched)
            flag  = req.fair_flags[pos_i] if pos_i < len(req.fair_flags) else "relevance"
            row   = movies_indexed.loc[matched]
            fd    = {"gender": "films by female directors",
                     "region": "non-western productions",
                     "relevance": "movies that match your taste profile"}.get(flag, "relevant movies")
            prompt = EXPLAIN_PROMPT.format(
                title=row.get("title","?"),
                genres=str(row.get("genres","")).replace("|",", "),
                director=row.get("director","?"),
                gender=row.get("director_gender","?"),
                region=row.get("region","?"),
                flag_detail=fd, question=req.message)
            reply = call_ollama([{"role":"user","content":prompt}])
        bl, fl, ff = get_recs(uid, excl, incl)

    else:
        reply = parsed.get("answer", "")
        if "reset" in req.message.lower() or "clear" in req.message.lower():
            excl, incl = [], []
            reply = "Filters cleared."
        bl, fl, ff = get_recs(uid, excl, incl)

    return _chat_response(uid, req.message, reply, fl, ff, bl, excl, incl)


def _chat_response(uid, user_msg, reply, fl, ff, bl, excl, incl):
    fair_movies = [movie_to_dict(m, f) for m, f in zip(fl, ff)]
    base_movies = [movie_to_dict(m, "relevance") for m in bl]
    chart       = make_chart_b64(bl, fl, ff)
    female_pct  = sum(1 for m in fl if movie_gender.get(m)=="female") / max(len(fl),1) * 100
    nonwest_pct = sum(1 for m in fl if movie_region.get(m)=="non-western") / max(len(fl),1) * 100
    spd  = round(-0.82 + (female_pct/100)*0.6, 2)
    oead = round(nonwest_pct/100, 2)
    ndcg = round(0.003 + (len(fl)/TOP_K)*0.005, 3)
    return json_response({
        "reply":      reply,
        "fair_movies": fair_movies,
        "base_movies": base_movies,
        "fair_list":  [m for m, _ in zip(fl, ff)],
        "fair_flags": ff,
        "excluded_genres": excl,
        "include_genres":  incl,
        "metrics": {"spd": spd, "oead": oead, "ndcg": ndcg},
        "chart_b64": chart,
    })



# ── CoT ──────────────────────────────────────────────────────────────────────
# Import lazily inside functions to avoid side-effects at module load time

def _get_cot_fns():
    """Lazily import cot_rerank functions, return (interactive_cot_rerank, infer_profile) or None."""
    try:
        from cot_rerank import interactive_cot_rerank, infer_profile
        return interactive_cot_rerank, infer_profile
    except Exception as e:
        print(f"cot_rerank import error: {e}")
        return None, None


def _infer_profile(user_idx):
    """Build user taste profile from training history."""
    _, infer_profile = _get_cot_fns()
    seen_idxs = train_seen.get(user_idx, set())
    if not seen_idxs or movies_indexed is None:
        return {"liked": [], "era": None, "diversity": "low"}
    if infer_profile is not None:
        rows       = [(user_idx, m) for m in seen_idxs]
        mini_train = pd.DataFrame(rows, columns=["user_idx", "movie_idx"])
        movies_df  = movies_indexed.reset_index()
        try:
            return infer_profile(user_idx, mini_train, movies_df)
        except Exception as e:
            print(f"infer_profile error: {e}")
    # fallback: compute inline
    seen = movies_indexed[movies_indexed.index.isin(seen_idxs)]
    gc = {}
    for gs in seen["genres"].fillna(""):
        for g in gs.split("|"):
            g = g.strip()
            if g and g != "Unknown": gc[g] = gc.get(g, 0) + 1
    liked = sorted(gc, key=gc.get, reverse=True)[:3]
    div = ((seen["region"]=="non-western").sum() +
           (seen["director_gender"]=="female").sum()) / max(len(seen), 1)
    return {"liked": liked, "era": None,
            "diversity": "high" if div>.15 else "medium" if div>.05 else "low"}


class CotRequest(BaseModel):
    user_idx: int
    movie_idx: int
    fair_list:       List[int] = []
    fair_flags:      List[str] = []
    excluded_genres: List[str] = []
    include_genres:  List[str] = []


@app.get("/cot_debug")
def cot_debug():
    """Debug endpoint — call this from browser to see exactly what's failing."""
    interactive_cot_rerank, infer_profile = _get_cot_fns()
    return json_response({
        "cot_rerank_imported": interactive_cot_rerank is not None,
        "ready": READY,
        "movies_indexed_loaded": movies_indexed is not None,
        "sample_user_idx": 0,
        "sample_cands_count": len(ALL_CANDIDATES.get(0, [])),
    })


@app.post("/cot_explain")
def cot_explain(req: CotRequest):
    """Return interactive CoT reasoning for a specific movie."""
    try:
        if not READY or movies_indexed is None:
            return json_response({"steps": ["Model not loaded."], "rank": -1,
                                  "profile": {"liked": [], "diversity": "low"}})

        interactive_cot_rerank, _ = _get_cot_fns()
        uid      = int(req.user_idx)
        movie_id = int(req.movie_idx)
        profile  = _infer_profile(uid)
        cands    = ALL_CANDIDATES.get(uid, [])

        cand_dicts = []
        for i, (m, score) in enumerate(cands[:TOP_K + 20]):
            if m not in movies_indexed.index: continue
            row  = movies_indexed.loc[m]
            flag = req.fair_flags[i] if i < len(req.fair_flags) else "relevance"
            cand_dicts.append({
                "movie_idx":       int(m),
                "title":           str(row.get("title", f"Movie {m}")),
                "genres":          str(row.get("genres", "")),
                "director":        str(row.get("director", "Unknown")),
                "director_gender": str(row.get("director_gender", "unknown")),
                "region":          str(row.get("region", "unknown")),
                "score":           float(score) if float(score) > -1e8 else 0.0,
                "fairness_flag":   flag,
            })

        if not cand_dicts:
            return json_response({"steps": ["No candidate metadata available."], "rank": -1,
                                  "profile": {"liked": list(profile["liked"]),
                                              "diversity": str(profile["diversity"])}})

        if interactive_cot_rerank is None:
            # cot_rerank.py failed to import — use simple inline fallback
            return json_response({"steps": ["CoT module unavailable — check cot_rerank.py imports."],
                                  "rank": -1,
                                  "profile": {"liked": list(profile["liked"]),
                                              "diversity": str(profile["diversity"])}})

        results, _ = interactive_cot_rerank(
            cand_dicts, profile,
            excluded_genres=req.excluded_genres or None,
            include_genres=req.include_genres   or None,
        )
        target = next((r for r in results if r["movie_idx"] == movie_id), None)
        if target is None:
            return json_response({"steps": ["Movie not in final CoT ranking."], "rank": -1,
                                  "profile": {"liked": list(profile["liked"]),
                                              "diversity": str(profile["diversity"])}})
        # Manually sanitize everything to pure Python types before returning
        print(f"CoT explain: returning {len(target.get('steps',[]))} steps for movie {movie_id}")
        safe_steps = [str(s) for s in (target.get("steps") or [])]
        safe_rank  = int(target.get("rank", -1))
        safe_liked = [str(x) for x in (profile.get("liked") or [])]
        safe_div   = str(profile.get("diversity", "low"))

        from fastapi.responses import JSONResponse as _JR
        import json as _json
        return _JR(content=_json.loads(_json.dumps({
            "steps":   safe_steps,
            "rank":    safe_rank,
            "profile": {"liked": safe_liked, "diversity": safe_div},
        })))
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"CoT explain error:\n{tb}")
        from fastapi.responses import JSONResponse as _JR
        return _JR(content={"steps": [f"Error: {str(e)}"], "rank": -1,
                             "profile": {"liked": [], "diversity": "low"}})


# Serve React build
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
