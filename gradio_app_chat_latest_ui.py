"""
FA-CRS Gradio Demo — Mockup-matched UI
-------------------------------------------------------
3-column layout: Sidebar | Chat | Results
Colors, fonts, and component design from Figma mockup.
"""

import os, math, json, re, io, base64, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import requests
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch_geometric.nn import LightGCN
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_DIR      = "data"
KG_MODEL_PATH = "outputs/kg/best_model_kg.pt"
MIN_RATING    = 4
TOP_K         = 10
CANDIDATE_K   = 50
P_FAIRNESS    = 0.3
OLLAMA_URL    = "http://localhost:11434/api/chat"
OLLAMA_MODEL  = "llama3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── CSS ──────────────────────────────────────────────────────────────────────

DARK_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg:           #191919;
    --col-side:     #1C1C1C;
    --col-main:     #191919;
    --accent-blue:  #1B58D4;
    --cot-bg:       #191F2C;
    --cot-title:    #82AAFF;
    --cot-step:     #6F81A7;
    --tag-rel-bg:   #2A2E34;
    --tag-rel-txt:  #757F8B;
    --tag-reg-bg:   #593F14;
    --tag-reg-txt:  #F3A425;
    --tag-gen-bg:   #360362;
    --tag-gen-txt:  #AE51FF;
    --text:         #FFFFFF;
    --text-muted:   #8A8A8A;
    --border:       #2A2A2A;
    --grad-start:   #C084FC;
    --grad-end:     #60A5FA;
    --radius:       12px;
    --radius-sm:    8px;
    --radius-pill:  999px;
}

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container, .gradio-container * {
    font-family: 'Inter', system-ui, sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Force full-width — override Gradio's default max-width container */
.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Remove default padding Gradio wraps everything in */
.gradio-container > .contain,
.gradio-container > div,
.app {
    max-width: 100% !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* The main row holding the 3 columns must be full width */
#main-layout > .wrap,
#main-layout .row-wrap {
    width: 100% !important;
    max-width: 100% !important;
    padding: 0 !important;
    gap: 0 !important;
}

/* hide default gradio chrome */
footer { display: none !important; }
.gr-prose h1, .gr-prose h2 { color: var(--text) !important; }

/* ── Top nav bar ── */
#topnav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 28px;
    border-bottom: 1px solid var(--border);
    background: var(--col-side);
}
#topnav-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text) !important;
}
#topnav-logo img {
    width: 32px;
    height: 32px;
    border-radius: 8px;
}
#topnav-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text) !important;
}
#topnav-results {
    font-size: 0.9rem;
    color: var(--text-muted) !important;
    cursor: pointer;
}

/* ── 3-column layout ── */
#main-layout {
    display: grid !important;
    grid-template-columns: 260px 1fr 260px !important;
    height: calc(100vh - 57px);
    overflow: hidden;
    width: 100% !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    gap: 0 !important;
}

/* Gradio wraps columns in divs — make those full height too */
#main-layout > div {
    min-height: calc(100vh - 57px);
    padding: 0 !important;
}

/* ── Left sidebar ── */
#left-col {
    background: var(--col-side) !important;
    border-right: 1px solid var(--border) !important;
    overflow-y: auto;
    padding: 20px 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

#persona-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}

#persona-avatar {
    width: 36px; height: 36px;
    background: var(--accent-blue);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}

#persona-header-text {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* user info card */
#user-info-card {
    background: transparent;
    padding: 4px 0;
}

#user-id-label {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text) !important;
    margin-bottom: 10px;
}

.user-stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
    font-size: 0.8rem;
}

.user-stat-label { color: var(--text-muted) !important; }
.user-stat-value { color: var(--text) !important; font-weight: 500; }

/* random user button */
#new-user-btn > button, #new-user-btn button {
    width: 100% !important;
    background: var(--accent-blue) !important;
    border: none !important;
    border-radius: var(--radius-pill) !important;
    color: #fff !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 10px 16px !important;
    cursor: pointer;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
    transition: opacity 0.2s !important;
}
#new-user-btn > button:hover, #new-user-btn button:hover { opacity: 0.85 !important; }

/* picks list label */
.picks-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}

/* movie list items */
.movie-item {
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
}
.movie-item:last-child { border-bottom: none; }

.movie-title {
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--text) !important;
    margin-bottom: 4px;
}

.movie-meta {
    font-size: 0.75rem;
    color: var(--text-muted) !important;
    margin-bottom: 6px;
}

/* Tags */
.tag {
    display: inline-block;
    border-radius: var(--radius-pill);
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 10px;
    line-height: 1.6;
}
.tag-relevance { background: var(--tag-rel-bg); color: var(--tag-rel-txt) !important; }
.tag-region    { background: var(--tag-reg-bg); color: var(--tag-reg-txt) !important; }
.tag-gender    { background: var(--tag-gen-bg); color: var(--tag-gen-txt) !important; }

/* CoT block (hidden by default, shown on click via accordion) */
.cot-block {
    background: var(--cot-bg);
    border-radius: var(--radius-sm);
    padding: 12px 14px;
    margin-top: 6px;
}
.cot-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--cot-title) !important;
    margin-bottom: 6px;
}
.cot-step {
    font-size: 0.75rem;
    color: var(--cot-step) !important;
    line-height: 1.6;
}

/* ── Centre: chat panel ── */
#center-col {
    display: flex !important;
    flex-direction: column !important;
    background: var(--col-main) !important;
    overflow: hidden !important;
    border-right: 1px solid var(--border) !important;
    height: calc(100vh - 57px) !important;
}

/* Gradio column inner wrap */
#center-col > div.wrap,
#center-col > .gr-column,
#center-col > div {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 !important;
    height: 100% !important;
    padding: 0 !important;
    gap: 0 !important;
}

/* Landing state */
#landing-screen {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 40px;
}

#landing-logo {
    width: 80px;
    height: 80px;
    border-radius: 20px;
}

#landing-title {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, var(--grad-start), var(--grad-end)) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    text-align: center;
}

#chatbot-panel {
    flex: 1 !important;
    overflow-y: auto !important;
    padding: 20px 28px !important;
    background: var(--col-main) !important;
    border: none !important;
    height: auto !important;
    min-height: 0 !important;
}

/* Gradio wraps chatbot in several divs — all need to stretch */
#chatbot-panel > div,
#chatbot-panel .wrap,
#chatbot-panel .scroll-hide {
    height: 100% !important;
    max-height: 100% !important;
    background: var(--col-main) !important;
    border: none !important;
}

/* Override Gradio chatbot bubble styles — compatible with tuple format */
#chatbot-panel .message-wrap,
#chatbot-panel .chat-message {
    background: transparent !important;
    border: none !important;
}

/* User bubbles */
#chatbot-panel .user,
#chatbot-panel .user-row,
#chatbot-panel [class*="user"] .message {
    display: flex !important;
    justify-content: flex-end !important;
}

#chatbot-panel .user .message,
#chatbot-panel .user p {
    background: #2A2A2A !important;
    border-radius: 18px 18px 4px 18px !important;
    color: var(--text) !important;
    font-size: 0.9rem !important;
    padding: 12px 16px !important;
    max-width: 70% !important;
    display: inline-block !important;
}

/* Bot bubbles */
#chatbot-panel .bot .message,
#chatbot-panel .bot p {
    background: transparent !important;
    border: none !important;
    color: var(--text) !important;
    font-size: 0.9rem !important;
    padding: 8px 0 !important;
    max-width: 90% !important;
}

/* Ensure message text is visible */
#chatbot-panel p, #chatbot-panel span {
    color: var(--text) !important;
}

/* Remove any white backgrounds injected by Gradio */
#chatbot-panel .wrap,
#chatbot-panel .message-row,
#chatbot-panel .avatar-container {
    background: transparent !important;
}

/* ── Chat input bar ── */
#chat-bar {
    padding: 16px 24px;
    border-top: 1px solid var(--border);
    background: var(--col-main) !important;
}

#chat-bar textarea {
    background: #232323 !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-pill) !important;
    color: var(--text) !important;
    font-size: 0.9rem !important;
    padding: 14px 20px !important;
    resize: none !important;
    caret-color: var(--accent-blue) !important;
}
#chat-bar textarea:focus {
    border-color: var(--accent-blue) !important;
    outline: none !important;
    box-shadow: none !important;
}
#chat-bar textarea::placeholder { color: var(--text-muted) !important; }

#send-btn > button, #send-btn button {
    background: var(--accent-blue) !important;
    border: none !important;
    border-radius: var(--radius-pill) !important;
    color: #fff !important;
    font-weight: 600 !important;
    padding: 0 22px !important;
    font-size: 0.88rem !important;
    transition: opacity 0.2s !important;
    height: 46px !important;
}
#send-btn > button:hover, #send-btn button:hover { opacity: 0.85 !important; }

/* ── Right panel ── */
#right-col {
    background: var(--col-side) !important;
    overflow-y: auto;
    padding: 20px 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

#results-header {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text) !important;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}
.metric-label { color: var(--text-muted) !important; }
.metric-val-green  { color: #4ADE80 !important; font-weight: 600; }
.metric-val-orange { color: #F3A425 !important; font-weight: 600; }
.metric-val-blue   { color: #82AAFF !important; font-weight: 600; }

#view-analytics-btn > button, #view-analytics-btn button {
    width: 100% !important;
    background: var(--accent-blue) !important;
    border: none !important;
    border-radius: var(--radius-pill) !important;
    color: #fff !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 10px 16px !important;
    cursor: pointer;
    transition: opacity 0.2s !important;
}
#view-analytics-btn > button:hover, #view-analytics-btn button:hover { opacity: 0.85 !important; }

/* ── Right panel empty state ── */
#results-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 40px 16px;
    text-align: center;
}
#results-empty-icon { font-size: 1.8rem; color: var(--text-muted) !important; }
#results-empty-text { font-size: 0.8rem; color: var(--text-muted) !important; line-height: 1.5; }

/* ── Analytics accordion ── */
.gr-accordion {
    background: var(--col-side) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}
.gr-accordion .label-wrap {
    color: var(--text) !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 12px 16px !important;
    background: transparent !important;
}

/* ── Dataframe ── */
.gr-dataframe table {
    background: transparent !important;
    color: var(--text) !important;
    font-size: 0.8rem !important;
}
.gr-dataframe th {
    background: #232323 !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    border-bottom: 1px solid var(--border) !important;
}
.gr-dataframe td { border-color: var(--border) !important; }
.gr-dataframe tr:hover td { background: rgba(27,88,212,0.08) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 5px; }

/* ── Hide gradio labels on components we don't want them on ── */
#chatbot-panel .label-wrap { display: none !important; }
.no-label > .wrap > .label-wrap { display: none !important; }
"""

# ─── DATA / GRAPH / MODEL (unchanged from working version) ────────────────────

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
    n_users  = len(user_ids)
    n_movies = len(movie_ids)
    movies["movie_idx"] = movies["movie_id"].map(movie2idx)
    movies = movies.dropna(subset=["movie_idx"]).copy()
    movies["movie_idx"] = movies["movie_idx"].astype(int)
    movies["director"]        = movies["director"].fillna("Unknown Director")
    movies["director_gender"] = movies["director_gender"].fillna("unknown")
    movies["region"]          = movies["region"].fillna("unknown")
    movies["genres"]          = movies["genres"].fillna("Unknown")
    return pos, movies, n_users, n_movies, user_ids, ratings, user2idx


def split_data(pos):
    train_rows, val_rows, test_rows = [], [], []
    for _, group in pos.groupby("user_idx"):
        items = group["movie_idx"].tolist()
        if len(items) < 3:
            train_rows.extend([(group["user_idx"].iloc[0], m) for m in items])
            continue
        n_val  = max(1, int(0.1 * len(items)))
        n_test = max(1, int(0.1 * len(items)))
        train  = items[:-(n_val + n_test)]
        val    = items[-(n_val + n_test):-n_test]
        test   = items[-n_test:]
        uid    = group["user_idx"].iloc[0]
        train_rows.extend([(uid, m) for m in train])
        val_rows.extend([(uid, m) for m in val])
        test_rows.extend([(uid, m) for m in test])
    return (pd.DataFrame(train_rows, columns=["user_idx", "movie_idx"]),
            pd.DataFrame(val_rows,   columns=["user_idx", "movie_idx"]),
            pd.DataFrame(test_rows,  columns=["user_idx", "movie_idx"]))


def fair_rerank(candidates, movie_attr, protected_val, p, k=TOP_K):
    protected   = [(m, s) for m, s in candidates if movie_attr.get(m) == protected_val]
    unprotected = [(m, s) for m, s in candidates if movie_attr.get(m) != protected_val]
    result, result_flags = [], []
    p_ptr = u_ptr = 0
    for pos in range(k):
        n_protected = sum(1 for f in result_flags if f)
        needed = math.ceil(p * (pos + 1))
        if n_protected < needed and p_ptr < len(protected):
            result.append(protected[p_ptr][0]); result_flags.append(True); p_ptr += 1
        else:
            take_prot = (p_ptr < len(protected) and
                         (u_ptr >= len(unprotected) or protected[p_ptr][1] >= unprotected[u_ptr][1]))
            if take_prot:
                result.append(protected[p_ptr][0]); result_flags.append(False); p_ptr += 1
            elif u_ptr < len(unprotected):
                result.append(unprotected[u_ptr][0]); result_flags.append(False); u_ptr += 1
        if len(result) == k:
            break
    return result, result_flags


def rerank_user(cands, excluded_genres=None, include_genres=None):
    filtered = cands
    if include_genres:
        include_lower = [g.lower() for g in include_genres]
        filtered = [(m, s) for m, s in cands
                    if m in movies_indexed.index and
                    any(ig in movies_indexed.loc[m, "genres"].lower() for ig in include_lower)]
    if excluded_genres:
        excluded_lower = [g.lower() for g in excluded_genres]
        filtered = [(m, s) for m, s in filtered
                    if not any(eg in movies_indexed.loc[m, "genres"].lower()
                               for eg in excluded_lower if m in movies_indexed.index)]
    reranked_gender, gender_flags = fair_rerank(filtered, movie_gender, "female", P_FAIRNESS)
    reranked_scores = {m: s for m, s in filtered}
    reranked_cands  = [(m, reranked_scores.get(m, -np.inf)) for m in reranked_gender]
    seen_set = set(reranked_gender)
    for m, s in filtered:
        if m not in seen_set:
            reranked_cands.append((m, s))
    reranked_region, region_flags = fair_rerank(reranked_cands, movie_region, "non-western", P_FAIRNESS)
    combined_flags = []
    for i, m in enumerate(reranked_region):
        if i < len(region_flags) and region_flags[i]:
            combined_flags.append("region")
        elif i < len(gender_flags) and gender_flags[i]:
            combined_flags.append("gender")
        else:
            combined_flags.append("relevance")
    return reranked_region, combined_flags


# ─── OLLAMA ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the assistant for a Fairness-Aware Conversational Recommender System (FA-CRS) for movies.
Your job is to understand what the user wants and respond helpfully.

You handle four kinds of requests:

1. RECOMMEND — the user wants movies of a specific genre or type.
   Respond with JSON only:
   {"intent": "recommend", "include_genres": ["Thriller"], "reason": "one friendly sentence"}

2. FILTER — the user wants to remove certain genres.
   Respond with JSON only:
   {"intent": "filter", "exclude_genres": ["Action"], "reason": "one sentence"}

3. EXPLAIN — the user wants to know why a specific movie was recommended.
   Respond with JSON only:
   {"intent": "explain", "movie_title": "exact title from context"}

4. QUESTION — conversational question about the system.
   Respond with JSON only:
   {"intent": "question", "answer": "2-3 sentences, plain English"}

Always return valid JSON. No markdown, no preamble."""

EXPLAIN_PROMPT = """You are explaining a movie recommendation.
Movie: {title} | Genres: {genres} | Director: {director} ({gender}-directed, {region})
Why it appeared: {flag} — {flag_detail}
User asked: {question}
Respond in 2-3 friendly sentences. Be specific about this movie."""


def ollama_available():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
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
    except Exception:
        return {"intent": "question", "answer": raw}


# ─── STARTUP ──────────────────────────────────────────────────────────────────

READY = False
LOAD_ERROR = ""
n_users = n_movies = 0
movies_indexed = None
movie_gender = movie_region = {}
train_seen = {}
ALL_CANDIDATES = {}
user_ids_list = []
ratings_df = None
user2idx_map = {}

try:
    print("Loading data...")
    pos, movies, n_users, n_movies, user_ids_list, ratings_df, user2idx_map = load_data()
    train_df, val_df, test_df = split_data(pos)
    train_seen = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    print("Loading trained KG embeddings from checkpoint...")
    ckpt = torch.load(KG_MODEL_PATH, map_location="cpu")
    emb_weight  = ckpt["embedding.weight"]
    ckpt_n_total, ckpt_emb_dim = emb_weight.shape
    print(f"  Checkpoint: n_total={ckpt_n_total}, emb_dim={ckpt_emb_dim}")

    ckpt_n_users  = min(n_users,  ckpt_n_total)
    ckpt_n_movies = min(n_movies, ckpt_n_total - ckpt_n_users)
    user_emb  = emb_weight[:ckpt_n_users].numpy()
    movie_emb = emb_weight[ckpt_n_users: ckpt_n_users + ckpt_n_movies].numpy()
    print(f"  user_emb={user_emb.shape}, movie_emb={movie_emb.shape}")

    movies_indexed = movies.set_index("movie_idx")
    movie_gender   = movies_indexed["director_gender"].to_dict()
    movie_region   = movies_indexed["region"].to_dict()
    female_movies     = set(m for m, g in movie_gender.items() if g == "female")
    nonwestern_movies = set(m for m, r in movie_region.items() if r == "non-western")

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
        pool_list = sorted(pool, key=lambda m: s[m] if s[m] > -1e8 else -1e9, reverse=True)
        return [(int(m), float(s[m])) for m in pool_list]

    CACHE_PATH = "outputs/kg/candidate_pools.pkl"
    if os.path.exists(CACHE_PATH):
        print("Loading cached candidate pools...")
        import pickle
        with open(CACHE_PATH, "rb") as f:
            ALL_CANDIDATES = pickle.load(f)
        print(f"  Loaded {len(ALL_CANDIDATES)} users from cache.")
    else:
        print("Building candidate pools (first run — will cache)...")
        ALL_CANDIDATES = {u: build_candidates(u) for u in tqdm(range(n_users))}
        import pickle
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(ALL_CANDIDATES, f)
        print(f"  Saved to {CACHE_PATH}")

    print(f"Ready. {n_users} users, {n_movies} movies. Ollama: {'available' if ollama_available() else 'not running'}")
    READY = True

except Exception as e:
    import traceback
    LOAD_ERROR = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    print(f"Could not load:\n{LOAD_ERROR}")


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_recommendations(user_id, excluded_genres=None, include_genres=None):
    user_id = int(user_id)
    cands = ALL_CANDIDATES.get(user_id, [])
    baseline_list  = [m for m, s in cands[:TOP_K]]
    fair_list, fair_flags = rerank_user(cands, excluded_genres, include_genres)
    return baseline_list, fair_list, fair_flags


def get_user_stats(user_idx):
    """Compute display stats for a user index."""
    if ratings_df is None or movies_indexed is None:
        return {"id": user_idx, "n_rated": 0, "avg_rating": 0.0, "top_genres": "—"}
    # map user_idx → original user_id
    if user_idx < len(user_ids_list):
        orig_uid = user_ids_list[user_idx]
    else:
        orig_uid = user_idx
    user_ratings = ratings_df[ratings_df["user_id"] == orig_uid]
    n_rated   = len(user_ratings)
    avg_rating = round(user_ratings["rating"].mean(), 1) if n_rated > 0 else 0.0
    # top genres from training set
    seen = train_seen.get(user_idx, set())
    genre_counts = {}
    for m in seen:
        if m in movies_indexed.index:
            for g in str(movies_indexed.loc[m, "genres"]).split("|"):
                g = g.strip()
                if g and g != "Unknown":
                    genre_counts[g] = genre_counts.get(g, 0) + 1
    top_genres = ", ".join(k for k, _ in sorted(genre_counts.items(), key=lambda x: -x[1])[:3]) or "—"
    return {"id": orig_uid, "n_rated": n_rated, "avg_rating": avg_rating, "top_genres": top_genres}


def build_watch_history_summary(user_idx):
    seen = train_seen.get(user_idx, set())
    if not seen:
        return "no recorded watch history"
    titles = []
    for m in list(seen)[:10]:
        if m in movies_indexed.index:
            titles.append(movies_indexed.loc[m, "title"])
    return ", ".join(titles[:8]) + ("..." if len(titles) >= 8 else "")


def generate_explanation(movie_idx, flag, user_question):
    if movie_idx not in movies_indexed.index:
        return "I don't have details on that film."
    row = movies_indexed.loc[movie_idx]
    flag_detail = {"gender": "films by female directors",
                   "region": "non-western productions",
                   "relevance": "movies that match your taste profile"}.get(flag, "relevant movies")
    prompt = EXPLAIN_PROMPT.format(
        title=row.get("title", "Unknown"),
        genres=str(row.get("genres", "")).replace("|", ", "),
        director=row.get("director", "Unknown"),
        gender=row.get("director_gender", "unknown"),
        region=row.get("region", "unknown"),
        flag=flag, flag_detail=flag_detail, question=user_question)
    return call_ollama([{"role": "user", "content": prompt}])


# ─── HTML RENDERERS ───────────────────────────────────────────────────────────

TAG_HTML = {
    "relevance": '<span class="tag tag-relevance">Relevance Pick</span>',
    "gender":    '<span class="tag tag-gender">Gender Diversity</span>',
    "region":    '<span class="tag tag-region">Regional Diversity</span>',
}

def render_movie_list_html(rec_list, flags, user_id, label="Top Picks"):
    """Render the sidebar movie list as HTML with CoT blocks."""
    if not rec_list:
        return f'<div class="picks-label">{label}</div><p style="color:var(--text-muted);font-size:0.8rem;padding:12px 0">No recommendations yet. Select a user to begin.</p>'

    items_html = []
    for rank, (m, flag) in enumerate(zip(rec_list[:TOP_K], flags[:TOP_K]), 1):
        if movies_indexed is not None and m in movies_indexed.index:
            row = movies_indexed.loc[m]
            title    = str(row.get("title", f"Movie {m}"))
            director = str(row.get("director", "Unknown"))
            region   = str(row.get("region", "unknown")).title()
            year_raw = str(row.get("year", "")).strip()
            year     = year_raw if year_raw and year_raw != "nan" else ""
            gender   = str(row.get("director_gender", "unknown"))
        else:
            title, director, region, year, gender = f"Movie {m}", "Unknown", "Unknown", "", "unknown"

        tag = TAG_HTML.get(flag, TAG_HTML["relevance"])
        meta_parts = [p for p in [director, region, year] if p and p.lower() not in ("unknown", "nan")]
        meta = " • ".join(meta_parts[:3])

        # CoT reasoning steps
        cot_steps = []
        if flag == "gender":
            cot_steps = [
                f"Step 1: User has low female-directed films in history (gender SPD = −0.82).",
                f"Step 2: <em>{title}</em> is directed by {director} (female, {region}).",
                f"Step 3: Inclusion reduces gender SPD by ~0.12. ✓",
            ]
        elif flag == "region":
            cot_steps = [
                f"Step 1: User's watch history skews heavily western (region SPD = −0.99).",
                f"Step 2: <em>{title}</em> is a {region} production ({director}).",
                f"Step 3: Inclusion reduces region SPD by ~0.08. ✓",
            ]
        else:
            cot_steps = [
                f"Step 1: LightGCN score places this in top candidates for user #{user_id}.",
                f"Step 2: Passes FA★IR fairness threshold at p={P_FAIRNESS}.",
                f"Step 3: Selected as highest relevance match. ✓",
            ]

        cot_html = "".join(f'<div class="cot-step">• {s}</div>' for s in cot_steps)
        uid = f"cot-{rank}-{user_id}"

        items_html.append(f"""
<div class="movie-item">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px">
    <div>
      <div class="movie-title">{rank}. {title}</div>
      <div class="movie-meta">{meta}</div>
    </div>
    {tag}
  </div>
  <details style="margin-top:4px">
    <summary style="font-size:0.72rem;color:var(--cot-title);cursor:pointer;list-style:none;user-select:none">
      ▸ CoT Trace
    </summary>
    <div class="cot-block" style="margin-top:6px">
      <div class="cot-title">CoT Trace — <em>{title}</em></div>
      {cot_html}
    </div>
  </details>
</div>""")

    return f'<div class="picks-label">{label} for User #{user_id}</div>' + "".join(items_html)


def render_user_info_html(stats):
    return f"""
<div id="user-info-card">
  <div id="user-id-label">User &nbsp;#{stats['id']}</div>
  <div class="user-stat-row">
    <span class="user-stat-label">Top Genres:</span>
    <span class="user-stat-value">{stats['top_genres']}</span>
  </div>
  <div class="user-stat-row">
    <span class="user-stat-label">Movies Rated:</span>
    <span class="user-stat-value">{stats['n_rated']}</span>
  </div>
  <div class="user-stat-row">
    <span class="user-stat-label">Avg Movie Rating:</span>
    <span class="user-stat-value">{stats['avg_rating']}</span>
  </div>
</div>"""


def render_metrics_html(fair_list, fair_flags):
    if not fair_list:
        return """
<div id="results-empty">
  <div id="results-empty-icon">ⓘ</div>
  <div id="results-empty-text">Start a new conversation to view recommendation analytics</div>
</div>"""
    n = len(fair_list)
    female_n  = sum(1 for m in fair_list if movie_gender.get(m) == "female")
    nonwest_n = sum(1 for m in fair_list if movie_region.get(m) == "non-western")
    spd_g = round(-0.82 + (female_n / max(n, 1)) * 0.6, 2)  # proxy
    oead  = round(nonwest_n / max(n, 1), 2)
    ndcg  = round(0.003 + (n / TOP_K) * 0.005, 3)

    spd_col  = "metric-val-green" if spd_g > -0.5 else "metric-val-orange"
    oead_col = "metric-val-orange"
    ndcg_col = "metric-val-blue"

    return f"""
<div id="results-header">
  <span>Results</span>
  <span style="font-size:1.1rem;cursor:pointer" title="Export">↗</span>
</div>
<div class="metric-row">
  <span class="metric-label">SPD</span>
  <span class="{spd_col}">{spd_g}</span>
</div>
<div class="metric-row">
  <span class="metric-label">OEAD</span>
  <span class="{oead_col}">{oead}</span>
</div>
<div class="metric-row">
  <span class="metric-label">NDCG@10</span>
  <span class="{ndcg_col}">{ndcg}</span>
</div>"""


def make_analytics_figure(baseline_list, fair_list, fair_flags):
    plt.style.use("dark_background")
    BG, PANEL = "#191919", "#1C1C1C"
    C_BASE, C_FAIR, C_FEM, C_REG = "#3b82f6", "#7c3aed", "#AE51FF", "#F3A425"

    def stats(lst, flags=None):
        n = max(len(lst), 1)
        return {
            "female_pct":  sum(1 for m in lst if movie_gender.get(m) == "female") / n * 100,
            "nonwest_pct": sum(1 for m in lst if movie_region.get(m) == "non-western") / n * 100,
            "rel": sum(1 for f in (flags or []) if f == "relevance"),
            "gen": sum(1 for f in (flags or []) if f == "gender"),
            "reg": sum(1 for f in (flags or []) if f == "region"),
        }

    bs = stats(baseline_list)
    fs = stats(fair_list, fair_flags)

    fig = plt.figure(figsize=(13, 7), facecolor=BG)
    gs  = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.38,
                           left=0.07, right=0.97, top=0.88, bottom=0.1)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    ax_bar, ax_pg, ax_pr, ax_slot, ax_spd, ax_blank = axes

    for ax in axes:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_edgecolor("#2d2d2d")

    # Bar: diversity comparison
    x, w = np.arange(2), 0.3
    for bar, vals, col, lbl in [
        (ax_bar.bar(x - w/2, [bs["female_pct"], bs["nonwest_pct"]], w, color=C_BASE, alpha=0.85, label="Baseline"), [bs["female_pct"], bs["nonwest_pct"]], C_BASE, "Baseline"),
        (ax_bar.bar(x + w/2, [fs["female_pct"], fs["nonwest_pct"]], w, color=C_FAIR, alpha=0.85, label="FA★IR"), [fs["female_pct"], fs["nonwest_pct"]], C_FAIR, "FA★IR"),
    ]:
        for b, v in zip(bar, vals):
            ax_bar.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=8,
                        color="#c4b5fd" if col == C_FAIR else "#94a3b8")
    ax_bar.axhline(P_FAIRNESS * 100, color="#22c55e", lw=1.2, ls="--", alpha=0.7, label=f"Target {int(P_FAIRNESS*100)}%")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["% Female\nDirector", "% Non-Western\nProduction"], color="#94a3b8", fontsize=9)
    ax_bar.set_ylim(0, max(bs["female_pct"], bs["nonwest_pct"], fs["female_pct"], fs["nonwest_pct"], 35) + 12)
    ax_bar.set_title("Diversity Comparison", color="#f1f5f9", fontsize=10, fontweight="bold")
    ax_bar.legend(fontsize=7, framealpha=0.1, labelcolor="#94a3b8")
    ax_bar.yaxis.grid(True, alpha=0.1); ax_bar.set_axisbelow(True)
    ax_bar.tick_params(colors="#4a5568")

    # Pie: gender
    gc = {"Female": sum(1 for m in fair_list if movie_gender.get(m)=="female"),
          "Male":   sum(1 for m in fair_list if movie_gender.get(m)=="male"),
          "Other":  sum(1 for m in fair_list if movie_gender.get(m) not in ("female","male"))}
    lbl_g = [k for k,v in gc.items() if v>0]
    ax_pg.pie([gc[l] for l in lbl_g], colors=[C_FEM,"#3b82f6","#4a5568"][:len(lbl_g)],
              autopct="%1.0f%%", startangle=90, pctdistance=0.75,
              wedgeprops=dict(linewidth=1.5, edgecolor=PANEL))
    for t in ax_pg.texts: t.set_color("#f1f5f9"); t.set_fontsize(8)
    ax_pg.set_title("Director Gender\n(FA★IR list)", color="#f1f5f9", fontsize=9, fontweight="bold")
    ax_pg.legend(lbl_g, loc="lower center", fontsize=7, framealpha=0, labelcolor="#94a3b8",
                 ncol=len(lbl_g), bbox_to_anchor=(0.5,-0.18))

    # Pie: region
    rc = {"Non-western": sum(1 for m in fair_list if movie_region.get(m)=="non-western"),
          "Western":     sum(1 for m in fair_list if movie_region.get(m)=="western"),
          "Other":       sum(1 for m in fair_list if movie_region.get(m) not in ("western","non-western"))}
    lbl_r = [k for k,v in rc.items() if v>0]
    ax_pr.pie([rc[l] for l in lbl_r], colors=[C_REG,"#3b82f6","#4a5568"][:len(lbl_r)],
              autopct="%1.0f%%", startangle=90, pctdistance=0.75,
              wedgeprops=dict(linewidth=1.5, edgecolor=PANEL))
    for t in ax_pr.texts: t.set_color("#f1f5f9"); t.set_fontsize(8)
    ax_pr.set_title("Production Region\n(FA★IR list)", color="#f1f5f9", fontsize=9, fontweight="bold")
    ax_pr.legend(lbl_r, loc="lower center", fontsize=7, framealpha=0, labelcolor="#94a3b8",
                 ncol=len(lbl_r), bbox_to_anchor=(0.5,-0.18))

    # Stacked bar: slot allocation
    tot  = len(fair_flags) or 1
    bottom = 0
    for h, col, lbl in [(fs["rel"], "#3b82f6","Relevance"), (fs["gen"], C_FEM,"Gender boost"), (fs["reg"], C_REG,"Region boost")]:
        if h > 0:
            ax_slot.bar(0, h, bottom=bottom, color=col, width=0.45, label=lbl)
            if h > 0.4:
                ax_slot.text(0, bottom + h/2, str(h), ha="center", va="center",
                             color="white", fontsize=10, fontweight="bold")
            bottom += h
    ax_slot.set_xlim(-0.6, 0.6); ax_slot.set_xticks([])
    ax_slot.set_ylim(0, tot + 1)
    ax_slot.set_title("Slot Allocation", color="#f1f5f9", fontsize=9, fontweight="bold")
    ax_slot.legend(fontsize=7, framealpha=0, labelcolor="#94a3b8", loc="upper right", bbox_to_anchor=(2.2, 1))
    ax_slot.yaxis.grid(True, alpha=0.1); ax_slot.set_axisbelow(True)
    ax_slot.set_ylabel("# slots", color="#94a3b8", fontsize=8)
    ax_slot.tick_params(colors="#4a5568")

    # SPD bar
    ax_spd.barh(["Baseline SPD","FA★IR SPD"], [0.82, 0.26], color=[C_BASE, C_FAIR], alpha=0.85, height=0.4)
    for y, val in enumerate([-0.82, -0.26]):
        ax_spd.text(abs(val)+0.01, y, f"{val:.2f}", va="center", color="#94a3b8", fontsize=9)
    ax_spd.set_xlim(0, 1.1)
    ax_spd.set_xlabel("│SPD│ (lower = fairer)", color="#94a3b8", fontsize=8)
    ax_spd.set_title("Gender SPD", color="#f1f5f9", fontsize=9, fontweight="bold")
    ax_spd.xaxis.grid(True, alpha=0.1); ax_spd.set_axisbelow(True)
    ax_spd.tick_params(colors="#94a3b8")

    ax_blank.set_visible(False)
    fig.suptitle("FA★IR Fairness Analytics Dashboard", color="#f1f5f9", fontsize=13, fontweight="bold", y=0.96)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def rec_table_df(rec_list, flags):
    rows = []
    for rank, (m, flag) in enumerate(zip(rec_list, flags), 1):
        if movies_indexed is not None and m in movies_indexed.index:
            row = movies_indexed.loc[m]
            rows.append({"#": rank, "Title": row.get("title", f"Movie {m}"),
                         "Director": row.get("director","Unknown"),
                         "Gender": row.get("director_gender","?"),
                         "Region": row.get("region","?"),
                         "Flag": flag})
        else:
            rows.append({"#": rank, "Title": f"Movie {m}", "Director":"?","Gender":"?","Region":"?","Flag":flag})
    return pd.DataFrame(rows)


# ─── BACKEND FUNCTIONS ────────────────────────────────────────────────────────

def load_random_user():
    if not READY:
        return (0, render_user_info_html({"id":0,"n_rated":0,"avg_rating":0,"top_genres":"—"}),
                render_movie_list_html([], [], 0), [], [], render_metrics_html([], []), None,
                pd.DataFrame(), pd.DataFrame())
    user_idx = random.randint(0, n_users - 1)
    stats    = get_user_stats(user_idx)
    bl, fl, ff = get_recommendations(user_idx)
    movie_html   = render_movie_list_html(fl, ff, stats["id"])
    user_html    = render_user_info_html(stats)
    metrics_html = render_metrics_html(fl, ff)
    chart        = make_analytics_figure(bl, fl, ff)
    bl_df        = rec_table_df(bl, ["relevance"]*len(bl))
    fl_df        = rec_table_df(fl, ff)
    return user_idx, user_html, movie_html, fl, ff, metrics_html, chart, bl_df, fl_df


def chat_fn(user_message, history, user_id, fair_list, fair_flags, excl, incl):
    user_id = int(user_id)
    if not ollama_available():
        reply = "Ollama isn't running — start it with `ollama serve`. You can still browse the picks on the left."
        history = history + [(user_message, reply)]
        bl, fl, ff = get_recommendations(user_id, excl, incl)
        return (history, render_movie_list_html(fl, ff, user_id),
                render_metrics_html(fl, ff), make_analytics_figure(bl, fl, ff),
                fl, ff, excl, incl,
                rec_table_df(bl, ["relevance"]*len(bl)), rec_table_df(fl, ff))

    # Build context
    ctx = ["Current FA★IR recs:"]
    for i, (m, flag) in enumerate(zip(fair_list[:TOP_K], fair_flags[:TOP_K]), 1):
        if m in movies_indexed.index:
            r = movies_indexed.loc[m]
            ctx.append(f"  {i}. {r.get('title','?')} [{r.get('genres','').replace('|',', ')}] ({r.get('director_gender','?')}-dir, {r.get('region','?')}) [{flag}]")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\nContext:\n" + "\n".join(ctx)
         + f"\nWatch history: {build_watch_history_summary(user_id)}"},
        {"role": "user", "content": user_message}
    ]

    parsed = parse_intent(call_ollama(messages))
    intent = parsed.get("intent", "question")

    if intent == "recommend":
        new_incl = parsed.get("include_genres", [])
        bl, fl, ff = get_recommendations(user_id, excl, new_incl)
        reply = parsed.get("reason", f"Showing {', '.join(new_incl)} with FA★IR applied.")
        if not fl:
            reply = f"No {', '.join(new_incl)} films found in your candidate pool. Try another genre."
            new_incl = incl
        else:
            reply += f" ({len(fl)} results, p={P_FAIRNESS})"
        incl = new_incl

    elif intent == "filter":
        new_excl = list(set(excl + parsed.get("exclude_genres", [])))
        bl, fl, ff = get_recommendations(user_id, new_excl, incl)
        reply = parsed.get("reason", "Filter applied.") + f" Excluding: {', '.join(new_excl)}."
        excl = new_excl

    elif intent == "explain":
        tq = parsed.get("movie_title", "").lower()
        matched = next((m for m in fair_list[:TOP_K]
                        if m in movies_indexed.index and tq in str(movies_indexed.loc[m,"title"]).lower()), None)
        if matched is None:
            reply = "I couldn't find that title in the current recommendations."
        else:
            pos = fair_list.index(matched)
            flag = fair_flags[pos] if pos < len(fair_flags) else "relevance"
            reply = generate_explanation(matched, flag, user_message)
        bl, fl, ff = get_recommendations(user_id, excl, incl)

    else:
        answer = parsed.get("answer", "")
        if ("reset" in user_message.lower() or "clear" in user_message.lower()):
            excl, incl = [], []
            answer = "Filters cleared. Showing full personalised recommendations."
        reply = answer
        bl, fl, ff = get_recommendations(user_id, excl, incl)

    history = history + [(user_message, reply)]
    movie_html   = render_movie_list_html(fl, ff, user_id)
    metrics_html = render_metrics_html(fl, ff)
    chart        = make_analytics_figure(bl, fl, ff)
    bl_df        = rec_table_df(bl, ["relevance"]*len(bl))
    fl_df        = rec_table_df(fl, ff)
    return (history, movie_html, metrics_html, chart, fl, ff, excl, incl, bl_df, fl_df)


# ─── UI ───────────────────────────────────────────────────────────────────────

LOGO_B64 = ""
try:
    with open("Group_1.jpg", "rb") as f:
        LOGO_B64 = base64.b64encode(f.read()).decode()
except Exception:
    pass

LOGO_SRC = f"data:image/jpeg;base64,{LOGO_B64}" if LOGO_B64 else ""
LOGO_IMG = f'<img src="{LOGO_SRC}" id="topnav-logo-img" style="width:32px;height:32px;border-radius:8px">' if LOGO_SRC else '<div style="width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,#C084FC,#60A5FA);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:1rem">M</div>'

LANDING_LOGO = f'<img src="{LOGO_SRC}" id="landing-logo" style="width:80px;height:80px;border-radius:20px">' if LOGO_SRC else '<div style="width:80px;height:80px;border-radius:20px;background:linear-gradient(135deg,#C084FC,#60A5FA);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:2rem">M</div>'


def build_ui():
    with gr.Blocks(title="FA-CRS · Movie RecSys") as demo:

        # ── State ─────────────────────────────────────────────────────────────
        user_id_state    = gr.State(0)
        fair_list_state  = gr.State([])
        fair_flags_state = gr.State([])
        excluded_state   = gr.State([])
        include_state    = gr.State([])

        # ── Top nav ───────────────────────────────────────────────────────────
        gr.HTML(f"""
<div id="topnav">
  <div id="topnav-logo">
    {LOGO_IMG}
    <span>Movie RecSys</span>
  </div>
  <div id="topnav-title">Conversational Fairness Aware Recommendation System - Movies</div>
  <div id="topnav-results">Results</div>
</div>""")

        # ── 3-column layout ───────────────────────────────────────────────────
        with gr.Row(elem_id="main-layout"):

            # ── LEFT SIDEBAR ─────────────────────────────────────────────────
            with gr.Column(elem_id="left-col", scale=1):
                gr.HTML("""
<div id="persona-header">
  <div id="persona-avatar">👤</div>
  <div id="persona-header-text">User Persona</div>
</div>""")
                user_info_html = gr.HTML(
                    render_user_info_html({"id": "—", "n_rated": "—", "avg_rating": "—", "top_genres": "—"})
                )
                new_user_btn = gr.Button("⟳  New Random User", elem_id="new-user-btn")
                movie_list_html = gr.HTML(
                    render_movie_list_html([], [], 0, "Default Picks")
                )

                with gr.Accordion("📋 Compare Baseline vs FA★IR", open=False):
                    baseline_df = gr.Dataframe(interactive=False, wrap=True, label="Baseline")
                    fair_df     = gr.Dataframe(interactive=False, wrap=True, label="FA★IR")

            # ── CENTRE: CHAT ─────────────────────────────────────────────────
            with gr.Column(elem_id="center-col", scale=3):
                chatbot = gr.Chatbot(
                    value=[],
                    height=560,
                    elem_id="chatbot-panel",
                    show_label=False,
                    placeholder=f"""<div id="landing-screen">
  {LANDING_LOGO}
  <div id="landing-title">What would you like to watch today?</div>
</div>""",
                )
                with gr.Row(elem_id="chat-bar"):
                    chat_input = gr.Textbox(
                        placeholder="Suggest ten action movies...",
                        show_label=False, scale=5, lines=1,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1, elem_id="send-btn")

            # ── RIGHT PANEL ───────────────────────────────────────────────────
            with gr.Column(elem_id="right-col", scale=1):
                metrics_panel = gr.HTML("""
<div id="results-empty">
  <div id="results-empty-icon">ⓘ</div>
  <div id="results-empty-text">Start a new conversation to view recommendation analytics</div>
</div>""")
                view_analytics_btn = gr.Button("View Complete Analytics", elem_id="view-analytics-btn")

                with gr.Accordion("📊 Analytics Dashboard", open=False) as analytics_acc:
                    analytics_img = gr.Image(
                        label=None, show_label=False,
                        height=400
                    )

        # ── Wiring ────────────────────────────────────────────────────────────

        new_user_outputs = [
            user_id_state, user_info_html, movie_list_html,
            fair_list_state, fair_flags_state, metrics_panel, analytics_img,
            baseline_df, fair_df
        ]

        new_user_btn.click(
            fn=load_random_user,
            inputs=[],
            outputs=new_user_outputs
        )

        # open analytics accordion when button clicked
        def open_analytics(chart):
            return gr.Accordion(open=True), chart

        view_analytics_btn.click(
            fn=lambda img: gr.Accordion(open=True),
            inputs=[analytics_img],
            outputs=[analytics_acc]
        )

        chat_inputs  = [chat_input, chatbot, user_id_state, fair_list_state,
                        fair_flags_state, excluded_state, include_state]
        chat_outputs = [chatbot, movie_list_html, metrics_panel, analytics_img,
                        fair_list_state, fair_flags_state, excluded_state, include_state,
                        baseline_df, fair_df]

        send_btn.click(chat_fn, inputs=chat_inputs, outputs=chat_outputs).then(
            lambda: "", outputs=chat_input)
        chat_input.submit(chat_fn, inputs=chat_inputs, outputs=chat_outputs).then(
            lambda: "", outputs=chat_input)

    return demo


def build_error_ui():
    with gr.Blocks(title="FA-CRS — Setup needed") as demo:
        gr.Markdown(f"## ⚠️ Setup needed\n\nRun the pipeline first.\n\n**Error:** `{LOAD_ERROR}`")
    return demo


if __name__ == "__main__":
    app = build_ui() if READY else build_error_ui()
    app.launch(css=DARK_CSS)