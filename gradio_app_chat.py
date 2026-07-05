"""
FA-CRS Gradio Demo — Redesigned UI
------------------------------------------------------
Dark mode · Glassmorphism · Chat-first layout · Persona-based user selection
Metric visualisations · Recommendation tables as comparison panel

Run from the project root after heterogeneous_kg.py has been run:

    pip install gradio requests matplotlib
    ollama pull llama3
    python gradio_app_chat.py

Fairness target p is fixed at 0.3 (the elbow of the FUT curve).
Ollama must be running locally: ollama serve
"""

import os, math, json, re, io, base64
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import requests
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from torch_geometric.nn import LightGCN
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_DIR        = "data"
KG_MODEL_PATH   = "outputs/kg/best_model_kg.pt"
EMBEDDING_DIM   = 64
NUM_LAYERS      = 3
MIN_RATING      = 4
TOP_K           = 10
CANDIDATE_K     = 50
P_FAIRNESS      = 0.3
OLLAMA_URL      = "http://localhost:11434/api/chat"
OLLAMA_MODEL    = "llama3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── PERSONAS ─────────────────────────────────────────────────────────────────
# Each persona maps to a genre preference used to pick a representative user.

PERSONAS = {
    "🎬 Action Fan":        {"genres": ["Action", "Thriller"], "desc": "Loves high-octane action and edge-of-seat thrillers"},
    "🎭 Drama Lover":       {"genres": ["Drama", "Romance"],   "desc": "Prefers deep narratives and emotional storytelling"},
    "😂 Comedy Enthusiast": {"genres": ["Comedy"],             "desc": "Here for laughs — sitcoms, stand-up, and rom-coms"},
    "👽 Sci-Fi Explorer":   {"genres": ["Sci-Fi", "Fantasy"],  "desc": "Drawn to speculative worlds and imaginative futures"},
    "🔪 Horror Buff":       {"genres": ["Horror"],             "desc": "Seeks tension, scares, and the macabre"},
    "🕵️ Mystery Aficionado":{"genres": ["Mystery", "Crime"],   "desc": "Enjoys whodunits and investigative narratives"},
    "🎞️ Indie/Art House":   {"genres": ["Documentary", "Animation"], "desc": "Appreciates auteur cinema and non-mainstream picks"},
}

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────

DARK_CSS = """
/* ── Global Reset ─────────────────────────────────────────── */
:root {
    --bg-primary:   #0d0f14;
    --bg-secondary: #13161e;
    --glass-bg:     rgba(255,255,255,0.045);
    --glass-border: rgba(255,255,255,0.08);
    --glass-shadow: 0 8px 32px rgba(0,0,0,0.5);
    --accent-1:     #7c3aed;
    --accent-2:     #3b82f6;
    --accent-grad:  linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%);
    --text-primary: #f1f5f9;
    --text-muted:   #94a3b8;
    --success:      #22c55e;
    --warning:      #f59e0b;
    --radius:       14px;
    --radius-sm:    8px;
}

body, .gradio-container {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* ── Gradient Header ───────────────────────────────────────── */
#header-block {
    background: linear-gradient(135deg, rgba(124,58,237,0.18) 0%, rgba(59,130,246,0.12) 100%);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 28px 32px 20px;
    margin-bottom: 20px;
    backdrop-filter: blur(12px);
    box-shadow: var(--glass-shadow);
}

#header-block h1 {
    background: var(--accent-grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2rem !important;
    font-weight: 800 !important;
    margin-bottom: 6px !important;
}

#header-block p {
    color: var(--text-muted) !important;
    font-size: 0.95rem !important;
}

/* ── Glassmorphism Panels ──────────────────────────────────── */
.glass-panel {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius) !important;
    backdrop-filter: blur(16px) !important;
    box-shadow: var(--glass-shadow) !important;
    padding: 20px !important;
}

/* ── Section Labels ────────────────────────────────────────── */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 10px;
}

/* ── Persona Cards ─────────────────────────────────────────── */
#persona-row .gr-button {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-size: 0.82rem !important;
    padding: 10px 8px !important;
    transition: all 0.2s ease !important;
    cursor: pointer;
    width: 100% !important;
    text-align: left !important;
}

#persona-row .gr-button:hover {
    background: rgba(124,58,237,0.18) !important;
    border-color: var(--accent-1) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(124,58,237,0.2) !important;
}

#persona-row .gr-button.selected {
    background: linear-gradient(135deg, rgba(124,58,237,0.25), rgba(59,130,246,0.2)) !important;
    border-color: var(--accent-1) !important;
}

/* ── Chat Window ───────────────────────────────────────────── */
#chatbot-panel {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius) !important;
}

#chatbot-panel .message.user {
    background: linear-gradient(135deg, rgba(124,58,237,0.35), rgba(59,130,246,0.25)) !important;
    border-radius: 18px 18px 4px 18px !important;
    color: #fff !important;
}

#chatbot-panel .message.bot {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 18px 18px 18px 4px !important;
    color: var(--text-primary) !important;
}

/* ── Chat Input ────────────────────────────────────────────── */
#chat-input-row textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    caret-color: var(--accent-1) !important;
    font-size: 0.95rem !important;
    padding: 12px 16px !important;
}

#chat-input-row textarea:focus {
    border-color: var(--accent-1) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
    outline: none !important;
}

#send-btn {
    background: var(--accent-grad) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-weight: 600 !important;
    padding: 0 24px !important;
    font-size: 0.9rem !important;
    transition: opacity 0.2s !important;
}

#send-btn:hover { opacity: 0.88 !important; }

/* ── Suggestion Chips ──────────────────────────────────────── */
#suggestion-row .gr-button {
    background: rgba(124,58,237,0.12) !important;
    border: 1px solid rgba(124,58,237,0.3) !important;
    border-radius: 20px !important;
    color: #a78bfa !important;
    font-size: 0.8rem !important;
    padding: 6px 14px !important;
    white-space: nowrap;
    transition: all 0.2s !important;
}

#suggestion-row .gr-button:hover {
    background: rgba(124,58,237,0.25) !important;
    color: #fff !important;
}

/* ── Tabs ──────────────────────────────────────────────────── */
.gr-tabs > .tab-nav {
    background: var(--glass-bg) !important;
    border-bottom: 1px solid var(--glass-border) !important;
    border-radius: var(--radius) var(--radius) 0 0 !important;
    padding: 0 16px !important;
}

.gr-tabs > .tab-nav button {
    color: var(--text-muted) !important;
    border: none !important;
    font-weight: 500 !important;
    padding: 12px 20px !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
}

.gr-tabs > .tab-nav button.selected {
    color: var(--text-primary) !important;
    border-bottom-color: var(--accent-1) !important;
}

/* ── Dataframes ────────────────────────────────────────────── */
.gr-dataframe table {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-size: 0.82rem !important;
}

.gr-dataframe th {
    background: rgba(255,255,255,0.05) !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    border-bottom: 1px solid var(--glass-border) !important;
}

.gr-dataframe tr:hover td {
    background: rgba(124,58,237,0.08) !important;
}

.gr-dataframe td {
    border-color: var(--glass-border) !important;
}

/* ── User info badge ───────────────────────────────────────── */
#user-badge {
    background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(59,130,246,0.15));
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 0.88rem;
    color: #c4b5fd;
}

/* ── Divider ───────────────────────────────────────────────── */
.divider {
    border: none;
    border-top: 1px solid var(--glass-border);
    margin: 24px 0;
}

/* ── Accordion (bottom panels) ─────────────────────────────── */
.gr-accordion {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 12px !important;
}

.gr-accordion .label-wrap {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    padding: 14px 18px !important;
    background: transparent !important;
}

/* ── Metrics image ─────────────────────────────────────────── */
#metrics-img img {
    border-radius: var(--radius) !important;
    max-width: 100% !important;
}

/* ── Scrollbar ─────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
"""


# ─── DATA / GRAPH / MODEL ─────────────────────────────────────────────────────

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
    return pos, movies, n_users, n_movies


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
    train_df = pd.DataFrame(train_rows, columns=["user_idx", "movie_idx"])
    val_df   = pd.DataFrame(val_rows,   columns=["user_idx", "movie_idx"])
    test_df  = pd.DataFrame(test_rows,  columns=["user_idx", "movie_idx"])
    return train_df, val_df, test_df


def build_kg(train_df, movies, n_users, n_movies):
    directors    = sorted(movies["director"].unique())
    dir2idx      = {d: i for i, d in enumerate(directors)}
    n_directors  = len(directors)
    dir_offset   = n_users + n_movies
    genders      = ["female", "male", "unknown"]
    gender2idx   = {g: i for i, g in enumerate(genders)}
    n_genders    = 3
    gender_offset = dir_offset + n_directors
    regions      = ["western", "non-western", "unknown"]
    region2idx   = {r: i for i, r in enumerate(regions)}
    n_regions    = 3
    region_offset = gender_offset + n_genders
    all_genres = set()
    for g in movies["genres"]:
        for genre in g.split("|"):
            all_genres.add(genre.strip())
    genres      = sorted(all_genres)
    genre2idx   = {g: i for i, g in enumerate(genres)}
    n_genres    = len(genres)
    genre_offset = region_offset + n_regions

    u_idx  = torch.tensor(train_df["user_idx"].values, dtype=torch.long)
    m_idx  = torch.tensor(train_df["movie_idx"].values + n_users, dtype=torch.long)
    um_src = torch.cat([u_idx, m_idx]); um_dst = torch.cat([m_idx, u_idx])
    movie_nodes = torch.tensor(movies["movie_idx"].values + n_users, dtype=torch.long)
    dir_nodes   = torch.tensor([dir2idx[d] + dir_offset for d in movies["director"]], dtype=torch.long)
    md_src = torch.cat([movie_nodes, dir_nodes]); md_dst = torch.cat([dir_nodes, movie_nodes])
    dir_nodes_g = torch.tensor([dir2idx[d] + dir_offset for d in movies["director"]], dtype=torch.long)
    gen_nodes   = torch.tensor([gender2idx[g] + gender_offset for g in movies["director_gender"]], dtype=torch.long)
    dg_src = torch.cat([dir_nodes_g, gen_nodes]); dg_dst = torch.cat([gen_nodes, dir_nodes_g])
    reg_nodes = torch.tensor([region2idx[r] + region_offset for r in movies["region"]], dtype=torch.long)
    mr_src = torch.cat([movie_nodes, reg_nodes]); mr_dst = torch.cat([reg_nodes, movie_nodes])
    mg_srcs, mg_dsts = [], []
    for _, row in movies.iterrows():
        m_node = int(row["movie_idx"]) + n_users
        for genre in row["genres"].split("|"):
            genre = genre.strip()
            if genre in genre2idx:
                g_node = genre2idx[genre] + genre_offset
                mg_srcs.extend([m_node, g_node]); mg_dsts.extend([g_node, m_node])
    mg_src = torch.tensor(mg_srcs, dtype=torch.long); mg_dst = torch.tensor(mg_dsts, dtype=torch.long)
    all_src    = torch.cat([um_src, md_src, dg_src, mr_src, mg_src])
    all_dst    = torch.cat([um_dst, md_dst, dg_dst, mr_dst, mg_dst])
    edge_index = torch.stack([all_src, all_dst], dim=0).to(device)
    n_total    = n_users + n_movies + n_directors + n_genders + n_regions + n_genres
    return edge_index, n_total


class LightGCNModel(nn.Module):
    def __init__(self, n_total, n_users, n_movies, embedding_dim, num_layers):
        super().__init__()
        self.n_users  = n_users
        self.n_movies = n_movies
        self.embedding = nn.Embedding(n_total, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.lgcn = LightGCN(n_total, embedding_dim, num_layers)
        self.lgcn.embedding = self.embedding

    def forward(self, edge_index):
        x = self.lgcn.get_embedding(edge_index)
        return x[:self.n_users], x[self.n_users: self.n_users + self.n_movies]


# ─── FA*IR RERANKING ──────────────────────────────────────────────────────────

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
                    any(ig in movies_indexed.loc[m, "genres"].lower()
                        for ig in include_lower)]
    if excluded_genres:
        excluded_lower = [g.lower() for g in excluded_genres]
        filtered = [(m, s) for m, s in filtered
                    if not any(eg in movies_indexed.loc[m, "genres"].lower()
                               for eg in excluded_lower
                               if m in movies_indexed.index)]
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

1. RECOMMEND — the user wants movies of a specific genre or type (e.g. "show me thrillers", "give me 10 horror films", "recommend some comedies").
   The system will fetch movies from the user's personalised model score AND apply FA*IR fairness reranking (p=0.3) to ensure gender and production country diversity.
   Respond with JSON only:
   {"intent": "recommend", "include_genres": ["Thriller"], "reason": "one friendly sentence telling the user what you're showing them and that fairness reranking is applied"}

2. FILTER — the user wants to remove certain genres from all recommendations (e.g. "no more action movies", "hide horror").
   Respond with JSON only:
   {"intent": "filter", "exclude_genres": ["Action"], "reason": "one sentence plain-English explanation"}

3. EXPLAIN — the user wants to know why a specific movie was recommended.
   Respond with JSON only:
   {"intent": "explain", "movie_title": "exact title from context"}

4. QUESTION — the user is asking something conversational (what is SPD? how does this work? why fairness? what is p=0.3?).
   Respond with JSON only:
   {"intent": "question", "answer": "your answer in 2-3 sentences, plain English, no jargon"}

Always return valid JSON. No markdown, no preamble. Pick the closest intent.
Key distinction: RECOMMEND = user wants a genre-focused list; FILTER = user wants to permanently hide a genre from all results."""

EXPLAIN_PROMPT = """You are explaining a movie recommendation to a user.

Movie: {title}
Genres: {genres}
Director: {director} ({gender}-directed, {region} production)
Why it appeared: {flag}

The user asked: {question}

Respond in 2-3 friendly sentences. Explain why this movie was recommended given their viewing history and the fairness goal of surfacing {flag_detail}. Be specific about the movie, not generic."""


def ollama_available():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def call_ollama(messages):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.3}
        }, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as e:
        return json.dumps({"intent": "question", "answer": f"Ollama error: {e}"})


def parse_intent(raw):
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(clean)
    except Exception:
        return {"intent": "question", "answer": raw}


def build_watch_history_summary(user_idx):
    seen = train_seen.get(user_idx, set())
    if not seen:
        return "no recorded watch history"
    rows = []
    for m in list(seen)[:10]:
        if m in movies_indexed.index:
            rows.append(movies_indexed.loc[m, "title"])
    return ", ".join(rows[:8]) + ("..." if len(rows) == 8 else "")


def generate_explanation(movie_idx, flag, user_question):
    if movie_idx not in movies_indexed.index:
        return "I don't have details on that film."
    row = movies_indexed.loc[movie_idx]
    flag_detail = {
        "gender":    "films by female directors",
        "region":    "non-western productions",
        "relevance": "movies that match your taste profile",
    }.get(flag, "relevant movies")
    prompt = EXPLAIN_PROMPT.format(
        title=row.get("title", "Unknown"),
        genres=str(row.get("genres", "")).replace("|", ", "),
        director=row.get("director", "Unknown"),
        gender=row.get("director_gender", "unknown"),
        region=row.get("region", "unknown"),
        flag=flag,
        question=user_question,
        flag_detail=flag_detail,
    )
    raw = call_ollama([{"role": "user", "content": prompt}])
    return raw


# ─── STARTUP ─────────────────────────────────────────────────────────────────

READY = False
LOAD_ERROR = ""
n_users = n_movies = 0
movies_indexed = None
movie_gender = movie_region = {}
test_df = train_seen = {}
ALL_CANDIDATES = {}

try:
    print("Loading data...")
    pos, movies, n_users, n_movies = load_data()
    train_df, val_df, test_df = split_data(pos)
    train_seen = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()

    print("Rebuilding knowledge graph...")
    edge_index, n_total = build_kg(train_df, movies, n_users, n_movies)

    print("Loading trained KG model weights...")
    model = LightGCNModel(n_total, n_users, n_movies, EMBEDDING_DIM, NUM_LAYERS).to(device)
    model.load_state_dict(torch.load(KG_MODEL_PATH, map_location=device))
    model.eval()

    movies_indexed = movies.set_index("movie_idx")
    movie_gender   = movies_indexed["director_gender"].to_dict()
    movie_region   = movies_indexed["region"].to_dict()

    female_movies     = set(m for m, g in movie_gender.items() if g == "female")
    nonwestern_movies = set(m for m, r in movie_region.items() if r == "non-western")

    print("Scoring all users (one forward pass)...")
    with torch.no_grad():
        user_emb, movie_emb = model(edge_index)
    all_scores = torch.matmul(user_emb, movie_emb.T).cpu().numpy()

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

    print("Building candidate pools...")
    ALL_CANDIDATES = {u: build_candidates(u) for u in tqdm(range(n_users))}

    # Build per-genre user index for persona selection
    # For each genre, find users whose training set contains the most of that genre
    genre_user_map = {}
    for u, seen_set in train_seen.items():
        for m in seen_set:
            if m in movies_indexed.index:
                for g in str(movies_indexed.loc[m, "genres"]).split("|"):
                    g = g.strip()
                    genre_user_map.setdefault(g, {})
                    genre_user_map[g][u] = genre_user_map[g].get(u, 0) + 1

    def persona_to_user(genres):
        """Return the user index with the most watches across the given genres."""
        counts = {}
        for g in genres:
            for u, c in genre_user_map.get(g, {}).items():
                counts[u] = counts.get(u, 0) + c
        if not counts:
            return 0
        return max(counts, key=counts.get)

    print(f"Ready. {n_users} users, {n_movies} movies. Ollama: {'available' if ollama_available() else 'not running'}")
    READY = True

except Exception as e:
    LOAD_ERROR = f"{type(e).__name__}: {e}"
    print(f"Could not load: {LOAD_ERROR}")


# ─── DISPLAY HELPERS ─────────────────────────────────────────────────────────

FLAG_LABELS = {
    "relevance": "Relevance match",
    "gender":    "Gender diversity pick",
    "region":    "Region diversity pick",
}

FLAG_EMOJI = {
    "relevance": "⭐",
    "gender":    "♀️",
    "region":    "🌍",
}


def rec_table(rec_list, flags):
    rows = []
    for rank, (m, flag) in enumerate(zip(rec_list, flags), 1):
        if movies_indexed is not None and m in movies_indexed.index:
            row = movies_indexed.loc[m]
            title    = row.get("title", f"Movie {m}")
            genres   = str(row.get("genres", "")).replace("|", ", ")
            director = row.get("director", "Unknown")
            gender   = row.get("director_gender", "unknown")
            region   = row.get("region", "unknown")
        else:
            title, genres, director, gender, region = f"Movie {m}", "", "Unknown", "unknown", "unknown"
        emoji = FLAG_EMOJI.get(flag, "")
        rows.append({
            "#": rank, "Title": title, "Genres": genres,
            "Director": director, "Gender": gender,
            "Region": region, "Reason": f"{emoji} {FLAG_LABELS.get(flag, flag)}",
        })
    return pd.DataFrame(rows)


def get_recommendations(user_id, excluded_genres=None, include_genres=None):
    user_id = int(user_id)
    cands = ALL_CANDIDATES.get(user_id, [])
    baseline_list  = [m for m, s in cands[:TOP_K]]
    fair_list, fair_flags = rerank_user(cands, excluded_genres, include_genres)
    return (rec_table(baseline_list, ["relevance"] * len(baseline_list)),
            rec_table(fair_list, fair_flags),
            fair_list, fair_flags)


# ─── METRICS CHART ───────────────────────────────────────────────────────────

def make_metrics_chart(baseline_list, fair_list, fair_flags):
    """Return a matplotlib figure comparing baseline vs FA*IR across fairness metrics."""
    plt.style.use("dark_background")

    # Compute stats
    def compute_stats(rec_list, flags=None):
        n = len(rec_list)
        if n == 0:
            return {"female_pct": 0, "nonwest_pct": 0, "relevance_pct": 0, "gender_pct": 0, "region_pct": 0}
        female   = sum(1 for m in rec_list if movie_gender.get(m) == "female")
        nonwest  = sum(1 for m in rec_list if movie_region.get(m) == "non-western")
        stats = {"female_pct": female / n * 100, "nonwest_pct": nonwest / n * 100}
        if flags:
            stats["relevance_pct"] = sum(1 for f in flags if f == "relevance") / n * 100
            stats["gender_pct"]    = sum(1 for f in flags if f == "gender")    / n * 100
            stats["region_pct"]    = sum(1 for f in flags if f == "region")    / n * 100
        return stats

    b_stats = compute_stats(baseline_list)
    f_stats = compute_stats(fair_list, fair_flags)

    # Colour palette
    COL_BASE  = "#3b82f6"
    COL_FAIR  = "#7c3aed"
    COL_FEM   = "#ec4899"
    COL_WEST  = "#f59e0b"
    BG        = "#13161e"
    PANEL_BG  = "#1a1d26"

    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    fig.patch.set_facecolor(BG)

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.88, bottom=0.1)

    ax_bar   = fig.add_subplot(gs[0, :2])   # bar chart spanning 2 cols
    ax_pie_g = fig.add_subplot(gs[0, 2])    # gender pie
    ax_pie_r = fig.add_subplot(gs[1, 0])    # region pie
    ax_flags = fig.add_subplot(gs[1, 1])    # flag breakdown
    ax_spd   = fig.add_subplot(gs[1, 2])    # SPD gauge

    for ax in [ax_bar, ax_pie_g, ax_pie_r, ax_flags, ax_spd]:
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3148")

    # ── Bar Chart: Diversity % comparison ─────────────────────────────────────
    metrics  = ["% Female\nDirector", "% Non-Western\nProduction"]
    baseline = [b_stats["female_pct"], b_stats["nonwest_pct"]]
    fair     = [f_stats["female_pct"], f_stats["nonwest_pct"]]
    x = np.arange(len(metrics))
    w = 0.32
    bars_b = ax_bar.bar(x - w/2, baseline, w, label="Baseline (relevance only)",
                        color=COL_BASE, alpha=0.85, zorder=3)
    bars_f = ax_bar.bar(x + w/2, fair,     w, label="FA★IR reranked",
                        color=COL_FAIR, alpha=0.85, zorder=3)
    ax_bar.axhline(P_FAIRNESS * 100, color="#22c55e", linewidth=1.2, linestyle="--",
                   alpha=0.7, label=f"Target p={P_FAIRNESS}")
    for bar in bars_b:
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f"{bar.get_height():.0f}%", ha="center", va="bottom",
                    color="#94a3b8", fontsize=9)
    for bar in bars_f:
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f"{bar.get_height():.0f}%", ha="center", va="bottom",
                    color="#c4b5fd", fontsize=9, fontweight="bold")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(metrics, color="#94a3b8", fontsize=10)
    ax_bar.set_ylabel("% of top-10 list", color="#94a3b8", fontsize=9)
    ax_bar.set_title("Fairness Dimension Comparison", color="#f1f5f9", fontsize=11, fontweight="bold", pad=10)
    ax_bar.legend(loc="upper right", fontsize=8, framealpha=0.15, labelcolor="#94a3b8")
    ax_bar.set_ylim(0, max(max(baseline + fair), P_FAIRNESS * 100) + 15)
    ax_bar.yaxis.grid(True, alpha=0.12, zorder=0); ax_bar.set_axisbelow(True)
    ax_bar.tick_params(colors="#4a5568")

    # ── Pie: Gender breakdown (FA*IR list) ────────────────────────────────────
    gender_counts = {"Female": 0, "Male": 0, "Unknown": 0}
    for m in fair_list:
        g = movie_gender.get(m, "unknown")
        if g == "female":    gender_counts["Female"]  += 1
        elif g == "male":    gender_counts["Male"]    += 1
        else:                gender_counts["Unknown"] += 1
    labels_g = [k for k, v in gender_counts.items() if v > 0]
    sizes_g  = [v for v in gender_counts.values()    if v > 0]
    colors_g = {"Female": COL_FEM, "Male": COL_BASE, "Unknown": "#4a5568"}
    pie_colors_g = [colors_g[l] for l in labels_g]
    wedges, texts, autotexts = ax_pie_g.pie(
        sizes_g, labels=None, colors=pie_colors_g, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(linewidth=1.5, edgecolor=PANEL_BG))
    for at in autotexts: at.set_color("#f1f5f9"); at.set_fontsize(9)
    ax_pie_g.set_title("Director Gender\n(FA★IR list)", color="#f1f5f9", fontsize=10, fontweight="bold")
    ax_pie_g.legend(labels_g, loc="lower center", fontsize=8, framealpha=0.0,
                    labelcolor="#94a3b8", ncol=len(labels_g), bbox_to_anchor=(0.5, -0.18))

    # ── Pie: Region breakdown (FA*IR list) ────────────────────────────────────
    region_counts = {"Non-western": 0, "Western": 0, "Unknown": 0}
    for m in fair_list:
        r = movie_region.get(m, "unknown")
        if r == "non-western": region_counts["Non-western"] += 1
        elif r == "western":   region_counts["Western"]     += 1
        else:                  region_counts["Unknown"]     += 1
    labels_r = [k for k, v in region_counts.items() if v > 0]
    sizes_r  = [v for v in region_counts.values()    if v > 0]
    colors_r = {"Non-western": COL_WEST, "Western": COL_BASE, "Unknown": "#4a5568"}
    pie_colors_r = [colors_r[l] for l in labels_r]
    wedges2, texts2, autotexts2 = ax_pie_r.pie(
        sizes_r, labels=None, colors=pie_colors_r, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(linewidth=1.5, edgecolor=PANEL_BG))
    for at in autotexts2: at.set_color("#f1f5f9"); at.set_fontsize(9)
    ax_pie_r.set_title("Production Region\n(FA★IR list)", color="#f1f5f9", fontsize=10, fontweight="bold")
    ax_pie_r.legend(labels_r, loc="lower center", fontsize=8, framealpha=0.0,
                    labelcolor="#94a3b8", ncol=len(labels_r), bbox_to_anchor=(0.5, -0.18))

    # ── Stacked bar: flag breakdown ────────────────────────────────────────────
    if fair_flags:
        rel_n   = fair_flags.count("relevance")
        gen_n   = fair_flags.count("gender")
        reg_n   = fair_flags.count("region")
        total   = len(fair_flags)
        heights = [rel_n, gen_n, reg_n]
        colors_f= ["#3b82f6", "#ec4899", "#f59e0b"]
        bottom  = 0
        for h, c, lbl in zip(heights, colors_f, ["Relevance", "Gender boost", "Region boost"]):
            if h > 0:
                ax_flags.bar(0, h, bottom=bottom, color=c, width=0.5, label=lbl)
                if h > 0.3:
                    ax_flags.text(0, bottom + h/2, f"{h}", ha="center", va="center",
                                  color="white", fontsize=11, fontweight="bold")
                bottom += h
        ax_flags.set_xlim(-0.6, 0.6)
        ax_flags.set_ylim(0, total + 1)
        ax_flags.set_xticks([])
        ax_flags.set_yticks(range(0, total + 1, 2))
        ax_flags.tick_params(colors="#4a5568")
        ax_flags.set_title("Slot Allocation\n(FA★IR list)", color="#f1f5f9", fontsize=10, fontweight="bold")
        ax_flags.legend(fontsize=8, framealpha=0.0, labelcolor="#94a3b8",
                        loc="lower right", bbox_to_anchor=(1.5, 0))
        ax_flags.yaxis.grid(True, alpha=0.12); ax_flags.set_axisbelow(True)
        ax_flags.set_ylabel("# slots", color="#94a3b8", fontsize=9)

    # ── SPD indicator ─────────────────────────────────────────────────────────
    # Estimate SPD from the top-10: mean score of protected vs unprotected
    # (simplified proxy — full SPD needs held-out labels)
    prot_g  = [m for m in fair_list if movie_gender.get(m) == "female"]
    unprot_g = [m for m in fair_list if movie_gender.get(m) != "female"]
    raw_female_pct = len(prot_g) / len(fair_list) * 100 if fair_list else 0
    # Show a simple "before / after" SPD proxy as a horizontal gauge
    spd_before = -0.82   # from paper
    spd_after  = -0.26   # from paper (68% reduction)
    gauge_vals  = [abs(spd_before), abs(spd_after)]
    gauge_cols  = [COL_BASE, COL_FAIR]
    gauge_lbls  = ["Baseline\nSPD", "FA★IR\nSPD"]
    bars_spd = ax_spd.barh(gauge_lbls, gauge_vals, color=gauge_cols, alpha=0.85, height=0.4)
    for bar, val in zip(bars_spd, [spd_before, spd_after]):
        ax_spd.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}", va="center", color="#94a3b8", fontsize=10)
    ax_spd.set_xlim(0, 1.1)
    ax_spd.set_xlabel("│SPD│ (lower = fairer)", color="#94a3b8", fontsize=8)
    ax_spd.set_title("Gender SPD\n(paper results)", color="#f1f5f9", fontsize=10, fontweight="bold")
    ax_spd.tick_params(colors="#94a3b8")
    ax_spd.xaxis.grid(True, alpha=0.12); ax_spd.set_axisbelow(True)

    # Super-title
    fig.suptitle("FA★IR Fairness Metrics Dashboard",
                 color="#f1f5f9", fontsize=14, fontweight="bold", y=0.96)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# ─── CONVERSATIONAL HANDLER ──────────────────────────────────────────────────

def chat(user_message, history, user_id, current_fair_list, current_fair_flags, excluded_genres_state, include_genres_state):
    user_id = int(user_id)

    if not ollama_available():
        reply = ("Ollama isn't running. Start it with `ollama serve` in a terminal, "
                 "then refresh the page. In the meantime you can still browse the recommendation tables.")
        history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]
        bt, ft, fl, ff = get_recommendations(user_id, excluded_genres_state, include_genres_state)
        chart = make_metrics_chart(fl, fl, ff)
        return history, bt, ft, excluded_genres_state, include_genres_state, fl, ff, chart

    context_lines = ["Current FA*IR recommendations for this user:"]
    if include_genres_state:
        context_lines.append(f"  (currently filtered to genres: {', '.join(include_genres_state)})")
    if excluded_genres_state:
        context_lines.append(f"  (currently excluding genres: {', '.join(excluded_genres_state)})")
    for i, (m, flag) in enumerate(zip(current_fair_list[:TOP_K], current_fair_flags[:TOP_K]), 1):
        if movies_indexed is not None and m in movies_indexed.index:
            row = movies_indexed.loc[m]
            context_lines.append(
                f"  {i}. {row.get('title','?')} — {row.get('genres','').replace('|',', ')} "
                f"({row.get('director_gender','?')}-directed, {row.get('region','?')}) [{flag}]")

    context = "\n".join(context_lines)
    watch_summary = build_watch_history_summary(user_id)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\n\nContext:\n{context}\nUser watch history sample: {watch_summary}"},
        {"role": "user",   "content": user_message}
    ]

    raw = call_ollama(messages)
    parsed = parse_intent(raw)
    intent = parsed.get("intent", "question")

    if intent == "recommend":
        new_include = parsed.get("include_genres", [])
        reason      = parsed.get("reason", f"Showing {', '.join(new_include)} films with FA*IR fairness applied.")
        _, fair_table, new_fair_list, new_fair_flags = get_recommendations(user_id, excluded_genres_state, new_include)
        baseline_table, _, _, _ = get_recommendations(user_id)
        n_found = len(new_fair_list)
        if n_found == 0:
            reply = (f"I couldn't find any {', '.join(new_include)} films in your unrated candidate pool. "
                     "Try a different genre or say 'reset filters' to start over.")
            new_include = include_genres_state
        else:
            reply = f"{reason} ({n_found} result{'s' if n_found != 1 else ''} found, fairness reranked at p={P_FAIRNESS}). Say 'show all genres' to clear."
        history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]
        chart = make_metrics_chart([m for m in new_fair_list], new_fair_list, new_fair_flags)
        return history, baseline_table, fair_table, excluded_genres_state, new_include, new_fair_list, new_fair_flags, chart

    elif intent == "filter":
        new_excluded = excluded_genres_state + parsed.get("exclude_genres", [])
        new_excluded = list(set(new_excluded))
        _, fair_table, new_fair_list, new_fair_flags = get_recommendations(user_id, new_excluded, include_genres_state)
        baseline_table, _, _, _ = get_recommendations(user_id)
        reason = parsed.get("reason", "Filtering applied.")
        excluded_str = ", ".join(new_excluded) if new_excluded else "none"
        reply = f"{reason} Currently excluding: {excluded_str}. Say 'reset filters' to clear."
        history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]
        chart = make_metrics_chart([m for m in new_fair_list], new_fair_list, new_fair_flags)
        return history, baseline_table, fair_table, new_excluded, include_genres_state, new_fair_list, new_fair_flags, chart

    elif intent == "explain":
        title_query = parsed.get("movie_title", "").lower()
        matched_idx = None
        for m in current_fair_list[:TOP_K]:
            if m in movies_indexed.index:
                t = str(movies_indexed.loc[m, "title"]).lower()
                if title_query in t or t in title_query:
                    matched_idx = m
                    break
        if matched_idx is None:
            reply = "I couldn't find that film in the current recommendations. Try asking about one of the titles shown."
        else:
            pos_in_list = current_fair_list.index(matched_idx)
            flag = current_fair_flags[pos_in_list] if pos_in_list < len(current_fair_flags) else "relevance"
            reply = generate_explanation(matched_idx, flag, user_message)
        history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]
        baseline_table, fair_table, fl2, ff2 = get_recommendations(user_id, excluded_genres_state, include_genres_state)
        chart = make_metrics_chart(fl2, fl2, ff2)
        return history, baseline_table, fair_table, excluded_genres_state, include_genres_state, fl2, ff2, chart

    else:
        answer = parsed.get("answer", raw)
        reset_msg = user_message.lower()
        if ("reset" in reset_msg or "clear" in reset_msg) and ("filter" in reset_msg or "genre" in reset_msg or "all" in reset_msg):
            answer = "Filters cleared. Showing your full personalised recommendations with FA*IR fairness applied."
            excluded_genres_state = []
            include_genres_state  = []
        history = history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": answer}]
        baseline_table, fair_table, fl2, ff2 = get_recommendations(user_id, excluded_genres_state, include_genres_state)
        chart = make_metrics_chart(fl2, fl2, ff2)
        return history, baseline_table, fair_table, excluded_genres_state, include_genres_state, fl2, ff2, chart


# ─── PERSONA SELECTION HELPERS ────────────────────────────────────────────────

def select_persona(persona_name):
    """Return (user_id, badge_html, history, bt, ft, fl, ff, chart)."""
    if not READY:
        return 0, "<div id='user-badge'>No data loaded.</div>", [], pd.DataFrame(), pd.DataFrame(), [], [], None
    p = PERSONAS[persona_name]
    user_id = persona_to_user(p["genres"])
    bt, ft, fl, ff = get_recommendations(user_id)
    badge = (f"<div id='user-badge'>👤 <b>{persona_name}</b> &nbsp;·&nbsp; "
             f"User #{user_id} &nbsp;·&nbsp; {p['desc']}</div>")
    welcome = [{"role": "assistant",
                "content": f"I've loaded a profile for a **{persona_name}** (user #{user_id}). "
                           f"This viewer loves {', '.join(p['genres'])} films. "
                           f"Ask me to recommend something, filter genres, or explain why a film appeared!"}]
    chart = make_metrics_chart(fl, fl, ff)
    return user_id, badge, welcome, bt, ft, fl, ff, chart


def send_suggestion(msg, history, user_id, fair_list, fair_flags, excl, incl):
    return chat(msg, history, user_id, fair_list, fair_flags, excl, incl)


# ─── UI BUILD ─────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="FA-CRS · Fairness-Aware Movie Recommender") as demo:

        # ── State ─────────────────────────────────────────────────────────────
        user_id_state    = gr.State(0)
        fair_list_state  = gr.State([])
        fair_flags_state = gr.State([])
        excluded_state   = gr.State([])
        include_state    = gr.State([])

        # ── Header ────────────────────────────────────────────────────────────
        with gr.Group(elem_id="header-block"):
            gr.Markdown(
                "# FA-CRS — Fairness-Aware Movie Recommender\n"
                f"Fairness target fixed at **p = {P_FAIRNESS}** (elbow of the FUT curve). "
                "Choose a viewer persona below, then chat to get personalised, fairness-reranked recommendations."
            )

        # ── Persona Selection ─────────────────────────────────────────────────
        gr.Markdown('<div class="section-label">① Choose a viewer persona</div>')
        user_badge = gr.HTML("<div id='user-badge'>No persona selected yet — pick one above to begin.</div>")

        with gr.Row(elem_id="persona-row"):
            persona_btns = []
            for pname in PERSONAS:
                btn = gr.Button(pname, size="sm")
                persona_btns.append((pname, btn))

        gr.HTML('<hr class="divider">')

        # ── Chat Section (MAIN) ───────────────────────────────────────────────
        gr.Markdown('<div class="section-label">② Chat with your recommender</div>')

        chatbot = gr.Chatbot(
            height=420, type="messages",
            elem_id="chatbot-panel",
            show_copy_button=True,
            avatar_images=(None, "https://api.dicebear.com/7.x/bottts-neutral/svg?seed=facrs"),
            bubble_full_width=False,
        )

        # Suggestion chips
        SUGGESTIONS = [
            "🎬 Recommend me 10 films",
            "🌍 Show non-western picks",
            "♀️ Films by female directors",
            "❓ How does FA★IR work?",
            "🔍 Explain the first recommendation",
            "🚫 No more action movies",
            "🔄 Reset all filters",
        ]
        with gr.Row(elem_id="suggestion-row"):
            sug_btns = [gr.Button(s, size="sm") for s in SUGGESTIONS]

        with gr.Row(elem_id="chat-input-row"):
            chat_input = gr.Textbox(
                placeholder="Ask for a genre, explain a pick, or filter results...",
                show_label=False, scale=6, lines=1,
            )
            send_btn = gr.Button("Send ↗", variant="primary", scale=1, elem_id="send-btn")

        gr.HTML('<hr class="divider">')

        # ── Bottom Panels ─────────────────────────────────────────────────────
        gr.Markdown('<div class="section-label">③ Comparison & metrics (for analysis)</div>')

        with gr.Accordion("📊 Fairness Metrics Dashboard", open=True):
            metrics_img = gr.Image(label=None, show_label=False, elem_id="metrics-img",
                                   show_download_button=True, height=420)

        with gr.Accordion("📋 Recommendation Lists (side-by-side comparison)", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Baseline** — Relevance only (LightGCN scores, no fairness)")
                    baseline_out = gr.Dataframe(interactive=False, wrap=True)
                with gr.Column():
                    gr.Markdown(f"**FA★IR Reranked** — Fairness target p = {P_FAIRNESS}")
                    fair_out = gr.Dataframe(interactive=False, wrap=True)

        # ── Wire up Persona Buttons ───────────────────────────────────────────
        for pname, pbtn in persona_btns:
            pbtn.click(
                fn=lambda p=pname: select_persona(p),
                inputs=[],
                outputs=[user_id_state, user_badge, chatbot, baseline_out, fair_out,
                         fair_list_state, fair_flags_state, metrics_img],
            )

        # ── Wire up Suggestion Chips ──────────────────────────────────────────
        chat_in_list = [chat_input, chatbot, user_id_state, fair_list_state, fair_flags_state, excluded_state, include_state]
        chat_out_list = [chatbot, baseline_out, fair_out, excluded_state, include_state, fair_list_state, fair_flags_state, metrics_img]

        for sbtn, suggestion in zip(sug_btns, SUGGESTIONS):
            sbtn.click(
                fn=lambda h, uid, fl, ff, ex, inc, s=suggestion:
                    chat(s, h, uid, fl, ff, ex, inc),
                inputs=[chatbot, user_id_state, fair_list_state, fair_flags_state, excluded_state, include_state],
                outputs=chat_out_list,
            )

        # ── Wire up Chat Send ─────────────────────────────────────────────────
        send_btn.click(chat, inputs=chat_in_list, outputs=chat_out_list).then(
            lambda: "", outputs=chat_input)
        chat_input.submit(chat, inputs=chat_in_list, outputs=chat_out_list).then(
            lambda: "", outputs=chat_input)

    return demo


def build_error_ui():
    with gr.Blocks(css=DARK_CSS, title="FA-CRS — Setup needed") as demo:
        with gr.Group(elem_id="header-block"):
            gr.Markdown("# FA-CRS — Setup needed")
            gr.Markdown(
                "Run the pipeline first:\n\n"
                "1. `python data_prep.py`\n"
                "2. `python lightgcn_baseline.py`\n"
                "3. `python heterogeneous_kg.py`\n\n"
                f"Error: `{LOAD_ERROR}`"
            )
    return demo


if __name__ == "__main__":
    app = build_ui() if READY else build_error_ui()
    app.launch(css=DARK_CSS)
