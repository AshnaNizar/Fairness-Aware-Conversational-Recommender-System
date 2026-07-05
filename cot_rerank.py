"""
CoT Reranking — Stage 3 of FA-CRS pipeline
============================================
Three modes:

1. BATCH (python cot_rerank.py)
   LightGCN scores → FA*IR → CoT rerank → saves metrics + examples

2. INTERACTIVE SINGLE USER (python cot_rerank.py --interactive --user 42)
   Loads one user, applies preference constraints from CLI, prints CoT per movie

3. API (imported by api_server.py)
   interactive_cot_rerank() — takes a user's candidate list + conversational
   preferences and returns a reranked list with per-movie CoT steps

"""

import os, json, math, time, re, argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR      = "data"
KG_DIR        = "outputs/kg"
OUTPUT_DIR    = "outputs/cot"
MIN_RATING    = 4
TOP_K         = 10
COT_USERS     = 200
COT_MIN_PROT  = 0.25
LLM_MODEL     = "claude-sonnet-4-6"
API_DELAY     = 0.3

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── DATA ──────────────────────────────────────────────────────────────────────
def load_data():
    ratings  = pd.read_csv(f"{DATA_DIR}/ratings.csv")
    movies   = pd.read_csv(f"{DATA_DIR}/movies_enriched.csv")
    pos      = ratings[ratings["rating"] >= MIN_RATING][["user_id","movie_id"]].copy()
    user2idx = {u: i for i, u in enumerate(sorted(pos["user_id"].unique()))}
    movie2idx= {m: i for i, m in enumerate(sorted(pos["movie_id"].unique()))}
    pos["user_idx"]  = pos["user_id"].map(user2idx)
    pos["movie_idx"] = pos["movie_id"].map(movie2idx)
    movies["movie_idx"] = movies["movie_id"].map(movie2idx)
    movies = movies.dropna(subset=["movie_idx"]).copy()
    movies["movie_idx"] = movies["movie_idx"].astype(int)
    for col in ["director","director_gender","region","genres","title"]:
        movies[col] = movies.get(col, pd.Series(dtype=str)).fillna("unknown")
    return pos, movies, len(user2idx), len(movie2idx)


def split_data(pos):
    train, val, test = [], [], []
    for _, g in pos.groupby("user_idx"):
        items = g["movie_idx"].tolist()
        uid   = g["user_idx"].iloc[0]
        if len(items) < 3:
            train += [(uid, m) for m in items]; continue
        nv = nt = max(1, int(.1*len(items)))
        train += [(uid, m) for m in items[:-(nv+nt)]]
        val   += [(uid, m) for m in items[-(nv+nt):-nt]]
        test  += [(uid, m) for m in items[-nt:]]
    mk = lambda r: pd.DataFrame(r, columns=["user_idx","movie_idx"])
    return mk(train), mk(val), mk(test)


# ── EMBEDDINGS ────────────────────────────────────────────────────────────────
class _EmbScorer:
    def __init__(self, user_emb, movie_emb):
        self.user_emb  = user_emb
        self.movie_emb = movie_emb
    def score_all(self):
        return self.user_emb @ self.movie_emb.T


def load_scorer(n_users, n_movies):
    ckpt = torch.load(f"{KG_DIR}/best_model_kg.pt", map_location="cpu")
    emb  = ckpt["embedding.weight"]
    nt, _ = emb.shape
    nu = min(n_users,  nt)
    nm = min(n_movies, nt - nu)
    return _EmbScorer(emb[:nu].numpy(), emb[nu:nu+nm].numpy())


# ── CANDIDATES ────────────────────────────────────────────────────────────────
def get_candidates(scorer, train_df, n_users, n_movies, movie_gender, movie_region):
    scores  = scorer.score_all()
    seen    = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    f_pool  = {m for m,g in movie_gender.items() if g == "female"}
    nw_pool = {m for m,r in movie_region.items() if r == "non-western"}
    cands   = {}
    for u in range(min(n_users, scores.shape[0])):
        s = scores[u].copy()
        for m in seen.get(u, set()):
            if m < len(s): s[m] = -np.inf
        top = set(np.argpartition(s, -50)[-50:].tolist())
        for pool in [f_pool, nw_pool]:
            for m in sorted([m for m in pool if m not in seen.get(u,set()) and m<len(s)],
                            key=lambda m: s[m], reverse=True)[:10]:
                top.add(m)
        cands[u] = [(int(m), float(s[m]))
                    for m in sorted(top, key=lambda m: s[m] if s[m]>-1e8 else -1e9, reverse=True)]
    return cands


# ── FA*IR ─────────────────────────────────────────────────────────────────────
def fair_rerank(cands, attr, protected, p, k=TOP_K):
    prot   = [(m,s) for m,s in cands if attr.get(m)==protected]
    unprot = [(m,s) for m,s in cands if attr.get(m)!=protected]
    result, flags, pi, ui = [], [], 0, 0
    for pos in range(k):
        need = math.ceil(p*(pos+1)) - sum(flags)
        if need > 0 and pi < len(prot):
            result.append(prot[pi][0]); flags.append(True); pi += 1
        elif pi < len(prot) and (ui>=len(unprot) or prot[pi][1]>=unprot[ui][1]):
            result.append(prot[pi][0]); flags.append(False); pi += 1
        elif ui < len(unprot):
            result.append(unprot[ui][0]); flags.append(False); ui += 1
        if len(result)==k: break
    return result, flags


def run_fair(cands, movie_gender, movie_region, p=0.3):
    recs, flags = {}, {}
    for u, c in cands.items():
        rg, gf = fair_rerank(c, movie_gender, "female", p)
        sc = {m:s for m,s in c}
        rc = [(m, sc.get(m,-np.inf)) for m in rg] + [(m,s) for m,s in c if m not in set(rg)]
        rr, rf = fair_rerank(rc, movie_region, "non-western", p)
        recs[u]  = rr
        flags[u] = ["region" if i<len(rf) and rf[i] else
                    "gender" if i<len(gf) and gf[i] else "relevance"
                    for i in range(len(rr))]
    return recs, flags


# ── PREFERENCE PROFILE ────────────────────────────────────────────────────────
def infer_profile(user_idx, train_df, movies):
    seen  = movies[movies["movie_idx"].isin(
        train_df[train_df["user_idx"]==user_idx]["movie_idx"])]
    gc = {}
    for gs in seen["genres"].fillna(""):
        for g in gs.split("|"):
            gc[g.strip()] = gc.get(g.strip(), 0) + 1
    liked = sorted(gc, key=gc.get, reverse=True)[:3]
    years = [int(m.group(1)) for t in seen.get("title", pd.Series()).fillna("")
             for m in [re.search(r"\((\d{4})\)", str(t))] if m]
    era   = f"{(int(np.median(years))//10)*10}s" if years else None
    div   = ((seen["region"]=="non-western").sum() +
             (seen["director_gender"]=="female").sum()) / max(len(seen), 1)
    return {"liked": liked, "era": era,
            "diversity": "high" if div>.15 else "medium" if div>.05 else "low"}


# ── RULE-BASED CoT (batch, one step string per movie) ─────────────────────────
def rule_based_cot(cand_dicts, profile, k=TOP_K):
    liked = set(profile["liked"])
    for c in cand_dicts:
        match  = len({g.strip() for g in c["genres"].split("|")} & liked)
        bonus  = (0.15*(c["director_gender"]=="female") +
                  0.15*(c["region"]=="non-western")) if profile["diversity"] != "low" else 0
        c["_cot_score"] = c["score"] + 0.1*match + bonus
    ranked = sorted(cand_dicts, key=lambda c: c["_cot_score"], reverse=True)
    prot_min   = math.ceil(COT_MIN_PROT * k)
    prot_count = sum(1 for c in ranked[:k]
                     if c["director_gender"]=="female" or c["region"]=="non-western")
    if prot_count < prot_min:
        prot_extra = [c for c in ranked[k:]
                      if c["director_gender"]=="female" or c["region"]=="non-western"]
        swap_idxs  = [i for i,c in enumerate(ranked[:k])
                      if c["director_gender"]!="female" and c["region"]!="western"]
        for si, pc in zip(swap_idxs, prot_extra):
            if prot_count >= prot_min: break
            ranked[si] = pc; prot_count += 1
    steps = [
        f"#{i+1} {c['title']} — "
        f"{'diversity pick (' + c['director_gender'] + ', ' + c['region'] + ')' if c['director_gender']=='female' or c['region']=='non-western' else 'relevance pick'}"
        f", genres: {c['genres']}"
        for i, c in enumerate(ranked[:k])
    ]
    return [c["movie_idx"] for c in ranked[:k]], steps


# ── LLM CoT ───────────────────────────────────────────────────────────────────
def llm_cot(cand_dicts, profile, client, k=TOP_K):
    lines = [
        f"{i+1}. {c['title']} | {c['genres']} | {c['director']} ({c['director_gender']}) "
        f"| {c['region']} | score:{c['score']:.3f}"
        + (" [fairness-promoted]" if c["fairness_flag"]!="relevance" else "")
        for i, c in enumerate(cand_dicts)
    ]
    prompt = (
        f"Re-rank these {k} movies for a user who likes {', '.join(profile['liked'])} "
        f"(diversity appetite: {profile['diversity']}).\n\n"
        f"Candidates:\n{chr(10).join(lines)}\n\n"
        f"Think step by step, then output:\n"
        f"REASONING: <brief per-item reasoning>\n"
        f"FINAL_RANKING:\n1. <title>\n...{k}. <title>\n\n"
        f"Constraint: at least {math.ceil(COT_MIN_PROT*k)} items must be female-directed OR non-western."
    )
    try:
        resp = client.messages.create(
            model=LLM_MODEL, max_tokens=600,
            messages=[{"role":"user","content":prompt}])
        text = resp.content[0].text
    except Exception as e:
        print(f"  LLM error: {e}, falling back to rule-based")
        ids, steps = rule_based_cot(cand_dicts, profile, k)
        return ids, "\n".join(steps)
    title2idx = {c["title"]: c["movie_idx"] for c in cand_dicts}
    final_ids = []
    if "FINAL_RANKING:" in text:
        for line in text.split("FINAL_RANKING:")[-1].strip().split("\n"):
            line = re.sub(r"^\d+\.\s*", "", line.strip())
            for title, idx in title2idx.items():
                if title.lower() in line.lower() and idx not in final_ids:
                    final_ids.append(idx); break
            if len(final_ids)==k: break
    for c in cand_dicts:
        if len(final_ids)==k: break
        if c["movie_idx"] not in final_ids: final_ids.append(c["movie_idx"])
    return final_ids[:k], text


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE CoT — the core new function used by both CLI and api_server.py
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_cot_rerank(
    cand_dicts,          # list of movie dicts from FA*IR stage
    profile,             # user taste profile from infer_profile()
    user_message=None,   # raw conversational message (optional, for richer steps)
    excluded_genres=None,# genres to filter out  e.g. ["Horror", "Action"]
    include_genres=None, # genres to prioritise  e.g. ["Sci-Fi"]
    k=TOP_K,
):
    """
    Interactive CoT reranking.

    Takes the FA*IR candidate list, applies conversational preference constraints
    (genre filters / boosts from the user's message), re-scores with CoT logic,
    enforces the fairness floor, and returns the final ranked list with per-movie
    step-by-step reasoning that references the user's stated preferences.

    Returns
    -------
    results : list[dict]
        Each dict has:
          movie_idx, title, director, director_gender, region, genres,
          fairness_flag, cot_score, rank, steps (list[str])
    """
    excluded_genres = [g.lower() for g in (excluded_genres or [])]
    include_genres  = [g.lower() for g in (include_genres  or [])]
    liked_genres    = set(profile.get("liked", []))

    # ── Step 1: Apply hard genre exclusions ───────────────────────────────────
    filtered = []
    excluded_titles = []
    for c in cand_dicts:
        movie_genres_lower = [g.strip().lower() for g in c["genres"].split("|")]
        if any(eg in movie_genres_lower for eg in excluded_genres):
            excluded_titles.append(c["title"])
            continue
        filtered.append(c)

    # ── Step 2: Score each movie with CoT reasoning ───────────────────────────
    for c in filtered:
        movie_genres = {g.strip() for g in c["genres"].split("|")}
        movie_genres_lower = {g.lower() for g in movie_genres}

        # genre match with user's historical preferences
        history_match  = len(movie_genres & liked_genres)
        # genre match with user's current request
        request_match  = len(movie_genres_lower & set(include_genres))
        # fairness bonus
        is_female   = c["director_gender"] == "female"
        is_nonwest  = c["region"] == "non-western"
        fair_bonus  = (0.15*is_female + 0.15*is_nonwest) if profile.get("diversity") != "low" else 0

        c["_history_match"] = history_match
        c["_request_match"] = request_match
        c["_fair_bonus"]    = fair_bonus
        c["_cot_score"]     = (
            c["score"]
            + 0.1  * history_match
            + 0.25 * request_match   # stronger weight for explicit user request
            + fair_bonus
        )

    ranked = sorted(filtered, key=lambda c: c["_cot_score"], reverse=True)

    # ── Step 3: Enforce soft fairness floor ───────────────────────────────────
    prot_min   = math.ceil(COT_MIN_PROT * k)
    prot_count = sum(1 for c in ranked[:k]
                     if c["director_gender"]=="female" or c["region"]=="non-western")
    if prot_count < prot_min:
        prot_extra = [c for c in ranked[k:]
                      if c["director_gender"]=="female" or c["region"]=="non-western"]
        swap_idxs  = [i for i,c in enumerate(ranked[:k])
                      if c["director_gender"]!="female" and c["region"]!="western"]
        for si, pc in zip(swap_idxs, prot_extra):
            if prot_count >= prot_min: break
            ranked[si] = pc; prot_count += 1

    # ── Step 4: Build per-movie CoT steps ─────────────────────────────────────
    results = []
    for rank, c in enumerate(ranked[:k], 1):
        steps = []
        movie_genres = {g.strip() for g in c["genres"].split("|")}
        is_female  = c["director_gender"] == "female"
        is_nonwest = c["region"] == "non-western"

        # Step 1 — user context
        liked_str = ", ".join(profile["liked"]) if profile["liked"] else "general"
        steps.append(
            f"User history: prefers {liked_str} films "
            f"(diversity appetite: {profile.get('diversity','unknown')})."
        )

        # Step 2 — genre exclusion context
        if excluded_genres:
            steps.append(
                f"User excluded: {', '.join(excluded_genres)}. "
                f"This film's genres ({c['genres']}) do not match any excluded genre — kept."
            )

        # Step 3 — request match
        if include_genres and c["_request_match"] > 0:
            matched = movie_genres & {g.title() for g in include_genres}
            steps.append(
                f"Genre match with user request ({', '.join(include_genres)}): "
                f"{', '.join(matched)} — score boosted (+{0.25 * c['_request_match']:.2f})."
            )
        elif include_genres:
            steps.append(
                f"No direct match with requested genres ({', '.join(include_genres)}), "
                f"but included based on relevance and fairness score."
            )

        # Step 4 — history match
        if c["_history_match"] > 0:
            hist_matched = movie_genres & liked_genres
            steps.append(
                f"Matches user's historical preferences: {', '.join(hist_matched)} "
                f"(+{0.1 * c['_history_match']:.2f} history boost)."
            )

        # Step 5 — fairness reasoning
        if is_female or is_nonwest:
            fair_parts = []
            if is_female:  fair_parts.append(f"female-directed ({c['director']})")
            if is_nonwest: fair_parts.append(f"non-western production ({c['region']})")
            steps.append(
                f"Fairness signal: {', '.join(fair_parts)}. "
                f"FA\u2605IR flag was '{c['fairness_flag']}'. "
                f"Diversity bonus applied (+{c['_fair_bonus']:.2f})."
            )
        else:
            steps.append(
                f"Relevance pick: no fairness boost needed at rank #{rank}. "
                f"FA\u2605IR flag: '{c['fairness_flag']}'."
            )

        # Step 6 — final score summary
        steps.append(
            f"Final CoT score: {c['_cot_score']:.3f} "
            f"(base {c['score']:.3f} + genre {0.1*c['_history_match'] + 0.25*c['_request_match']:.3f} "
            f"+ fairness {c['_fair_bonus']:.3f}) → rank #{rank}."
        )

        results.append({
            "movie_idx":       int(c["movie_idx"]),
            "title":           c["title"],
            "director":        c["director"],
            "director_gender": c["director_gender"],
            "region":          c["region"],
            "genres":          c["genres"],
            "fairness_flag":   c["fairness_flag"],
            "cot_score":       round(float(c["_cot_score"]), 4),
            "rank":            rank,
            "steps":           steps,
        })

    return results, excluded_titles


# ── EVALUATION ────────────────────────────────────────────────────────────────
def evaluate(recs, test_df, movies, k=TOP_K):
    gt = test_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    P, R, N = [], [], []
    for u, rl in recs.items():
        if u not in gt: continue
        hits = [1 if m in gt[u] else 0 for m in rl[:k]]
        P.append(sum(hits)/k)
        R.append(sum(hits)/len(gt[u]) if gt[u] else 0)
        dcg  = sum(h/np.log2(i+2) for i,h in enumerate(hits))
        idcg = sum(1/np.log2(i+2) for i in range(min(len(gt[u]),k)))
        N.append(dcg/idcg if idcg else 0)

    def fairness(attr_col, ga, gb):
        mg = movies.set_index("movie_idx")[attr_col].to_dict()
        spd, eod = [], []
        for u, rl in recs.items():
            rs = set(rl)
            ra = sum(1 for m in rs if mg.get(m)==ga)
            rb = sum(1 for m in rs if mg.get(m)==gb)
            if ra+rb: spd.append(ra/(ra+rb) - rb/(ra+rb))
            rel = gt.get(u, set())
            ha  = sum(1 for m in rel if m in rs and mg.get(m)==ga)
            hb  = sum(1 for m in rel if m in rs and mg.get(m)==gb)
            rla = sum(1 for m in rel if mg.get(m)==ga)
            rlb = sum(1 for m in rel if mg.get(m)==gb)
            if rla and rlb: eod.append(ha/rla - hb/rlb)
        ea = [sum(1/np.log2(i+2) for i,m in enumerate(recs.get(u,[])) if mg.get(m)==ga)
              for u in recs]
        eb = [sum(1/np.log2(i+2) for i,m in enumerate(recs.get(u,[])) if mg.get(m)==gb)
              for u in recs]
        return (np.mean(spd) if spd else 0,
                np.mean(eod) if eod else 0,
                round(np.mean(ea)-np.mean(eb), 4))

    spd_g, eod_g, dexp_g = fairness("director_gender", "female", "male")
    spd_r, eod_r, dexp_r = fairness("region", "non-western", "western")
    return {
        "ndcg_at_10": round(np.mean(N),4), "precision_at_10": round(np.mean(P),4),
        "recall_at_10": round(np.mean(R),4), "gender_spd": round(spd_g,4),
        "gender_eod": round(eod_g,4), "region_spd": round(spd_r,4),
        "region_eod": round(eod_r,4), "gender_dexp": dexp_g, "region_dexp": dexp_r,
    }


# ── INTERACTIVE CLI ───────────────────────────────────────────────────────────
def run_interactive_cli(user_idx, excluded_genres=None, include_genres=None,
                        user_message=None, train_df=None, movies=None,
                        cands=None, fair_recs=None, fair_flags=None):
    """
    Run interactive CoT for a single user from the command line.
    Prints per-movie reasoning steps.
    """
    mi = movies.set_index("movie_idx")
    movie_gender = mi["director_gender"].to_dict()
    movie_region = mi["region"].to_dict()

    profile = infer_profile(user_idx, train_df, movies)
    print(f"\nUser #{user_idx} profile:")
    print(f"  Liked genres:       {profile['liked']}")
    print(f"  Era:                {profile['era']}")
    print(f"  Diversity appetite: {profile['diversity']}")

    # Build cand_dicts from FA*IR list
    fl   = fair_recs.get(user_idx, [])
    ff   = fair_flags.get(user_idx, [])
    cand_pool = {m: s for m, s in cands.get(user_idx, [])}

    cand_dicts = []
    for i, m in enumerate(fl):
        if m not in mi.index: continue
        row = mi.loc[m]
        cand_dicts.append({
            "movie_idx":       int(m),
            "title":           str(row.get("title", f"Movie {m}")),
            "genres":          str(row.get("genres", "")),
            "director":        str(row.get("director", "Unknown")),
            "director_gender": str(row.get("director_gender", "unknown")),
            "region":          str(row.get("region", "unknown")),
            "score":           float(cand_pool.get(m, 0.0)),
            "fairness_flag":   ff[i] if i < len(ff) else "relevance",
        })

    if not cand_dicts:
        print("No candidate metadata available for this user.")
        return

    print(f"\nConstraints:")
    print(f"  Excluded genres: {excluded_genres or 'none'}")
    print(f"  Include genres:  {include_genres  or 'none'}")
    if user_message:
        print(f"  User message:    \"{user_message}\"")

    results, excluded_titles = interactive_cot_rerank(
        cand_dicts, profile,
        user_message=user_message,
        excluded_genres=excluded_genres,
        include_genres=include_genres,
    )

    if excluded_titles:
        print(f"\n  Excluded {len(excluded_titles)} film(s): {', '.join(excluded_titles)}")

    print(f"\n{'─'*65}")
    print(f"  CoT Reranked Results for User #{user_idx}")
    print(f"{'─'*65}")
    for r in results:
        print(f"\n  #{r['rank']}. {r['title']}")
        print(f"       {r['director']} · {r['director_gender']} · {r['region']}")
        print(f"       Genres: {r['genres']}")
        print(f"       Tag: {r['fairness_flag']}  |  CoT score: {r['cot_score']}")
        print(f"       Chain of Thought:")
        for i, step in enumerate(r["steps"], 1):
            print(f"         Step {i}: {step}")


# ── BATCH MAIN ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FA-CRS CoT Reranker")
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive single-user mode")
    parser.add_argument("--user",    type=int, default=0,
                        help="User index for interactive mode (default: 0)")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Genres to exclude, e.g. --exclude Horror Action")
    parser.add_argument("--include", nargs="*", default=[],
                        help="Genres to prioritise, e.g. --include Sci-Fi")
    parser.add_argument("--message", type=str, default=None,
                        help="Simulated user message for context")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    use_llm = bool(api_key and ANTHROPIC_AVAILABLE)
    client  = anthropic.Anthropic(api_key=api_key) if use_llm else None
    print(f"Backend: {'LLM (' + LLM_MODEL + ')' if use_llm else 'rule-based'}")

    print("Loading data...")
    pos, movies, n_users, n_movies = load_data()
    train_df, _, test_df = split_data(pos)
    mi           = movies.set_index("movie_idx")
    movie_gender = mi["director_gender"].to_dict()
    movie_region = mi["region"].to_dict()

    print("Loading embeddings...")
    scorer = load_scorer(n_users, n_movies)

    print("Scoring candidates...")
    cands = get_candidates(scorer, train_df, n_users, n_movies, movie_gender, movie_region)

    print("FA*IR reranking...")
    fair_recs, fair_flags = run_fair(cands, movie_gender, movie_region)

    # ── INTERACTIVE MODE ──────────────────────────────────────────────────────
    if args.interactive:
        run_interactive_cli(
            user_idx=args.user,
            excluded_genres=args.exclude or None,
            include_genres=args.include  or None,
            user_message=args.message,
            train_df=train_df, movies=movies,
            cands=cands, fair_recs=fair_recs, fair_flags=fair_flags,
        )
        return

    # ── BATCH MODE ────────────────────────────────────────────────────────────
    print(f"\nCoT reranking on {COT_USERS} users...")
    cot_recs, cot_reasons = {}, {}
    users_to_run = list(fair_recs.keys())[:COT_USERS]

    for u in tqdm(users_to_run):
        profile    = infer_profile(u, train_df, movies)
        fl         = fair_recs.get(u, [])
        ff         = fair_flags.get(u, [])
        cand_pool  = {m: s for m, s in cands.get(u, [])}
        cand_dicts = []
        for i, m in enumerate(fl):
            if m not in mi.index: continue
            row = mi.loc[m]
            cand_dicts.append({
                "movie_idx":       int(m),
                "title":           str(row.get("title", f"Movie {m}")),
                "genres":          str(row.get("genres", "")),
                "director":        str(row.get("director", "Unknown")),
                "director_gender": str(row.get("director_gender", "unknown")),
                "region":          str(row.get("region", "unknown")),
                "score":           float(cand_pool.get(m, 0.0)),
                "fairness_flag":   ff[i] if i < len(ff) else "relevance",
            })
        if not cand_dicts:
            cot_recs[u] = fl; cot_reasons[u] = "no metadata"; continue

        if use_llm:
            ids, reason = llm_cot(cand_dicts, profile, client)
            time.sleep(API_DELAY)
            cot_recs[u] = ids; cot_reasons[u] = reason
        else:
            # Use interactive_cot_rerank even in batch — gives richer per-movie steps
            results, _ = interactive_cot_rerank(cand_dicts, profile)
            cot_recs[u]   = [r["movie_idx"] for r in results]
            cot_reasons[u] = "\n".join(
                f"#{r['rank']} {r['title']}: " + " | ".join(r["steps"])
                for r in results
            )

    subset_fair = {u: fair_recs[u] for u in cot_recs}
    m_fair = evaluate(subset_fair, test_df, movies)
    m_cot  = evaluate(cot_recs,   test_df, movies)

    def load_j(p): return json.load(open(p)) if os.path.exists(p) else {}
    base     = load_j("outputs/baseline/baseline_results.json")
    kg       = load_j("outputs/kg/kg_results.json")
    fair_res = load_j("outputs/fair/fair_results.json")
    fp       = next((r for r in fair_res.get("fut_curve",[]) if r.get("p")==0.3), m_fair)

    print(f"\n{'Metric':<22}{'Baseline':>10}{'KG':>10}{'FA*IR':>10}{'CoT':>10}{'ΔCoT':>8}")
    print("-"*62)
    for label, k_ in [
        ("NDCG@10","ndcg_at_10"),("Precision@10","precision_at_10"),
        ("Recall@10","recall_at_10"),("Gender SPD","gender_spd"),
        ("Gender EOD","gender_eod"),("Region SPD","region_spd"),("Region EOD","region_eod"),
    ]:
        b,kg_,f,c = base.get(k_,0),kg.get(k_,0),fp.get(k_,m_fair.get(k_,0)),m_cot.get(k_,0)
        print(f"{label:<22}{b:>10.4f}{kg_:>10.4f}{f:>10.4f}{c:>10.4f}{c-f:>+8.4f}")
    print(f"\nExposure Gap | Gender: FA*IR={m_fair['gender_dexp']:+.4f} CoT={m_cot['gender_dexp']:+.4f}"
          f" | Region: FA*IR={m_fair['region_dexp']:+.4f} CoT={m_cot['region_dexp']:+.4f}")

    json.dump({"model": f"CoT_{'llm' if use_llm else 'rule'}", **m_cot,
               "fair_metrics": m_fair, "n_users": len(cot_recs)},
              open(f"{OUTPUT_DIR}/cot_results.json","w"), indent=2)

    with open(f"{OUTPUT_DIR}/cot_examples.txt","w",encoding="utf-8") as f:
        for u in list(cot_recs.keys())[:5]:
            profile = infer_profile(u, train_df, movies)
            f.write(f"User {u} | likes: {profile['liked']} | diversity: {profile['diversity']}\n")
            f.write(cot_reasons.get(u,"") + "\n")
            for rank, m in enumerate(cot_recs[u], 1):
                if m in mi.index:
                    f.write(f"  {rank}. {mi.loc[m,'title']} [{mi.loc[m,'director_gender']}, {mi.loc[m,'region']}]\n")
            f.write("\n")

    print(f"\nSaved: {OUTPUT_DIR}/cot_results.json, cot_examples.txt")


if __name__ == "__main__":
    main()