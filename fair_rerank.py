import os
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LightGCN
from torch_geometric.utils import structured_negative_sampling
from tqdm import tqdm

# fa_crs_core.py and metrics_extended.py must sit in the same directory.
from fa_crs_core import fair_rerank as core_fair_rerank, ExposureTracker, diversified_injection, min_protected
from metrics_extended import evaluate_extended

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATA_DIR      = "data"
KG_OUTPUT_DIR = "outputs/kg"
OUTPUT_DIR    = "outputs/fair"
EMBEDDING_DIM = 64      
NUM_LAYERS    = 3       
MIN_RATING    = 3       
TOP_K         = 10
RANDOM_SEED   = 42

# FA*IR parameters
# p: minimum proportion of protected group in top-K

FAIR_P_VALUES = [0.10, 0.20, 0.30, 0.40, 0.50] 
EVAL_USERS = 8000
FAIR_ALPHA    = 0.15  
RERANK_DEPTH  = 10    

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

EXPOSURE_TRACKER = ExposureTracker()

# ─── LOAD DATA ────────────────────────────────────────────────────────────────


class EmbeddingHolder:
    def __init__(self, user_emb, movie_emb):
        self.user_emb  = user_emb
        self.movie_emb = movie_emb
    def eval(self):
        return self
    def __call__(self, edge_index=None):
        return self.user_emb, self.movie_emb


def load_data():

    _subset  = os.path.join(KG_OUTPUT_DIR, "movies_subset.csv")
    _train_p = os.path.join(KG_OUTPUT_DIR, "train_subset.csv")

    if os.path.exists(_subset) and os.path.exists(_train_p):
        # Trained-subset path: trust the saved indices, do NOT remap.
        movies = pd.read_csv(_subset)
        movies = movies.dropna(subset=["movie_idx"]).copy()
        movies["movie_idx"] = movies["movie_idx"].astype(int)
        for col, d in [("director", "Unknown Director"),
                       ("director_gender", "unknown"),
                       ("region", "unknown"), ("genres", "Unknown")]:
            if col in movies.columns:
                movies[col] = movies[col].fillna(d)

        train = pd.read_csv(_train_p)
        n_users  = int(train["user_idx"].max()) + 1
        n_movies = int(movies["movie_idx"].max()) + 1
        # pos/user2idx/movie2idx are not needed downstream when subset files exist
        # (main() loads train/test subsets directly), so return minimal stand-ins.
        pos = train[["user_idx", "movie_idx"]].copy()
        return pos, movies, {}, {}, n_users, n_movies

    # ── Fallback: no subset files, rebuild from the full catalogue ──
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    movies  = pd.read_csv(os.path.join(DATA_DIR, "movies_enriched.csv"))

    pos = ratings[ratings["rating"] >= MIN_RATING][["user_id", "movie_id"]].copy()
    user_counts  = pos["user_id"].value_counts()
    active_users = user_counts[user_counts >= 20].index
    pos = pos[pos["user_id"].isin(active_users)].copy()

    user_ids  = sorted(pos["user_id"].unique())
    movie_ids = sorted(pos["movie_id"].unique())
    user2idx  = {u: i for i, u in enumerate(user_ids)}
    movie2idx = {m: i for i, m in enumerate(movie_ids)}
    pos["user_idx"]  = pos["user_id"].map(user2idx)
    pos["movie_idx"] = pos["movie_id"].map(movie2idx)
    n_users, n_movies = len(user_ids), len(movie_ids)

    movies["movie_idx"] = movies["movie_id"].map(movie2idx)
    movies = movies.dropna(subset=["movie_idx"]).copy()
    movies["movie_idx"] = movies["movie_idx"].astype(int)
    for col, d in [("director", "Unknown Director"), ("director_gender", "unknown"),
                   ("region", "unknown"), ("genres", "Unknown")]:
        movies[col] = movies[col].fillna(d)

    return pos, movies, user2idx, movie2idx, n_users, n_movies

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


# ─── REBUILD KG (same as Day 7-9) ─────────────────────────────────────────────

def build_kg(train_df, movies, n_users, n_movies):
    directors    = sorted(movies["director"].unique())
    dir2idx      = {d: i for i, d in enumerate(directors)}
    n_directors  = len(directors)
    dir_offset   = n_users + n_movies

    genders       = ["female", "male", "unknown"]
    gender2idx    = {g: i for i, g in enumerate(genders)}
    n_genders     = len(genders)
    gender_offset = dir_offset + n_directors

    regions       = ["western", "non-western", "unknown"]
    region2idx    = {r: i for i, r in enumerate(regions)}
    n_regions     = len(regions)
    region_offset = gender_offset + n_genders

    all_genres = set()
    for g in movies["genres"]:
        for genre in g.split("|"):
            all_genres.add(genre.strip())
    genres       = sorted(all_genres)
    genre2idx    = {g: i for i, g in enumerate(genres)}
    n_genres     = len(genres)
    genre_offset = region_offset + n_regions

    u_idx    = torch.tensor(train_df["user_idx"].values, dtype=torch.long)
    m_idx    = torch.tensor(train_df["movie_idx"].values + n_users, dtype=torch.long)
    um_src   = torch.cat([u_idx, m_idx])
    um_dst   = torch.cat([m_idx, u_idx])

    movie_nodes = torch.tensor(movies["movie_idx"].values + n_users, dtype=torch.long)
    dir_nodes   = torch.tensor([dir2idx[d] + dir_offset for d in movies["director"]], dtype=torch.long)
    md_src = torch.cat([movie_nodes, dir_nodes])
    md_dst = torch.cat([dir_nodes, movie_nodes])

    dir_nodes_g = torch.tensor([dir2idx[d] + dir_offset for d in movies["director"]], dtype=torch.long)
    gen_nodes   = torch.tensor([gender2idx[g] + gender_offset for g in movies["director_gender"]], dtype=torch.long)
    dg_src = torch.cat([dir_nodes_g, gen_nodes])
    dg_dst = torch.cat([gen_nodes, dir_nodes_g])

    reg_nodes = torch.tensor([region2idx[r] + region_offset for r in movies["region"]], dtype=torch.long)
    mr_src = torch.cat([movie_nodes, reg_nodes])
    mr_dst = torch.cat([reg_nodes, movie_nodes])

    mg_srcs, mg_dsts = [], []
    for _, row in movies.iterrows():
        m_node = int(row["movie_idx"]) + n_users
        for genre in row["genres"].split("|"):
            genre = genre.strip()
            if genre in genre2idx:
                g_node = genre2idx[genre] + genre_offset
                mg_srcs.extend([m_node, g_node])
                mg_dsts.extend([g_node, m_node])

    mg_src = torch.tensor(mg_srcs, dtype=torch.long)
    mg_dst = torch.tensor(mg_dsts, dtype=torch.long)

    all_src    = torch.cat([um_src, md_src, dg_src, mr_src, mg_src])
    all_dst    = torch.cat([um_dst, md_dst, dg_dst, mr_dst, mg_dst])
    edge_index = torch.stack([all_src, all_dst], dim=0).to(device)

    n_total = n_users + n_movies + n_directors + n_genders + n_regions + n_genres
    return edge_index, n_total


# ─── LIGHTGCN MODEL (same as Day 7-9) ────────────────────────────────────────

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


# ─── GET RAW SCORES ───────────────────────────────────────────────────────────

def get_scores(model, edge_index, train_df, n_users, n_movies,
               candidate_k=50, movie_gender=None, movie_region=None):

    model.eval()
    with torch.no_grad():
        user_emb, movie_emb = model(edge_index)

    seen          = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    female_movies = set(m for m, g in movie_gender.items() if g == "female")     if movie_gender else set()
    nonwestern_movies = set(m for m, r in movie_region.items() if r == "non-western") if movie_region else set()

    candidates = {}
    SCORE_BATCH = 1000
    # FIX: optionally cap the number of users scored/reranked for speed.
    eval_n = n_users if EVAL_USERS is None else min(EVAL_USERS, n_users)
    for start in range(0, eval_n, SCORE_BATCH):
        end = min(start + SCORE_BATCH, eval_n)
        batch_scores = torch.matmul(user_emb[start:end], movie_emb.T).cpu().numpy()

        for i, u in enumerate(range(start, end)):
            seen_u = seen.get(u, set())
            s = batch_scores[i].copy()
            for m in seen_u:
                if m < len(s):
                    s[m] = -np.inf

            top  = np.argpartition(s, -candidate_k)[-candidate_k:]
            top  = top[np.argsort(s[top])[::-1]]
            pool = set(top.tolist())

            female_scores     = {m: float(s[m]) for m in female_movies if m < len(s)}
            nonwestern_scores = {m: float(s[m]) for m in nonwestern_movies if m < len(s)}
            pool.update(diversified_injection(female_scores, seen_u, female_movies,
                                               n_inject=10, tracker=EXPOSURE_TRACKER,
                                               sample_from_top=30, seed=u))
            pool.update(diversified_injection(nonwestern_scores, seen_u, nonwestern_movies,
                                               n_inject=10, tracker=EXPOSURE_TRACKER,
                                               sample_from_top=30, seed=u))

            pool_list = sorted(pool, key=lambda m: s[m] if s[m] > -1e8 else -1e9, reverse=True)
            candidates[u] = [(int(m), float(s[m])) for m in pool_list]

    return candidates   # ← single correct return



def _joint_rerank(cands, movie_gender, movie_region, p_g, p_r, k, alpha):
    
    # Split by protected status per attribute
    def bucket(attr, protected_val):
        prot = sorted([(m, s) for m, s in cands if attr.get(m) == protected_val],
                      key=lambda x: x[1], reverse=True)
        unpr = sorted([(m, s) for m, s in cands if attr.get(m) != protected_val],
                      key=lambda x: x[1], reverse=True)
        return prot, unpr

    g_prot, _ = bucket(movie_gender, "female")
    r_prot, _ = bucket(movie_region, "non-western")
    all_sorted = sorted(cands, key=lambda x: x[1], reverse=True)

    result, flags = [], []
    used = set()
    g_placed = 0        # actual female-directed films placed so far
    r_placed = 0        # actual non-western films placed so far
    g_i, r_i = 0, 0     # next protected film to consider per axis
    a_i = 0             # next best-by-score film to consider

    for pos in range(k):
        need_g = min_protected(pos + 1, p_g, alpha)
        need_r = min_protected(pos + 1, p_r, alpha)

        placed = False
        # Priority: region first if BOTH under-served (it has smaller supply here).
        if r_placed < need_r:
            while r_i < len(r_prot) and r_prot[r_i][0] in used:
                r_i += 1
            if r_i < len(r_prot):
                m = r_prot[r_i][0]
                result.append(m); flags.append("region")
                used.add(m); r_placed += 1; r_i += 1
                if movie_gender.get(m) == "female":
                    g_placed += 1
                placed = True

        if not placed and g_placed < need_g:
            while g_i < len(g_prot) and g_prot[g_i][0] in used:
                g_i += 1
            if g_i < len(g_prot):
                m = g_prot[g_i][0]
                result.append(m); flags.append("gender")
                used.add(m); g_placed += 1; g_i += 1
                if movie_region.get(m) == "non-western":
                    r_placed += 1
                placed = True

        if not placed:
            # Fill with best remaining item by score
            while a_i < len(all_sorted) and all_sorted[a_i][0] in used:
                a_i += 1
            if a_i >= len(all_sorted):
                break
            m = all_sorted[a_i][0]
            result.append(m); flags.append("relevance")
            used.add(m); a_i += 1
            if movie_gender.get(m) == "female":  g_placed += 1
            if movie_region.get(m) == "non-western": r_placed += 1

    return result, flags


def rerank_all_users(candidates, movie_gender, movie_region, p_gender, p_region):

    recs, flags = {}, {}
    for u, cands in candidates.items():
        rec, fl = _joint_rerank(cands, movie_gender, movie_region,
                                p_gender, p_region, k=TOP_K, alpha=FAIR_ALPHA)
        recs[u]  = rec
        flags[u] = fl
    return recs, flags


def precision_recall_ndcg(recs, ground_truth_df):
    gt = ground_truth_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    precisions, recalls, ndcgs = [], [], []
    for u, rec_list in recs.items():
        if u not in gt:
            continue
        actual = gt[u]
        hits   = [1 if m in actual else 0 for m in rec_list[:TOP_K]]
        precision = sum(hits) / TOP_K
        recall    = sum(hits) / len(actual) if actual else 0
        dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(actual), TOP_K)))
        ndcg = dcg / idcg if idcg > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)


def compute_fairness_metrics(recs, test_df, movies, attribute_col, group_a, group_b):
    movie_group = movies.set_index("movie_idx")[attribute_col].to_dict()
    gt          = test_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    spd_list, eod_list = [], []
    for u, rec_list in recs.items():
        rec_set = set(rec_list)
        rec_a   = sum(1 for m in rec_set if movie_group.get(m) == group_a)
        rec_b   = sum(1 for m in rec_set if movie_group.get(m) == group_b)
        total   = rec_a + rec_b
        if total == 0:
            continue
        spd_list.append(rec_a / total - rec_b / total)
        relevant = gt.get(u, set())
        rel_a = sum(1 for m in relevant if movie_group.get(m) == group_a)
        rel_b = sum(1 for m in relevant if movie_group.get(m) == group_b)
        hit_a = sum(1 for m in relevant if m in rec_set and movie_group.get(m) == group_a)
        hit_b = sum(1 for m in relevant if m in rec_set and movie_group.get(m) == group_b)
        tpr_a = hit_a / rel_a if rel_a > 0 else None
        tpr_b = hit_b / rel_b if rel_b > 0 else None
        if tpr_a is not None and tpr_b is not None:
            eod_list.append(tpr_a - tpr_b)
    return (np.mean(spd_list) if spd_list else 0.0,
            np.mean(eod_list) if eod_list else 0.0)


# ─── EXPLANATION MODULE ───────────────────────────────────────────────────────

def generate_explanation(movie_row, flag, rank):
    """
    Template-based explanation for a single recommendation.
    flag: 'relevance', 'gender', or 'region'
    """
    title  = movie_row.get("title", "This movie")
    director = movie_row.get("director", "the director")
    genres = movie_row.get("genres", "").replace("|", ", ")
    gender = movie_row.get("director_gender", "unknown")
    region = movie_row.get("region", "unknown")

    if flag == "relevance":
        return (f"#{rank} {title} — Recommended based on your viewing history. "
                f"Genre: {genres}. Directed by {director}.")

    elif flag == "gender":
        return (f"#{rank} {title} — Highlighted to support gender diversity in recommendations. "
                f"Directed by {director} ({gender}-directed). Genre: {genres}.")

    elif flag == "region":
        return (f"#{rank} {title} — Highlighted to support geographic diversity. "
                f"This is a {region} production directed by {director}. Genre: {genres}.")

    return f"#{rank} {title}"


def generate_user_explanations(user_idx, rec_list, flags, movies_indexed):
    lines = [f"Recommendations for User {user_idx}:", "-" * 50]
    for rank, (movie_idx, flag) in enumerate(zip(rec_list, flags), 1):
        if movie_idx in movies_indexed.index:
            row = movies_indexed.loc[movie_idx]
            row_dict = row.to_dict() if hasattr(row, "to_dict") else {}
        else:
            row_dict = {}
        lines.append(generate_explanation(row_dict, flag, rank))
    return "\n".join(lines)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # Load data
    pos, movies, user2idx, movie2idx, n_users, n_movies = load_data()

    _train_p = os.path.join(KG_OUTPUT_DIR, "train_subset.csv")
    _test_p  = os.path.join(KG_OUTPUT_DIR, "test_subset.csv")
    if os.path.exists(_train_p) and os.path.exists(_test_p):
        train_df = pd.read_csv(_train_p)
        test_df  = pd.read_csv(_test_p)
        val_df   = test_df   # not used downstream for the FUT curve
        print("Loaded train/test subset written by lightgcn_pyg.py")
    else:
        train_df, val_df, test_df = split_data(pos)

    emb_path = os.path.join(KG_OUTPUT_DIR, "user_movie_emb.pt")
    _emb = torch.load(emb_path, map_location=device)
    model = EmbeddingHolder(_emb["user_emb"].to(device), _emb["movie_emb"].to(device))
    edge_index = None  # unused: embeddings are precomputed
    print(f"Loaded trained embeddings: users={_emb['user_emb'].shape}, "
          f"movies={_emb['movie_emb'].shape}")

    # Movie attribute lookups (must be defined before get_scores)
    movie_gender   = movies.set_index("movie_idx")["director_gender"].to_dict()
    movie_region   = movies.set_index("movie_idx")["region"].to_dict()
    movies_indexed = movies.set_index("movie_idx")

    # Get candidate scores — injects protected movies into pool
    print("Scoring candidates (expanding pool with protected group movies)...")
    candidates = get_scores(model, edge_index, train_df, n_users, n_movies,
                            candidate_k=50,
                            movie_gender=movie_gender,
                            movie_region=movie_region)

    print("\nEvaluating no-rerank baseline (same candidates, top-10 by score)...")
    norerank_recs = {u: [m for m, _ in cands[:TOP_K]] for u, cands in candidates.items()}
    nb_prec, nb_rec, nb_ndcg = precision_recall_ndcg(norerank_recs, test_df)
    nb_spd_g, nb_eod_g = compute_fairness_metrics(norerank_recs, test_df, movies,
                                                  "director_gender", "female", "male")
    nb_spd_r, nb_eod_r = compute_fairness_metrics(norerank_recs, test_df, movies,
                                                  "region", "non-western", "western")
    nb_ext_g = evaluate_extended(norerank_recs, movies, "director_gender", "female",
                                 n_movies, k=TOP_K)
    nb_ext_r = evaluate_extended(norerank_recs, movies, "region", "non-western",
                                 n_movies, k=TOP_K)
    norerank_metrics = {
        "ndcg_at_10": nb_ndcg, "precision_at_10": nb_prec, "recall_at_10": nb_rec,
        "gender_spd": nb_spd_g, "gender_eod": nb_eod_g,
        "region_spd": nb_spd_r, "region_eod": nb_eod_r,
        "gender_rND": nb_ext_g.get("rND"), "gender_exposure_gap": nb_ext_g.get("exposure_gap"),
        "gender_collapse_rate": nb_ext_g.get("collapse_rate"),
        "region_rND": nb_ext_r.get("rND"), "region_exposure_gap": nb_ext_r.get("exposure_gap"),
        "region_collapse_rate": nb_ext_r.get("collapse_rate"),
        "gini_exposure": nb_ext_g.get("gini_exposure"), "catalog_coverage": nb_ext_g.get("catalog_coverage"),
    }
    print(f"  no-rerank | NDCG: {nb_ndcg:.4f} | G-SPD: {nb_spd_g:.4f} | R-SPD: {nb_spd_r:.4f}")

    print("\nRunning FA*IR reranking across p values...")
    fut_curve = []

    for p in FAIR_P_VALUES:
        recs, flags = rerank_all_users(
            candidates, movie_gender, movie_region,
            p_gender=p, p_region=p
        )
        prec, rec, ndcg = precision_recall_ndcg(recs, test_df)
        spd_g, eod_g    = compute_fairness_metrics(recs, test_df, movies, "director_gender", "female", "male")
        spd_r, eod_r    = compute_fairness_metrics(recs, test_df, movies, "region", "non-western", "western")

        # ── Extended metrics: rND, exposure gap, gini, coverage, collapse rate ──
        ext_gender = evaluate_extended(recs, movies, "director_gender", "female", n_movies, k=TOP_K)
        ext_region = evaluate_extended(recs, movies, "region", "non-western", n_movies, k=TOP_K)

        fut_curve.append({
            "p":             p,
            "ndcg_at_10":    round(ndcg, 4),
            "precision_at_10": round(prec, 4),
            "recall_at_10":  round(rec, 4),
            "gender_spd":    round(spd_g, 4),
            "gender_eod":    round(eod_g, 4),
            "region_spd":    round(spd_r, 4),
            "region_eod":    round(eod_r, 4),
            "gender_rND":            ext_gender["rND"],
            "gender_exposure_gap":   ext_gender["exposure_gap"],
            "gender_collapse_rate":  ext_gender["collapse_rate"],
            "gender_collapse_top":   ext_gender["collapse_top_items"],
            "region_rND":            ext_region["rND"],
            "region_exposure_gap":   ext_region["exposure_gap"],
            "region_collapse_rate":  ext_region["collapse_rate"],
            "region_collapse_top":   ext_region["collapse_top_items"],
            "gini_exposure":         ext_gender["gini_exposure"],
            "catalog_coverage":      ext_gender["catalog_coverage"],
        })

        print(f"p={p:.2f} | NDCG: {ndcg:.4f} | G-SPD: {spd_g:.4f} | R-SPD: {spd_r:.4f} | "
              f"G-rND: {ext_gender['rND']:.4f} | R-rND: {ext_region['rND']:.4f} | "
              f"G-Collapse: {ext_gender['collapse_rate']:.4f} | R-Collapse: {ext_region['collapse_rate']:.4f}")

    # Save FUT curve
    with open(os.path.join(OUTPUT_DIR, "fut_curve.json"), "w") as f:
        json.dump(fut_curve, f, indent=2)
    print(f"\nFUT curve saved.")

    # ── Primary result: pick a p that ACTUALLY EXISTS in the sweep ──
    primary_p = FAIR_P_VALUES[len(FAIR_P_VALUES) // 2]
    primary   = next((r for r in fut_curve if r["p"] == primary_p), fut_curve[0])
    primary_p = primary["p"]

    # ── Full comparison table ──
    print(f"\n{'='*70}")
    print(f"COMPARISON: LightGCN vs LightGCN+FA*IR  (same run, p={primary_p})")
    print(f"{'='*70}")
    print(f"{'Metric':<24} {'No-Rerank':>12} {'FA*IR':>12} {'Δ Fair':>12}")
    print("-" * 64)

    metric_keys = [
        ("NDCG@10",              "ndcg_at_10"),
        ("Precision@10",         "precision_at_10"),
        ("Recall@10",            "recall_at_10"),
        ("Gender SPD",           "gender_spd"),
        ("Gender EOD",           "gender_eod"),
        ("Region SPD",           "region_spd"),
        ("Region EOD",           "region_eod"),
        ("Gender rND",           "gender_rND"),
        ("Gender Exposure Gap",  "gender_exposure_gap"),
        ("Gender Collapse Rate", "gender_collapse_rate"),
        ("Region rND",           "region_rND"),
        ("Region Exposure Gap",  "region_exposure_gap"),
        ("Region Collapse Rate", "region_collapse_rate"),
        ("Gini Exposure",        "gini_exposure"),
        ("Catalog Coverage",     "catalog_coverage"),
    ]

    comparison = {}
    for label, key in metric_keys:
        nb = norerank_metrics.get(key)
        fv = primary.get(key)
        if nb is None or fv is None:
            continue
        delta = fv - nb
        print(f"{label:<24} {nb:>12.4f} {fv:>12.4f} {delta:>+12.4f}")
        comparison[label] = {"no_rerank": nb, "fair": fv, "delta_fair": delta}

    print("-" * 64)
    print("Both columns: PyG LightGCN, dense 10k-movie subset, identical")
    print("candidates and test split. Only difference: FA*IR reranking.")

    # Save full comparison
    full_results = {
        "primary_p":        primary_p,
        "fair_alpha":       FAIR_ALPHA,
        "rerank_depth":     RERANK_DEPTH,
        "norerank_metrics": norerank_metrics,
        "comparison_table": comparison,
        "fut_curve":        fut_curve,
    }
    with open(os.path.join(OUTPUT_DIR, "fair_results.json"), "w") as f:
        json.dump(full_results, f, indent=2)

    # (Old Table 2 removed: it compared against stale full-dataset JSONs.
    # All extended metrics now appear in the unified two-system table above.)

    # ── Generate example explanations for 5 users ──
    print(f"\n{'='*70}")
    print("EXAMPLE RECOMMENDATIONS WITH EXPLANATIONS")
    print(f"{'='*70}")

    # Re-run FA*IR at p=0.3 to get flags for explanations
    recs, flags = rerank_all_users(
        candidates, movie_gender, movie_region,
        p_gender=primary_p, p_region=primary_p
    )

    explanation_lines = []
    for u in list(recs.keys())[:5]:
        block = generate_user_explanations(u, recs[u], flags[u], movies_indexed)
        print(block)
        print()
        explanation_lines.append(block)
        explanation_lines.append("")

    with open(os.path.join(OUTPUT_DIR, "example_recommendations.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(explanation_lines))

    print(f"All outputs saved to {OUTPUT_DIR}/")
    print("\nNext step: Day 13-15 write-up. Use fair_results.json for your results table.")


if __name__ == "__main__":
    main()
