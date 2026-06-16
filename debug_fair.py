"""
Debug script - run this in your project folder to diagnose FA*IR issue.
python debug_fair.py
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import LightGCN

DATA_DIR      = "data"
EMBEDDING_DIM = 64
NUM_LAYERS    = 3
MIN_RATING    = 4

# ── Load data ──
ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
movies  = pd.read_csv(f"{DATA_DIR}/movies_enriched.csv")

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
movies["director_gender"] = movies["director_gender"].fillna("unknown")
movies["region"]          = movies["region"].fillna("unknown")

# ── Stats ──
print("=== MOVIE METADATA STATS ===")
print("Director gender distribution:")
print(movies["director_gender"].value_counts())
print()
print("Region distribution:")
print(movies["region"].value_counts())
print()

female_movies     = set(movies[movies["director_gender"] == "female"]["movie_idx"].tolist())
nonwestern_movies = set(movies[movies["region"] == "non-western"]["movie_idx"].tolist())
print(f"Female-directed movie indices in dataset: {len(female_movies)}")
print(f"Non-western movie indices in dataset:     {len(nonwestern_movies)}")
print()

# ── Check what's in a sample candidate pool ──
print("=== SAMPLE CANDIDATE POOL CHECK (User 1) ===")
# Simulate a fake uniform score just to check pool logic
n = n_movies
fake_scores = np.random.rand(n)

seen_u = set(pos[pos["user_idx"] == 1]["movie_idx"].tolist())
for m in seen_u:
    if m < len(fake_scores):
        fake_scores[m] = -np.inf

top50 = np.argpartition(fake_scores, -50)[-50:]
top50_set = set(top50.tolist())

female_in_pool = top50_set & female_movies
nw_in_pool     = top50_set & nonwestern_movies

print(f"Top-50 candidates: {len(top50_set)}")
print(f"Female-directed in top-50: {len(female_in_pool)}")
print(f"Non-western in top-50:     {len(nw_in_pool)}")
print()

# ── Check if injection would work ──
female_unseen = [m for m in female_movies if m not in seen_u and m < n]
nw_unseen     = [m for m in nonwestern_movies if m not in seen_u and m < n]
print(f"Female-directed movies available to inject: {len(female_unseen)}")
print(f"Non-western movies available to inject:     {len(nw_unseen)}")
print()

# ── Check reranking logic directly ──
print("=== FA*IR MINIMUM COUNT TABLE ===")
from scipy.stats import binom

def min_protected(k, p, alpha=0.1):
    for m in range(k + 1):
        if 1 - binom.cdf(m - 1, k, p) >= alpha:
            return m
    return k

print(f"{'k':>4} | {'p=0.1':>6} | {'p=0.3':>6} | {'p=0.5':>6}")
print("-" * 30)
for k in range(1, 11):
    print(f"{k:>4} | {min_protected(k,0.1):>6} | {min_protected(k,0.3):>6} | {min_protected(k,0.5):>6}")
print()
print("If all counts above are 0, FA*IR never forces a protected item.")
print("This would explain why p values have no effect.")
