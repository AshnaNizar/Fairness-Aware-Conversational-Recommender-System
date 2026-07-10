import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import LightGCN
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR         = "data"
OUTPUT_DIR       = "outputs/kg"          # same path fair_rerank.py reads from
EMBEDDING_DIM    = 32
NUM_LAYERS       = 2       # 2 is standard for LightGCN; faster than 3, similar quality
LEARNING_RATE    = 1e-3
EPOCHS           = 50
BATCH_SIZE       = 8192
MIN_RATING       = 3       # include 3-star ratings as positive signal (was 4)
TOP_K            = 10
RANDOM_SEED      = 42

# PATH B — dense-subset filtering (the single biggest stability lever)
MIN_USER_RATINGS = 50        # drop sparse users
TOP_N_MOVIES     = 4000      # keep only the most-rated movies

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─── LOAD + DENSE-SUBSET FILTER ───────────────────────────────────────────────
def load_data():
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    movies  = pd.read_csv(os.path.join(DATA_DIR, "movies_enriched.csv"))

    pos = ratings[ratings["rating"] >= MIN_RATING][["user_id", "movie_id"]].copy()
    print(f"Positive interactions (rating>={MIN_RATING}): {len(pos)}")

    # PATH B: keep the TOP_N_MOVIES most-rated movies
    top_movies = pos["movie_id"].value_counts().head(TOP_N_MOVIES).index
    pos = pos[pos["movie_id"].isin(top_movies)].copy()

    # PATH B: keep users with >= MIN_USER_RATINGS (after movie filter)
    uc = pos["user_id"].value_counts()
    keep_users = uc[uc >= MIN_USER_RATINGS].index
    pos = pos[pos["user_id"].isin(keep_users)].copy()
    print(f"After dense-subset filter: {pos['user_id'].nunique()} users, "
          f"{pos['movie_id'].nunique()} movies, {len(pos)} interactions")

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
    for col, d in [("director", "Unknown"), ("director_gender", "unknown"),
                   ("region", "unknown"), ("genres", "Unknown")]:
        movies[col] = movies[col].fillna(d)

    return pos, movies, user2idx, movie2idx, n_users, n_movies


def split_data(pos):
    train_rows, val_rows, test_rows = [], [], []
    for _, g in pos.groupby("user_idx"):
        items = g["movie_idx"].tolist()
        uid   = g["user_idx"].iloc[0]
        if len(items) < 3:
            train_rows += [(uid, m) for m in items]; continue
        n_val = max(1, int(0.1 * len(items)))
        n_test = max(1, int(0.1 * len(items)))
        train_rows += [(uid, m) for m in items[:-(n_val + n_test)]]
        val_rows   += [(uid, m) for m in items[-(n_val + n_test):-n_test]]
        test_rows  += [(uid, m) for m in items[-n_test:]]
    mk = lambda rows: pd.DataFrame(rows, columns=["user_idx", "movie_idx"])
    train_df, val_df, test_df = mk(train_rows), mk(val_rows), mk(test_rows)
    print(f"Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")
    return train_df, val_df, test_df


def build_edge_index(train_df, n_users):
    """Bipartite user<->movie edges; movie node ids offset by n_users."""
    u = torch.tensor(train_df["user_idx"].values, dtype=torch.long)
    m = torch.tensor(train_df["movie_idx"].values + n_users, dtype=torch.long)
    edge_index = torch.stack([torch.cat([u, m]), torch.cat([m, u])], dim=0)
    return edge_index.to(device)


# ─── WRAPPER: exposes (user_emb, movie_emb) like fair_rerank.py expects ────────
class LightGCNRec(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim, num_layers):
        super().__init__()
        self.n_users  = n_users
        self.n_movies = n_movies
        self.num_nodes = n_users + n_movies
        self.lgcn = LightGCN(self.num_nodes, embedding_dim, num_layers)

    def forward(self, edge_index):
        emb = self.lgcn.get_embedding(edge_index)      # PyG propagation (correct)
        return emb[:self.n_users], emb[self.n_users:]

    # convenience for training with PyG's native loss
    def rank(self, edge_index, edge_label_index):
        return self.lgcn(edge_index, edge_label_index)

    def rec_loss(self, pos_rank, neg_rank, node_id):
        return self.lgcn.recommendation_loss(pos_rank, neg_rank, node_id=node_id)


# ─── EVAL (same metric definitions as your pipeline) ──────────────────────────
def get_recommendations(model, edge_index, train_df, n_users, n_movies, top_k):
    model.eval()
    with torch.no_grad():
        user_emb, movie_emb = model(edge_index)
        user_emb, movie_emb = user_emb.cpu(), movie_emb.cpu()
    seen = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    recs = {}
    B = 1024
    for s in range(0, n_users, B):
        e = min(s + B, n_users)
        scores = (user_emb[s:e] @ movie_emb.T).numpy()
        for i, u in enumerate(range(s, e)):
            row = scores[i].copy()
            for m in seen.get(u, set()):
                if m < len(row):
                    row[m] = -np.inf
            top = np.argpartition(row, -top_k)[-top_k:]
            recs[u] = top[np.argsort(row[top])[::-1]].tolist()
    return recs


def precision_recall_ndcg(recs, gt_df, top_k):
    gt = gt_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()
    P, R, N = [], [], []
    for u, rl in recs.items():
        if u not in gt: continue
        actual = gt[u]
        hits = [1 if m in actual else 0 for m in rl[:top_k]]
        P.append(sum(hits) / top_k)
        R.append(sum(hits) / len(actual) if actual else 0)
        dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(actual), top_k)))
        N.append(dcg / idcg if idcg > 0 else 0)
    return np.mean(P), np.mean(R), np.mean(N)


# ─── TRAIN (PyG-native, stable) ───────────────────────────────────────────────
def train_epoch(model, edge_index, train_pos, n_users, n_movies, optimizer,
                batch_size):
    model.train()
    n = train_pos.size(1)

    # Cap the number of batches per epoch so we get many gradient updates without
    # thousands of propagations. ~50 batches/epoch is plenty and stays fast.
    MAX_BATCHES = 20
    eff_batch = max(batch_size, (n + MAX_BATCHES - 1) // MAX_BATCHES)

    perm = torch.randperm(n, device=edge_index.device)
    total, nb = 0.0, 0

    for s in range(0, n, eff_batch):
        idx = perm[s:s + eff_batch]
        src     = train_pos[0, idx]
        pos_dst = train_pos[1, idx]
        neg_dst = torch.randint(0, n_movies, (idx.numel(),),
                                device=edge_index.device) + n_users

        # Combined pos+neg edge_label_index; one forward = one propagation.
        pos_eli = torch.stack([src, pos_dst], dim=0)
        neg_eli = torch.stack([src, neg_dst], dim=0)
        edge_label_index = torch.cat([pos_eli, neg_eli], dim=1)

        optimizer.zero_grad()
        rank = model.lgcn(edge_index, edge_label_index)          # PyG forward
        pos_rank, neg_rank = rank.chunk(2)
        loss = model.lgcn.recommendation_loss(
            pos_rank, neg_rank, node_id=edge_label_index.unique())
        loss.backward()
        optimizer.step()

        total += float(loss); nb += 1

    return total / max(nb, 1)

def main():
    pos, movies, u2i, m2i, n_users, n_movies = load_data()
    train_df, val_df, test_df = split_data(pos)
    edge_index = build_edge_index(train_df, n_users)

    # positive edges as [2, N] with movie offset (for training)
    train_pos = torch.stack([
        torch.tensor(train_df["user_idx"].values, dtype=torch.long),
        torch.tensor(train_df["movie_idx"].values + n_users, dtype=torch.long),
    ], dim=0).to(device)

    model = LightGCNRec(n_users, n_movies, EMBEDDING_DIM, NUM_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nTraining PyG-native LightGCN...")
    # Early stopping: stop if val NDCG doesn't improve for PATIENCE consecutive
    # evaluations. With eval_every=5, PATIENCE=3 means "stop after 15 epochs
    # with no improvement." Prevents the overfitting drift we saw (peak at
    # epoch 5, then steady decline for 45 more epochs).
    PATIENCE = 3
    best_ndcg, best_epoch = 0.0, 0
    evals_since_improve = 0
    for epoch in tqdm(range(1, EPOCHS + 1)):
        loss = train_epoch(model, edge_index, train_pos, n_users, n_movies,
                           optimizer, BATCH_SIZE)
        if epoch % 5 == 0 or epoch == EPOCHS:
            recs = get_recommendations(model, edge_index, train_df, n_users, n_movies, TOP_K)
            p, r, nd = precision_recall_ndcg(recs, val_df, TOP_K)
            tqdm.write(f"epoch {epoch:3d} | loss {loss:.4f} | "
                       f"val P@10 {p:.4f} R@10 {r:.4f} NDCG@10 {nd:.4f}")
            if nd > best_ndcg:
                best_ndcg, best_epoch = nd, epoch
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model_kg.pt"))
                evals_since_improve = 0
            else:
                evals_since_improve += 1
                if evals_since_improve >= PATIENCE:
                    tqdm.write(f"Early stopping at epoch {epoch}: "
                               f"no improvement for {PATIENCE} evals "
                               f"(best NDCG {best_ndcg:.4f} at epoch {best_epoch}).")
                    break
    print(f"\nBest val NDCG@10: {best_ndcg:.4f} at epoch {best_epoch}")

    # Test
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model_kg.pt")))
    recs = get_recommendations(model, edge_index, train_df, n_users, n_movies, TOP_K)
    p, r, nd = precision_recall_ndcg(recs, test_df, TOP_K)
    print(f"\n--- Test ---\nP@10 {p:.4f}  R@10 {r:.4f}  NDCG@10 {nd:.4f}")

    # Save final embedding matrices directly — fair_rerank.py / api_server load
    # these instead of rebuilding the model (avoids state_dict key mismatch).
    model.eval()
    with torch.no_grad():
        ue, me = model(edge_index)
    torch.save({"user_emb": ue.cpu(), "movie_emb": me.cpu()},
               os.path.join(OUTPUT_DIR, "user_movie_emb.pt"))

    # Save the filtered movie/id maps so fair_rerank.py + api_server use the SAME subset
    movies.to_csv(os.path.join(OUTPUT_DIR, "movies_subset.csv"), index=False)
    json.dump({"n_users": n_users, "n_movies": n_movies,
               "embedding_dim": EMBEDDING_DIM, "num_layers": NUM_LAYERS},
              open(os.path.join(OUTPUT_DIR, "model_meta.json"), "w"), indent=2)
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_subset.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_subset.csv"), index=False)
    print(f"\nSaved model + subset maps to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
