# Fairness-Aware Conversational Recommender System (FA-CRS)

Group project for the course **Human-Centered Artificial Intelligence**.

FA-CRS is a movie recommender that combines graph-based collaborative filtering
(LightGCN) with **FA\*IR post-hoc reranking** to reduce provider-side bias along
two protected, item-side axes:

- **director gender** — under-representation of female-directed films
- **production region** — under-representation of non-western cinema

Recommendations are delivered through a **conversational React interface** backed
by a **FastAPI** server. Every recommendation carries a chain-of-thought (CoT)
explanation grounded in the actual reranking decision (relevance score, rank
movement, and the fairness metrics it affected), and the UI shows live per-list
and cohort fairness metrics.

> **Dataset note:** this system runs on a **dense ~4,000-movie / ~91,700-user
> subset of MovieLens-25M**, enriched with TMDB metadata (director gender,
> production country/region). The embedding dimension is 32, with 2 LightGCN
> propagation layers, and the primary fairness operating point is **p = 0.30**
> (α = 0.15).

---

## Important: large files are NOT in this repository

GitHub rejects files above its size limit, so the **trained model weights and
embedding tensors are `.gitignore`d** and are **not** in this repo. You must
**regenerate them locally** (one command — see
[Regenerating the large artifacts](#regenerating-the-large-artifacts-required)).

Files intentionally excluded (see `.gitignore`):

| Excluded file | What it is | How it comes back |
|---|---|---|
| `outputs/kg/best_model_kg.pt` | Trained LightGCN weights | run `lightgcn_pyg.py` |
| `outputs/kg/user_movie_emb.pt` | Exported user + movie embeddings (the file the reranker and API actually load) | run `lightgcn_pyg.py` |
| `outputs/kg/kg_graph.pt` | Cached PyG graph tensor | run `lightgcn_pyg.py` |
| `outputs/kg/train_subset.csv` | Processed user–movie training interactions used to construct the LightGCN training graph | run `lightgcn_pyg.py` |
| `data/` (raw ratings, TMDB cache) | Raw MovieLens + enrichment cache | run `data_prep.py` (or drop in your own `ratings.csv`) |
| `frontend/node_modules`, `frontend/dist` | Node dependencies and production build files | `npm install` in `frontend/` |


**What *is* committed** (small enough for GitHub) and can be used directly:
`movies_subset.csv`, `test_subset.csv`, `fair_results.json`,
`baseline_results.json`, `kg_results.json`, `fut_curve.json`, `model_meta.json`,
`results_table.txt`, and the figure PNGs.

---

## Repository structure

```
├── data_prep.py            # enrich MovieLens with TMDB metadata + fairness attributes
├── lightgcn_pyg.py         # train LightGCN (PyG) → writes the .pt embeddings + subset CSVs
├── fair_rerank.py          # offline FA*IR joint reranking + full metric sweep (p = 0.1–0.5)
├── fa_crs_core.py          # shared core: candidate injection, joint reranker, CoT helpers
├── metrics_extended.py     # rND, exposure gap, gini, coverage, collapse rate
├── results.py              # generate paper figures + comparison table
├── debug_fair.py           # diagnostics: verify candidate pool + protected-group injection
├── api_server.py     # FastAPI backend (loads trained artifacts, serves recs + metrics + CoT)
├── frontend/               # React + Vite conversational UI
│   ├── App.jsx
│   ├── main.jsx
│   ├── index.css / index.html / vite_config.js
│   └── package.json
│
├── movies_subset.csv       # committed: catalogue subset with fairness attributes
├── test_subset.csv         # committed: held-out test split
├── *_results.json, fut_curve.json, model_meta.json, results_table.txt   # committed results
└── outputs/                # created at runtime (holds the regenerated .pt files)
    └── kg/
```

---

## How the pipeline works

**1. Data preparation (`data_prep.py`)** — Loads the MovieLens ratings, filters
to a dense subset, and enriches each film via TMDB (director, production
country). Director gender is inferred and production country is mapped to
**western / non-western**. Produces the enriched catalogue used everywhere
downstream.

**2. Train LightGCN (`lightgcn_pyg.py`)** — PyTorch-Geometric-native LightGCN
(Adam, lr 1e-3, 2 layers, embedding dim 32), trained with BPR over uniformly
sampled negatives, with L2 applied to the **base** embedding table. **This is
the step that produces the large `.pt` files** and the aligned subset splits
(`user_movie_emb.pt`, `train_subset.csv`, `test_subset.csv`, `movies_subset.csv`,
`best_model_kg.pt`, `model_meta.json`). The embeddings are then frozen.

**3. FA\*IR reranking + metric sweep (`fair_rerank.py`)** — Loads the frozen
embeddings, builds each user's candidate pool (top-scored films **plus**
exposure-aware injection of protected films), then applies a **joint single-pass
FA\*IR reranker** that services the gender and region quotas together. A fairness
strength `p` is swept over {0.1, 0.2, 0.3, 0.4, 0.5} to trace the
Fairness–Utility Tradeoff (FUT) curve. Writes `fair_results.json`,
`baseline_results.json`, `fut_curve.json`, `results_table.txt`.

**4. Figures + table (`results.py`)** — Reads the JSON results and produces the
paper-ready comparison table and figures (`bias_comparison.png`,
`accuracy_comparison.png`, `fut_curve_combined.png`, `collapse_curve.png`).

**5. Interactive system (`api_server.py` + `frontend/`)** — The FastAPI
server loads the frozen embeddings and serves live recommendations, reranked at
the same `p`/`α` as the offline results (read from `fair_results.json`). It
computes all 15 metrics live, generates metrics-grounded CoT explanations, and
the React app renders the conversation, the recommendation lists, and the
per-list / cohort metric panels.

---

## Installation

Python 3.9+ and Node 18+ recommended.

```bash
# Python (backend + pipeline)
pip install torch torch-geometric pandas numpy scipy scikit-learn tqdm \
            requests gender-guesser matplotlib fastapi uvicorn

# Node (frontend)
cd frontend && npm install && cd ..
```

For GPU training, install the CUDA build of PyTorch from
[pytorch.org](https://pytorch.org) first. Training was done on a Colab Tesla T4.

---

## Regenerating the large artifacts (REQUIRED)

Because the `.pt` files are not in the repo, run this **once** after cloning. The
reviewer needs the raw ratings for training; two paths are supported.

### Path A — you have (or want to build) the full dataset

```bash
# 1. Provide raw MovieLens ratings at data/ratings.csv
#    (and add a TMDB API key in data_prep.py to build movies_enriched.csv)
python data_prep.py          # writes data/movies_enriched.csv (+ resumable TMDB cache)

# 2. Train LightGCN — THIS REGENERATES THE LARGE .pt FILES
python lightgcn_pyg.py       # writes outputs/kg/user_movie_emb.pt, best_model_kg.pt,
                             #        train_subset.csv, test_subset.csv,
                             #        movies_subset.csv, model_meta.json

# 3. (optional) reproduce the offline fairness results/tables
python fair_rerank.py
python results.py
```

### Path B — quickest route to a runnable demo

The committed `movies_subset.csv` and `test_subset.csv` are already aligned to
the trained embeddings. You only need to regenerate the embeddings themselves:

```bash
# Point lightgcn_pyg.py at your ratings.csv, then:
python lightgcn_pyg.py
```

This writes everything the API server needs into `outputs/kg/`. **No `.pkl`
candidate-pool file is required** — the server and reranker build candidate pools
in memory on startup/per request.

### Verifying the artifacts

After training you should have:

```
outputs/kg/
├── user_movie_emb.pt      # ← the file the API/reranker load (required)
├── best_model_kg.pt
├── train_subset.csv
├── test_subset.csv
├── movies_subset.csv
└── model_meta.json        # {"n_users": 91691, "n_movies": 4000, "embedding_dim": 32, "num_layers": 2}
```

If any are missing, re-run `lightgcn_pyg.py`. The API server will refuse to start
and print a `FileNotFoundError` naming the missing artifact if the embeddings
aren't present.

---

## Running the interactive app

Two terminals.

**Terminal 1 — FastAPI backend:**

```bash
# ensure outputs/kg/ contains the regenerated .pt files first
python -m uvicorn api_server:app --reload --port 7860
```

The server searches `outputs/kg/` and the current directory for artifacts. If
your files live elsewhere, set `KG_OUTPUT_DIR=/path/to/outputs/kg`.

On startup it prints the active configuration and warms up the cohort metrics,
e.g.:

```
Ready. 91691 users, 4000 movies. p=0.3, alpha=0.15, backend=rule-based (no LLM)
[warmup] seeded 200 users ... NDCG@10=0.011 Precision@10=... Recall@10=...
```

**Terminal 2 — React frontend:**

```bash
cd frontend
npm run dev        # serves on http://localhost:3000
```

Open http://localhost:3000. The app calls the backend on port 7860.

### Conversational LLM backend (optional, three tiers)

The chat intent-parsing and CoT explanations use, in priority order:

1. **Anthropic** — if `ANTHROPIC_API_KEY` is set (`export ANTHROPIC_API_KEY=...`).
2. **Ollama** — if a local Ollama server is reachable (default
   `http://localhost:11434`, model `llama3`). Configurable via `OLLAMA_URL` /
   `OLLAMA_MODEL`.
3. **Rule-based** — deterministic fallback requiring no LLM at all.

The rule-based tier is fully functional and **recommended for reproducible
grading** — the fairness pipeline is identical regardless of tier; only the
phrasing of explanations changes. Check the active tier at
`GET http://localhost:7860/health`.

Useful env vars: `ACCURACY_WARMUP_USERS` (default 200; set 0 to disable the
cohort warm-up), `KG_OUTPUT_DIR`, `ANTHROPIC_MODEL`, `OLLAMA_URL`,
`OLLAMA_MODEL`.

---

## Evaluation metrics

The system reports **15 metrics**, split by the granularity at which they are
meaningful.

**Per-list** (well-defined for a single user's top-10, recomputed per request):

| Metric | Meaning |
|---|---|
| Gender / Region **SPD** | Statistical Parity Difference — representation gap between groups (primary fairness metric) |
| Gender / Region **EOD** | Equal Opportunity Difference — true-positive-rate gap (null when a group has no test ground truth) |
| Gender / Region **rND** | Normalized rank-aware group distribution |
| Gender / Region **Exposure Gap** | Rank-discounted exposure difference between groups |

**Cohort** (only meaningful across users; accumulated over the session +
startup warm-up):

| Metric | Meaning |
|---|---|
| **NDCG@10 / Precision@10 / Recall@10** | Accuracy vs held-out test set — reported as a session average because per-user values are near-binary on sparse data |
| Gender / Region **Collapse Rate** | Fraction of protected slots filled by recycled items (**supply-bounded**, near 1.0 by design) |
| **Gini Exposure** | Inequality of exposure across the catalogue |
| **Catalog Coverage** | Fraction of the catalogue ever surfaced |

Lower absolute SPD/EOD/exposure-gap = fairer. NDCG/Precision/Recall higher =
better. Catalogue supply in the subset is ~5.4% female-directed and ~2.5%
non-western, which bounds what fairness targets can physically achieve.

---

## Key design decisions

- **Frozen embeddings + post-hoc reranking.** LightGCN is trained once and
  frozen; fairness is enforced by FA\*IR reranking on top. Continuously
  retraining would risk re-encoding the very biases FA\*IR corrects.
- **Joint single-pass FA\*IR.** Gender and region quotas are enforced together
  in one pass. An earlier sequential (gender-then-region) design starved the
  region axis; the joint reranker fixes this.
- **Knowledge graph as a metadata layer, not a GNN component.** An earlier
  version added typed director/gender/region/genre nodes to the graph. Because
  LightGCN aggregation is a plain degree-normalized mean with no
  relation-specific transformation, a handful of extreme-degree hub nodes
  (e.g. the western-region node adjacent to ~3,775 films) acted as *smoothing
  sinks* that erased rather than sharpened protected-group distinctions, with no
  measurable fairness gain. The KG is therefore retained as the enriched
  attribute store the reranker queries — not as graph nodes.
- **Exposure-aware protected injection.** Each user's candidate pool is seeded
  with top-scoring unseen female-directed and non-western films so the reranker
  always has protected items available to promote.
- **Collapse rate is supply-bounded, not an algorithmic failure.** With ~2.5%
  non-western supply and a p = 0.30 target, the reranker must fill thousands of
  protected slots from ~98 films, so a collapse rate near 1.0 is close to the
  theoretical floor. This is reported as a finding about supply scarcity.

---

## Primary results (p = 0.30, subset)

| | No-rerank | FA\*IR | Δ |
|---|---|---|---|
| Gender SPD | −0.7995 | −0.6994 | **+0.1001** |
| Region SPD | −1.0000 | −0.5975 | **+0.4024** |
| NDCG@10 | 0.0127 | 0.0114 | −0.0013 |

Region SPD improves most (the baseline top-10 showed essentially no non-western
films), at a negligible accuracy cost. See `results_table.txt` and the figures
for the full 15-metric table and the FUT curve.

---

## Troubleshooting

- **`FileNotFoundError: user_movie_emb.pt` on server start** → run
  `lightgcn_pyg.py` to regenerate the embeddings (see
  [Regenerating the large artifacts](#regenerating-the-large-artifacts-required)).
- **CoT/chat feels slow or shows a spinner** → you're likely on the Ollama tier
  with a cold model; the first call warms it up, or set no LLM to use the instant
  rule-based tier.
- **Metrics show 0.0000 for NDCG/Precision/Recall** → these are cohort averages;
  with very few users served they can be 0. The startup warm-up
  (`ACCURACY_WARMUP_USERS`) pre-seeds them so they read a stable ~0.011.
- **`torch_geometric` import error when running only the API** → the server
  stubs PyG if it's absent, but the training script (`lightgcn_pyg.py`) requires
  it; install `torch-geometric` to (re)train.
```