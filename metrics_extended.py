import numpy as np
from collections import Counter


# ─── DISPARITY (rank-aware) ────────────────────────────────────────────────

def rND(recs, movie_group, protected_val, k=10):

    scores = []
    for u, ranked in recs.items():
        ranked = ranked[:k]
        n = len(ranked)
        if n == 0:
            continue
        full_prop = sum(1 for m in ranked if movie_group.get(m) == protected_val) / n

        z = sum(1 / np.log2(i + 2) for i in range(n))  # normalizer
        acc = 0.0
        for i in range(1, n + 1):
            prefix = ranked[:i]
            prefix_prop = sum(1 for m in prefix if movie_group.get(m) == protected_val) / i
            acc += abs(prefix_prop - full_prop) / np.log2(i + 1) if i > 1 else abs(prefix_prop - full_prop)
        scores.append(acc / z if z > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


# ─── EXPOSURE GAP ───────────────────────────────────────────────────────────

def exposure_gap(recs, movie_group, protected_val, k=10):

    exp_protected, exp_unprotected = 0.0, 0.0
    n_protected, n_unprotected = 0, 0

    for u, ranked in recs.items():
        for rank, m in enumerate(ranked[:k], start=1):
            e = 1.0 / np.log2(rank + 1)
            if movie_group.get(m) == protected_val:
                exp_protected += e
                n_protected += 1
            else:
                exp_unprotected += e
                n_unprotected += 1

    avg_protected = exp_protected / n_protected if n_protected > 0 else 0.0
    avg_unprotected = exp_unprotected / n_unprotected if n_unprotected > 0 else 0.0
    return avg_protected - avg_unprotected, avg_protected, avg_unprotected


# ─── INEQUALITY ─────────────────────────────────────────────────────────────

def gini_exposure(recs, k=10):

    counts = Counter(m for ranked in recs.values() for m in ranked[:k])
    if not counts:
        return 0.0
    vals = np.sort(np.array(list(counts.values()), dtype=np.float64))
    n = len(vals)
    cum = np.cumsum(vals)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def catalog_coverage(recs, n_movies, k=10):

    unique_items = set(m for ranked in recs.values() for m in ranked[:k])
    return len(unique_items) / n_movies if n_movies > 0 else 0.0


# ─── COLLAPSE RATE ──────────────────────────────────────────────────────────

def collapse_rate(recs, movie_group, protected_val, k=10):

    protected_slots = []
    for u, ranked in recs.items():
        for m in ranked[:k]:
            if movie_group.get(m) == protected_val:
                protected_slots.append(m)

    total_slots = len(protected_slots)
    if total_slots == 0:
        return 0.0, []

    counts = Counter(protected_slots)
    unique_items = len(counts)
    rate = 1 - (unique_items / total_slots)
    top_items = counts.most_common(5)
    return rate, top_items


# ─── COMBINED WRAPPER ───────────────────────────────────────────────────────

def evaluate_extended(recs, movies, attribute_col, protected_val, n_movies, k=10):

    movie_group = movies.set_index("movie_idx")[attribute_col].to_dict()

    rnd_score = rND(recs, movie_group, protected_val, k=k)
    exp_gap, exp_protected, exp_unprotected = exposure_gap(recs, movie_group, protected_val, k=k)
    gini = gini_exposure(recs, k=k)
    coverage = catalog_coverage(recs, n_movies, k=k)
    coll_rate, coll_top_items = collapse_rate(recs, movie_group, protected_val, k=k)

    return {
        "rND":                round(rnd_score, 4),
        "exposure_gap":       round(exp_gap, 4),
        "exposure_protected": round(exp_protected, 4),
        "exposure_unprotected": round(exp_unprotected, 4),
        "gini_exposure":      round(gini, 4),
        "catalog_coverage":   round(coverage, 4),
        "collapse_rate":      round(coll_rate, 4),
        "collapse_top_items": coll_top_items,  # [(movie_idx, count), ...] — look these titles up for the writeup
    }
