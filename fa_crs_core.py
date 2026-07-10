import math
import random
import numpy as np
from scipy.stats import binom


# ─── REAL FA*IR: BINOMIAL SIGNIFICANCE TEST ────────────────────────────────

def min_protected(k, p, alpha=0.1):
    return int(binom.ppf(alpha, k, p))


def fair_rerank(candidates, movie_attr, protected_val, p, k=10, alpha=0.1):
    protected   = sorted([(m, s) for m, s in candidates if movie_attr.get(m) == protected_val],
                         key=lambda x: x[1], reverse=True)
    unprotected = sorted([(m, s) for m, s in candidates if movie_attr.get(m) != protected_val],
                         key=lambda x: x[1], reverse=True)

    result, flags = [], []
    pi, ui = 0, 0
    for pos in range(k):
        n_protected_so_far = sum(flags)  # count of protected items placed (True or already counted)
        needed = min_protected(pos + 1, p, alpha)
        if n_protected_so_far < needed and pi < len(protected):
            result.append(protected[pi][0]); flags.append(True); pi += 1
        else:
            take_prot = (pi < len(protected) and
                         (ui >= len(unprotected) or protected[pi][1] >= unprotected[ui][1]))
            if take_prot:
                result.append(protected[pi][0]); flags.append(False); pi += 1
            elif ui < len(unprotected):
                result.append(unprotected[ui][0]); flags.append(False); ui += 1
            else:
                break
        if len(result) == k:
            break
    return result, flags


# ─── EXPOSURE-AWARE DIVERSIFIED INJECTION ──────────────────────────────────

class ExposureTracker:

    def __init__(self, decay=0.85):
        self.counts = {}   # movie_idx -> times injected
        self.decay = decay  # optional: multiply all counts by this periodically to let old exposure fade

    def penalty(self, movie_idx):
        return self.counts.get(movie_idx, 0)

    def record(self, movie_ids):
        for m in movie_ids:
            self.counts[m] = self.counts.get(m, 0) + 1

    def decay_all(self):
        self.counts = {m: c * self.decay for m, c in self.counts.items() if c * self.decay > 0.01}


def diversified_injection(pool_scores, seen, group_ids, n_inject, tracker, sample_from_top=30, seed=None):

    rng = random.Random(seed)
    candidates = [(m, pool_scores[m]) for m in group_ids
                  if m not in seen and m in pool_scores]
    if not candidates:
        return []

    # Take a wider slice than before (top `sample_from_top`), not just top-10
    candidates.sort(key=lambda x: x[1], reverse=True)
    wide_slice = candidates[:sample_from_top]

    # Penalize items proportional to how often they've already been injected
    weighted = []
    for m, s in wide_slice:
        exposure_penalty = tracker.penalty(m)
        weight = 1.0 / (1.0 + exposure_penalty)   # more exposure -> lower weight
        weighted.append((m, s, weight))

    # Weighted sample without replacement, favoring high score but rotating
    # away from overexposed items
    chosen = []
    remaining = weighted[:]
    for _ in range(min(n_inject, len(remaining))):
        total_w = sum(w for _, _, w in remaining)
        if total_w <= 0:
            break
        r = rng.uniform(0, total_w)
        acc = 0.0
        for i, (m, s, w) in enumerate(remaining):
            acc += w
            if acc >= r:
                chosen.append(m)
                remaining.pop(i)
                break

    tracker.record(chosen)
    return chosen


# ─── COT HELPERS (ported from cot_rerank.py, LLM backend unified) ─────────

def infer_profile(user_idx, train_df, movies):
    import re
    seen = movies[movies["movie_idx"].isin(train_df[train_df["user_idx"] == user_idx]["movie_idx"])]
    gc = {}
    for gs in seen["genres"].fillna(""):
        for g in gs.split("|"):
            gc[g.strip()] = gc.get(g.strip(), 0) + 1
    liked = sorted(gc, key=gc.get, reverse=True)[:3]
    div = ((seen["region"] == "non-western").sum() + (seen["director_gender"] == "female").sum()) / max(len(seen), 1)
    return {"liked": liked, "diversity": "high" if div > .15 else "medium" if div > .05 else "low"}


def rule_based_cot(cand_dicts, profile, k=10, cot_min_prot=0.25):
    liked = set(profile["liked"])
    for c in cand_dicts:
        match = len({g.strip() for g in c["genres"].split("|")} & liked)
        bonus = 0.15 * (c["director_gender"] == "female") + 0.15 * (c["region"] == "non-western") \
            if profile["diversity"] != "low" else 0
        c["_cot_score"] = c["score"] + 0.1 * match + bonus
    ranked = sorted(cand_dicts, key=lambda c: c["_cot_score"], reverse=True)
    prot_min = math.ceil(cot_min_prot * k)
    prot_count = sum(1 for c in ranked[:k] if c["director_gender"] == "female" or c["region"] == "non-western")
    if prot_count < prot_min:
        prot_extra = [c for c in ranked[k:] if c["director_gender"] == "female" or c["region"] == "non-western"]
        swap_idxs = [i for i, c in enumerate(ranked[:k]) if c["director_gender"] != "female" and c["region"] != "non-western"]
        for si, pc in zip(swap_idxs, prot_extra):
            if prot_count >= prot_min:
                break
            ranked[si] = pc; prot_count += 1
    steps = [f"#{i+1} {c['title']} — " +
             ("diversity pick (" + c["director_gender"] + ", " + c["region"] + ")"
              if c["director_gender"] == "female" or c["region"] == "non-western" else "relevance pick") +
             f", genres: {c['genres']}"
             for i, c in enumerate(ranked[:k])]
    return [c["movie_idx"] for c in ranked[:k]], steps


def call_llm(prompt, client, model="claude-sonnet-4-6", max_tokens=600):

    if client is None:
        return None  # caller should fall back to rule-based logic
    try:
        resp = client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text
    except Exception as e:
        return None


def llm_cot(cand_dicts, profile, client, k=10, cot_min_prot=0.25, model="claude-sonnet-4-6"):

    lines = [f"{i+1}. {c['title']} | {c['genres']} | {c['director']} ({c['director_gender']}) | {c['region']} | score:{c['score']:.3f}"
             + (" [fairness-promoted]" if c.get("fairness_flag", "relevance") != "relevance" else "")
             for i, c in enumerate(cand_dicts)]
    prompt = f"""Re-rank these {k} movies for a user who likes {', '.join(profile['liked'])} (diversity appetite: {profile['diversity']}).

Candidates:
{chr(10).join(lines)}

Think step by step, then output:
REASONING: <brief per-item reasoning>
FINAL_RANKING:
1. <title>
...{k}. <title>

Constraint: at least {math.ceil(cot_min_prot*k)} items must be female-directed OR non-western."""

    text = call_llm(prompt, client, model=model)
    if text is None:
        ids, steps = rule_based_cot(cand_dicts, profile, k, cot_min_prot)
        return ids, "\n".join(steps) + "\n\n(LLM unavailable — used rule-based fallback.)"

    import re as _re
    title2idx = {c["title"]: c["movie_idx"] for c in cand_dicts}
    final_ids = []
    if "FINAL_RANKING:" in text:
        for line in text.split("FINAL_RANKING:")[-1].strip().split("\n"):
            line = _re.sub(r"^\d+\.\s*", "", line.strip())
            for title, idx in title2idx.items():
                if title.lower() in line.lower() and idx not in final_ids:
                    final_ids.append(idx); break
            if len(final_ids) == k:
                break
    for c in cand_dicts:
        if len(final_ids) == k:
            break
        if c["movie_idx"] not in final_ids:
            final_ids.append(c["movie_idx"])
    return final_ids[:k], text
