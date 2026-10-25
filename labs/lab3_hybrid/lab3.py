# %% [markdown]
# # Lab 3 — Hybrid search, and why RRF beats blending scores
#
# **30 minutes.** You will:
#
# 1. Run dense and BM25 separately and find, **in the data**, which queries each one wins
# 2. Implement linear score interpolation and tune `alpha` until it beats both
# 3. Multiply BM25's scores by 100 and watch your tuned `alpha` become worthless
# 4. Implement `rrf()` — five lines, no constants to fit
# 5. Re-run the ×100 test and confirm RRF doesn't move
#
# Same corpus as Lab 2 (SciFact), same cached embeddings, same metrics. The
# numbers here continue directly from the ones you measured there.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "wslib").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from wslib import corpus as corpus_mod
from wslib import embed, fusion, indexes, metrics, sparse
from wslib.timing import stopwatch

CANDIDATES = 100      # depth retrieved from each retriever before fusing
K = 10                # depth we score at

c = corpus_mod.load("scifact")
docs = embed.cached_corpus_embeddings(c)
qids, qvecs = embed.cached_query_embeddings(c)
jq = c.judged_queries()
qtexts = [jq[q] for q in qids]

print(c.summary())

with stopwatch("build dense index"):
    dense = indexes.ExactIndex(docs)
with stopwatch("build BM25 index"):
    bm25 = sparse.BM25Index(c.texts)
print(bm25.describe())

# %% [markdown]
# ## 1 · Retrieve once, fuse many times
#
# We're about to try a dozen fusion settings. Re-running retrieval each time
# would waste the whole lab, so pull the candidates **once** and keep them.
# (This is also how you'd structure it in production: retrieval is the expensive
# part, fusion is arithmetic.)

# %%
with stopwatch(f"dense: top-{CANDIDATES} for {len(qids)} queries"):
    dense_res = [
        (dense.search(qvecs[i][None, :], CANDIDATES).ids[0],
         dense.search(qvecs[i][None, :], CANDIDATES).scores[0])
        for i in range(len(qids))
    ]

with stopwatch(f"bm25:  top-{CANDIDATES} for {len(qids)} queries"):
    bm25_res = [bm25.search(t, CANDIDATES) for t in qtexts]

print(f"\ndense returns {np.mean([len(r[0]) for r in dense_res]):.0f} candidates/query")
print(f"bm25  returns {np.mean([len(r[0]) for r in bm25_res]):.0f} candidates/query "
      f"(fewer — many documents share no query terms at all)")

# %% [markdown]
# ### Look at one query side by side
#
# The score columns are the thing to notice. Cosine sits in `[0, 1]`. BM25 is
# unbounded. Nothing about these two numbers is comparable.

# %%
i = 3
qid = qids[i]
print(f"query [{qid}]: {qtexts[i]}")
gold = {d for d, g in c.qrels[qid].items() if g > 0}
print(f"relevant: {sorted(gold)}\n")

print(f"{'':4} {'DENSE':<52} {'score':>7}   {'BM25':<52} {'score':>7}")
for r in range(5):
    d_id, d_sc = c.doc_ids[dense_res[i][0][r]], dense_res[i][1][r]
    b_id, b_sc = c.doc_ids[bm25_res[i][0][r]], bm25_res[i][1][r]
    d_txt = ("* " if d_id in gold else "  ") + c.titles[dense_res[i][0][r]][:48]
    b_txt = ("* " if b_id in gold else "  ") + c.titles[bm25_res[i][0][r]][:48]
    print(f"{r + 1:>3}. {d_txt:<52} {d_sc:>7.3f}   {b_txt:<52} {b_sc:>7.3f}")
print("\n(* marks a judged-relevant document)")

# %% [markdown]
# ## 2 · Neither retriever wins. They win different queries.
#
# The slides asserted this. Let's prove it on all 300 queries.

# %%
run_dense = {qids[i]: c.ids(dense_res[i][0][:K]) for i in range(len(qids))}
run_bm25 = {qids[i]: c.ids(bm25_res[i][0][:K]) for i in range(len(qids))}

print("dense :", {k: round(v, 4) for k, v in
                  metrics.evaluate(run_dense, c.qrels, ks=(K,)).items()})
print("bm25  :", {k: round(v, 4) for k, v in
                  metrics.evaluate(run_bm25, c.qrels, ks=(K,)).items()})

nd_dense = metrics.per_query(run_dense, c.qrels, "ndcg", K)
nd_bm25 = metrics.per_query(run_bm25, c.qrels, "ndcg", K)

diffs = sorted(((nd_dense[q] - nd_bm25[q], q) for q in nd_dense), reverse=True)
wins_d = sum(1 for d, _ in diffs if d > 1e-9)
wins_b = sum(1 for d, _ in diffs if d < -1e-9)
print(f"\ndense wins {wins_d} queries · bm25 wins {wins_b} · "
      f"tie on {len(diffs) - wins_d - wins_b}")

# %%
print("=" * 78)
print("QUERIES WHERE DENSE WINS BIG  (paraphrase — few shared terms)")
print("=" * 78)
for d, q in diffs[:3]:
    print(f"  Δ nDCG +{d:.3f}   {jq[q][:95]}")

print("\n" + "=" * 78)
print("QUERIES WHERE BM25 WINS BIG  (distinctive terminology)")
print("=" * 78)
for d, q in diffs[-3:]:
    print(f"  Δ nDCG {d:+.3f}   {jq[q][:95]}")

# %% [markdown]
# **That split is the whole argument.** Roughly a third of queries go each way.
# You cannot know in advance which kind a user will type, so you cannot pick one
# retriever — you have to run both and combine them well.
#
# "Well" is the hard part, and the next two sections are about how easy it is to
# get wrong.

# %% [markdown]
# ## 3 · TASK: fuse by score (the tempting way)
#
# The instinct is one line: `score = alpha * dense + (1 - alpha) * bm25`.
#
# Implement it. Two decisions you're forced to make — note both as you go:
# what weight, and what score to give a document one retriever never returned.

# %%
def my_linear_blend(results, weights, top_k=None):
    """Weighted sum of raw scores.

    results : list of (ids, scores) — one pair per retriever
    weights : list of floats, same length as results
    returns : list of doc row-indices, best first

    A document missing from a retriever's list contributes nothing from it.
    """
    # YOUR CODE HERE — accumulate weight * score per document id, then sort.
    raise NotImplementedError("implement my_linear_blend")


# %% [markdown]
# ### CHECKPOINT 1

# %%
_a = (np.array([1, 2, 3]), np.array([0.9, 0.5, 0.1]))
_b = (np.array([3, 1, 9]), np.array([20.0, 8.0, 2.0]))

try:
    got = list(my_linear_blend([_a, _b], [0.5, 0.5]))
    want = list(fusion.linear_blend([_a, _b], [0.5, 0.5]))
    assert got == want, f"expected {want}, got {got}"
    print("CP1 PASS — using your implementation")
    linear_blend = my_linear_blend
except NotImplementedError:
    print("CP1 — not implemented yet, using the reference")
    linear_blend = fusion.linear_blend
except AssertionError as e:
    print(f"CP1 FAIL — {e}\n   using the reference")
    linear_blend = fusion.linear_blend

# %% [markdown]
# ### Tune alpha
#
# Sweep the weight on dense and find the best nDCG@10. This is the step every
# hybrid-search blog post tells you to do.

# %%
def eval_fusion(fuse_fn, **kw):
    """Apply a fusion function to every query and score the result."""
    run = {}
    for i, qid in enumerate(qids):
        fused = fuse_fn([dense_res[i], bm25_res[i]], **kw)
        run[qid] = c.ids(list(fused)[:K])
    return metrics.evaluate(run, c.qrels, ks=(K,)), run


ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
alpha_rows = []
for a in ALPHAS:
    s, _ = eval_fusion(linear_blend, weights=[a, 1 - a])
    alpha_rows.append((a, s[f"ndcg@{K}"], s[f"recall@{K}"]))
    print(f"  alpha={a:<5} ndcg@{K} {s[f'ndcg@{K}']:.4f}  recall@{K} {s[f'recall@{K}']:.4f}")

best_alpha, best_ndcg, _ = max(alpha_rows, key=lambda r: r[1])
print(f"\nbest alpha = {best_alpha}  ->  nDCG@{K} {best_ndcg:.4f}")

# %% [markdown]
# Notice how *lopsided* the optimum is. That isn't a statement about how much
# dense retrieval matters — it's compensation for BM25 scores being an order of
# magnitude larger. **The weight is a unit conversion pretending to be a
# preference.**

# %% [markdown]
# ## 4 · Break it: multiply BM25 scores by 100
#
# Nothing about retrieval quality changes. BM25 returns the same documents in the
# same order. Only the units move — as they would if you changed corpus, switched
# tokenizer, or upgraded the embedding model.

# %%
bm25_x100 = [fusion.scale_scores(r, 100.0) for r in bm25_res]


def eval_scaled(fuse_fn, **kw):
    run = {}
    for i, qid in enumerate(qids):
        fused = fuse_fn([dense_res[i], bm25_x100[i]], **kw)
        run[qid] = c.ids(list(fused)[:K])
    return metrics.evaluate(run, c.qrels, ks=(K,))


before, _ = eval_fusion(linear_blend, weights=[best_alpha, 1 - best_alpha])
after = eval_scaled(linear_blend, weights=[best_alpha, 1 - best_alpha])

print(f"linear blend at your tuned alpha={best_alpha}")
print(f"  before x100 : ndcg@{K} {before[f'ndcg@{K}']:.4f}")
print(f"  after  x100 : ndcg@{K} {after[f'ndcg@{K}']:.4f}")
print(f"  change      : {after[f'ndcg@{K}'] - before[f'ndcg@{K}']:+.4f}")

# and what alpha would you need now?
rescued = []
for a in [0.9, 0.99, 0.999, 0.9999, 1.0]:
    s = eval_scaled(linear_blend, weights=[a, 1 - a])
    rescued.append((a, s[f"ndcg@{K}"]))
print("\nto recover, you would have to re-tune:")
for a, nd in rescued:
    print(f"  alpha={a:<8} ndcg@{K} {nd:.4f}")

# %% [markdown]
# **This is the trap.** Your tuned constant was fitted to one particular pair of
# score distributions. Change either and the constant is silently wrong — no
# error, no crash, just worse search until someone notices.
#
# Min-max normalisation is the usual fix. It helps, and it's still per-query and
# outlier-sensitive:

# %%
norm_before, _ = eval_fusion(linear_blend, weights=[0.5, 0.5], normalize=True)
norm_after = eval_scaled(linear_blend, weights=[0.5, 0.5], normalize=True)
print(f"min-max normalised, alpha=0.5")
print(f"  before x100 : ndcg@{K} {norm_before[f'ndcg@{K}']:.4f}")
print(f"  after  x100 : ndcg@{K} {norm_after[f'ndcg@{K}']:.4f}   "
      f"(scale-invariant, but still needs a weight)")

# %% [markdown]
# ## 5 · TASK: fuse by rank instead
#
# $$\mathrm{RRF}(d) = \sum_{r \in R} \frac{1}{k + \mathrm{rank}_r(d)}$$
#
# Five lines. No weights, no normalisation, no decision about missing documents —
# a document absent from a list simply contributes nothing from that list.

# %%
def my_rrf(results, k_const=60, top_k=None):
    """Reciprocal Rank Fusion.

    results : list of (ids, scores) — the scores are there and you must IGNORE them
    returns : list of doc row-indices, best first
    """
    # YOUR CODE HERE — accumulate 1/(k_const + rank) per document, 1-based rank.
    raise NotImplementedError("implement my_rrf")


# %% [markdown]
# ### CHECKPOINT 2 — arithmetic you can check by hand
#
# Doc 1 is at rank 1 in list A and rank 2 in list B, so it should score
# `1/(60+1) + 1/(60+2)`. Doc 3 is rank 3 then rank 1: `1/63 + 1/61`.

# %%
try:
    got = list(my_rrf([_a, _b]))
    want = list(fusion.rrf([_a, _b]))
    assert got == want, f"expected order {want}, got {got}"
    print("CP2 PASS — using your implementation")
    rrf = my_rrf
except NotImplementedError:
    print("CP2 — not implemented yet, using the reference")
    rrf = fusion.rrf
except AssertionError as e:
    print(f"CP2 FAIL — {e}\n   using the reference")
    rrf = fusion.rrf

_ids, _sc = fusion.rrf_with_scores([_a, _b])
print(f"\n  doc {_ids[0]} scores {_sc[0]:.5f}   hand-check 1/61 + 1/62 = "
      f"{1 / 61 + 1 / 62:.5f}")
print(f"  doc {_ids[1]} scores {_sc[1]:.5f}   hand-check 1/63 + 1/61 = "
      f"{1 / 63 + 1 / 61:.5f}")

# %% [markdown]
# ## 6 · The same ×100 test, on RRF

# %%
rrf_before, run_rrf = eval_fusion(rrf)
rrf_after = eval_scaled(rrf)

print(f"RRF (k=60), no tuning at all")
print(f"  before x100 : ndcg@{K} {rrf_before[f'ndcg@{K}']:.4f}")
print(f"  after  x100 : ndcg@{K} {rrf_after[f'ndcg@{K}']:.4f}")
print(f"  change      : {rrf_after[f'ndcg@{K}'] - rrf_before[f'ndcg@{K}']:+.4f}")

identical = all(
    list(rrf([dense_res[i], bm25_res[i]]))[:K]
    == list(rrf([dense_res[i], bm25_x100[i]]))[:K]
    for i in range(len(qids))
)
print(f"\nevery query returned a byte-identical ranking: {identical}")
print("RRF never looked at the scores, so there was nothing for the scale to break.")

# %% [markdown]
# ## 7 · Does k matter? Measure it — don't trust the default
#
# Every article on RRF quotes `k=60` from the original paper. Let's check it on
# *our* corpus rather than taking it on faith.

# %%
k_rows = []
for kc in (1, 2, 5, 10, 20, 60, 200, 1000):
    s, _ = eval_fusion(rrf, k_const=kc)
    k_rows.append((kc, s[f"ndcg@{K}"], s[f"recall@{K}"]))
    print(f"  k={kc:<5} ndcg@{K} {s[f'ndcg@{K}']:.4f}  recall@{K} {s[f'recall@{K}']:.4f}")

best_k, best_k_ndcg, _ = max(k_rows, key=lambda r: r[1])
print(f"\nbest k on this corpus = {best_k}  (ndcg@{K} {best_k_ndcg:.4f})")
print(f"the published default k=60 gives {dict((r[0], r[1]) for r in k_rows)[60]:.4f}")

# %% [markdown]
# **That is not flat, and the default is not optimal here.** Small `k` wins, by
# a meaningful margin.
#
# There's a mechanism behind it. `k` controls how much the head of each list is
# damped: large `k` makes rank 1 and rank 5 nearly equal in value, deliberately
# refusing to trust any single retriever's confidence. That's the right instinct
# when several documents are relevant and you want breadth.
#
# But SciFact has **~1.1 relevant documents per query**. There is usually exactly
# one right answer, and when a retriever puts it at rank 1 that confidence is
# precisely the signal you want to keep. Damping it throws away the information
# that mattered.
#
# So: `k=60` is a reasonable default for corpora with several relevant documents
# per query. **On a single-answer corpus, turn it down.** This is the workshop's
# whole thesis arriving in miniature — the received default was worth 2 points of
# nDCG, and you only know because you measured.

# %% [markdown]
# ## 8 · The leaderboard — quality *and* robustness
#
# One column is not enough. A fusion method has to be judged on how good it is
# **and** on what happens when the world shifts under it.

# %%
board = {
    "dense only": (metrics.evaluate(run_dense, c.qrels, ks=(K,)), None),
    "bm25 only": (metrics.evaluate(run_bm25, c.qrels, ks=(K,)), None),
    f"linear raw (a={best_alpha})": (before, after),
    "linear min-max (a=0.5)": (norm_before, norm_after),
    "RRF k=60": (rrf_before, rrf_after),
}
rrf_tuned, run_rrf_tuned = eval_fusion(rrf, k_const=best_k)
board[f"RRF k={best_k}"] = (rrf_tuned, eval_scaled(rrf, k_const=best_k))

w = max(len(n) for n in board) + 2
print(f"{'config'.ljust(w)}{'recall@10':>11}{'ndcg@10':>10}{'after x100':>12}{'Δ':>9}"
      f"{'tuning':>18}")
print("-" * (w + 60))
TUNING = {
    "dense only": "—", "bm25 only": "—",
    f"linear raw (a={best_alpha})": "1 weight, fitted",
    "linear min-max (a=0.5)": "1 weight, fitted",
    "RRF k=60": "none (default)",
    f"RRF k={best_k}": "1 constant",
}
for name, (b, a) in board.items():
    nd = b[f"ndcg@{K}"]
    if a is None:
        print(f"{name.ljust(w)}{b[f'recall@{K}']:>11.4f}{nd:>10.4f}"
              f"{'—':>12}{'—':>9}{TUNING[name]:>18}")
    else:
        nda = a[f"ndcg@{K}"]
        print(f"{name.ljust(w)}{b[f'recall@{K}']:>11.4f}{nd:>10.4f}"
              f"{nda:>12.4f}{nda - nd:>+9.4f}{TUNING[name]:>18}")

# %% [markdown]
# **Read that table honestly, because it does not say what the slides implied.**
#
# * Every fusion method beats either retriever alone. Hybrid search wins. That
#   part holds.
# * **Raw-score blending is the only one the ×100 destroys.** It is the version
#   most teams ship, and it is genuinely fragile.
# * **Min-max normalisation is scale-invariant too** — it survives ×100
#   unchanged, and on this corpus it scores highest. That is a real result and we
#   are not going to hide it.
# * RRF is close behind, and gets there **without fitting a weight**.
#
# So the honest claim is not "RRF wins". It is: *RRF gets you within a whisker of
# a tuned blend with nothing to tune and nothing to break.* Whether that trade is
# worth it depends on how many retrievers you have and how often your score
# distributions move.
#
# Section 9 shows the thing min-max normalisation does *not* survive.

# %%
best = max(board, key=lambda n: board[n][0][f"ndcg@{K}"])
print(f"highest nDCG: {best}")
print(f"most robust : every method except 'linear raw' is scale-invariant\n")

d_rrf = metrics.per_query(run_rrf, c.qrels, "ndcg", K)
boot = metrics.bootstrap_delta(nd_dense, d_rrf, n_resamples=2000, seed=7)
print("Is RRF really better than dense alone, or did we get lucky on 300 queries?")
print("  " + metrics.format_delta(boot, f"Δ nDCG@{K} (RRF - dense)"))

# %% [markdown]
# That bootstrap line is Module 04 arriving early, and it matters here: a
# leaderboard is a claim, and a claim on 300 queries needs an interval before you
# believe it.

# %%
fig, ax = plt.subplots(figsize=(8.5, 4.2))
names = list(board)
y = np.arange(len(names))
vals = [board[n][0][f"ndcg@{K}"] for n in names]
# second bar = after the x100, so fragility is visible in the same picture
after_vals = [
    board[n][1][f"ndcg@{K}"] if board[n][1] is not None else board[n][0][f"ndcg@{K}"]
    for n in names
]

ax.barh(y - 0.2, vals, height=0.38, color="#00b894", label="as measured")
ax.barh(y + 0.2, after_vals, height=0.38, color="#ff4d6d", label="after BM25 x100")
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel(f"nDCG@{K}")
ax.set_xlim(0, max(vals) * 1.18)
ax.set_title("Fusion quality, and what survives a score-scale change")
ax.legend(fontsize=8.5, loc="lower right")
for yi, (v, a) in enumerate(zip(vals, after_vals)):
    ax.text(v + 0.004, yi - 0.2, f"{v:.4f}", va="center", fontsize=8)
    if abs(a - v) > 1e-9:
        ax.text(a + 0.004, yi + 0.2, f"{a:.4f}", va="center", fontsize=8,
                color="#c2185b")
ax.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9 · What min-max normalisation does *not* survive
#
# Min-max maps each query's scores onto `[0, 1]` using **that query's own
# maximum**. So a single freak-high score compresses everything else toward zero
# and the retriever effectively stops contributing.
#
# This isn't hypothetical: one query term matching a title exactly, or a very
# short document, produces exactly this. Let's inject it and watch.

# %%
def inject_outlier(result, factor=50.0):
    """Multiply only the top-1 score. Ranking unchanged, distribution wrecked."""
    ids, scores = result
    s = np.asarray(scores, dtype=np.float64).copy()
    if len(s):
        s[0] *= factor
    return ids, s


bm25_outlier = [inject_outlier(r) for r in bm25_res]


def eval_outlier(fuse_fn, **kw):
    run = {}
    for i, qid in enumerate(qids):
        run[qid] = c.ids(list(fuse_fn([dense_res[i], bm25_outlier[i]], **kw))[:K])
    return metrics.evaluate(run, c.qrels, ks=(K,))


mm_out = eval_outlier(linear_blend, weights=[0.5, 0.5], normalize=True)
rrf_out = eval_outlier(rrf, k_const=best_k)

print("one inflated top-1 score per query — rankings within each retriever unchanged\n")
print(f"linear min-max : {norm_before[f'ndcg@{K}']:.4f} -> {mm_out[f'ndcg@{K}']:.4f}  "
      f"({mm_out[f'ndcg@{K}'] - norm_before[f'ndcg@{K}']:+.4f})")
print(f"RRF k={best_k:<9}: {rrf_tuned[f'ndcg@{K}']:.4f} -> {rrf_out[f'ndcg@{K}']:.4f}  "
      f"({rrf_out[f'ndcg@{K}'] - rrf_tuned[f'ndcg@{K}']:+.4f})")

# %% [markdown]
# ## 10 · STRETCH — RRF's one real weakness
#
# Add a **title-only** BM25 index as a third opinion. Prediction from the slides:
# RRF absorbs it for free. Let's check.

# %%
with stopwatch("build title-only BM25"):
    bm25_title = sparse.BM25Index(c.titles, field_name="bm25_title")
title_res = [bm25_title.search(t, CANDIDATES) for t in qtexts]

run_title = {qids[i]: c.ids(title_res[i][0][:K]) for i in range(len(qids))}
s_title = metrics.evaluate(run_title, c.qrels, ks=(K,))
print(f"\ntitle-only BM25 alone : ndcg@{K} {s_title[f'ndcg@{K}']:.4f}   <- a weak retriever")

run3 = {}
for i, qid in enumerate(qids):
    run3[qid] = c.ids(list(rrf([dense_res[i], bm25_res[i], title_res[i]]))[:K])
s3 = metrics.evaluate(run3, c.qrels, ks=(K,))

print(f"\nRRF with 2 retrievers : ndcg@{K} {rrf_before[f'ndcg@{K}']:.4f}")
print(f"RRF with 3 retrievers : ndcg@{K} {s3[f'ndcg@{K}']:.4f}   "
      f"({s3[f'ndcg@{K}'] - rrf_before[f'ndcg@{K}']:+.4f})")

# %% [markdown]
# **It got worse.** That is the honest result, and it is the most useful thing in
# this lab.
#
# RRF gives every list an **equal vote**. That equality is exactly what makes it
# scale-invariant and tuning-free — and it means a weak retriever gets the same
# say as a strong one. Adding one drags the fusion toward its mistakes.
#
# The fix is *not* to start fitting weights, because then you're back to the
# calibration problem. The fix is **retriever selection**: measure each retriever
# alone first, and only fuse the ones that earn their vote.
#
# So RRF's contract is precise: it is robust to score *scale*, and vulnerable to
# retriever *quality*. Those are different exposures, and knowing which one you
# have is the job.

# %% [markdown]
# ## What to take away
#
# 1. **Dense and BM25 fail on different queries.** You measured the split — 72
#    queries to dense, 71 to BM25 — so hybrid isn't a preference, it's arithmetic.
# 2. **Every fusion method beat both single retrievers.** That's the reliable win,
#    and it's why hybrid search is worth the extra index.
# 3. **Blending raw scores is genuinely fragile.** A tuned `alpha` is a unit
#    conversion fitted to one pair of distributions; the ×100 destroyed it with no
#    error message.
# 4. **RRF is not automatically the best — it's the most robust.** On this corpus
#    a tuned min-max blend scored slightly higher. RRF got within a whisker with
#    nothing fitted, and held up under both the ×100 and the outlier injection.
# 5. **Defaults are hypotheses.** `k=60` cost us ~2 nDCG points here, because
#    SciFact has ~1 relevant document per query and damping the head throws away
#    the confidence you wanted.
# 6. **RRF's equal vote is a liability as well as a feature.** Fuse only
#    retrievers that are individually good.
#
# ### The question to carry into Module 04
#
# Your leaderboard has six rows within about 0.06 nDCG of each other, and the
# bootstrap interval on the one comparison we ran was `[+0.0045, +0.0590]` — real,
# but barely. How many of the other differences would you actually stake a release
# on? That's the next module, and it's why we measured instead of eyeballing.
