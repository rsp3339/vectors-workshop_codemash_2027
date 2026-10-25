# %% [markdown]
# # Lab 1 — Meaning as geometry
#
# **25 minutes.** You will:
#
# 1. Embed sentences and read a similarity matrix
# 2. Implement `normalize` and verify that cosine, dot product, and L2 agree
# 3. Swap mean ↔ CLS pooling and watch retrieval quality move
# 4. Truncate Matryoshka embeddings and **measure what you lose**
# 5. Rescore with a cross-encoder and see where it disagrees
#
# Corpus: **BEIR SciFact**, 5,183 scientific abstracts, 300 judged queries.
# Everything heavy is precomputed, so this runs in seconds.
#
# `CP1` and `CP2` are the must-dos.

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
from wslib import embed, indexes, metrics, rerank
from wslib.timing import bytes_human, stopwatch

np.set_printoptions(precision=3, suppress=True)

K = 10
c = corpus_mod.load("scifact")
qids = sorted(c.judged_queries())
jq = c.judged_queries()
qtexts = [jq[q] for q in qids]

print(c.summary())

# %% [markdown]
# ## 1 · Meaning becomes geometry
#
# Six sentences: three paraphrase pairs on different topics. Nobody told the
# model which pairs go together.

# %%
SENTENCES = [
    "Smoking causes lung cancer.",                                  # A
    "Tobacco use leads to pulmonary malignancy.",                   # A'
    "Exercise lowers blood pressure.",                              # B
    "Physical activity reduces hypertension.",                      # B'
    "Vitamin D deficiency is common in winter.",                    # C
    "Low sunlight months bring insufficient cholecalciferol.",      # C'
]
LABELS = ["A smoking", "A' tobacco", "B exercise", "B' activity",
          "C vitaminD", "C' sunlight"]

# Note: not one of these pairs shares a content word with its partner.
# "cancer"/"malignancy", "exercise"/"physical activity",
# "vitamin D"/"cholecalciferol" — a keyword index scores every pair at zero.
with stopwatch("load model + encode 6 sentences"):
    V = embed.encode(SENTENCES, show_progress=False)

print(f"\neach sentence is now {V.shape[1]} numbers. The first six of sentence A:")
print(f"  {V[0][:6]}")
print("\nNo single number means anything. Only the relative positions do:\n")

S = V @ V.T                       # unit vectors, so this IS the cosine matrix
print(" " * 14 + "".join(f"{l[:9]:>11}" for l in LABELS))
for i, row in enumerate(S):
    cells = "".join(
        f"{v:>11.3f}" if i != j else f"{'—':>11}" for j, v in enumerate(row)
    )
    print(f"{LABELS[i]:<14}{cells}")

pairs = [(0, 1), (2, 3), (4, 5)]
print("\nparaphrase pairs:")
for i, j in pairs:
    print(f"  {LABELS[i]:<14} vs {LABELS[j]:<14} {S[i, j]:.3f}")
others = [S[i, j] for i in range(6) for j in range(6)
          if i < j and (i, j) not in pairs]
print(f"\n  mean of every other pair: {np.mean(others):.3f}")

# Check the claim rather than asserting it: is each sentence's nearest
# neighbour actually its own paraphrase?
off_diag = S.copy()
np.fill_diagonal(off_diag, -np.inf)
partner = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
all_mutual = all(int(np.argmax(off_diag[i])) == partner[i] for i in range(6))
print(f"  every sentence's nearest neighbour is its own paraphrase: {all_mutual}")
print("\n  Nobody labelled these pairs. The pairing fell out of the geometry —")
print("  and no pair shares a content word, so keyword search finds none of it.")

# %% [markdown]
# ## 2 · TASK: implement normalize, then check the identity
#
# Module 01 claimed that once vectors are unit length, cosine, dot product and L2
# distance all rank identically, because
#
# $$\|u - v\|^2 = 2 - 2(u \cdot v) \quad\Longrightarrow\quad u \cdot v = 1 - \tfrac{1}{2}\|u-v\|^2$$
#
# **That identity requires unit vectors.** You're about to see it fail and then
# hold.

# %%
def my_normalize(x):
    """Scale each row to unit L2 length. Leave all-zero rows alone.

    x : (n, d) array — or (d,) for a single vector
    """
    # YOUR CODE HERE — compute row norms, divide, guard against divide-by-zero.
    raise NotImplementedError("implement my_normalize")


# %% [markdown]
# ### CHECKPOINT 1

# %%
_probe = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])
try:
    got = my_normalize(_probe)
    want = embed.normalize(_probe)
    assert np.allclose(got, want, atol=1e-6), f"expected\n{want}\ngot\n{got}"
    print("CP1 PASS — using your implementation")
    normalize = my_normalize
except NotImplementedError:
    print("CP1 — not implemented yet, using the reference")
    normalize = embed.normalize
except AssertionError as e:
    print(f"CP1 FAIL — {e}\n   using the reference")
    normalize = embed.normalize

print(f"  [3,4] -> {normalize(_probe)[0]}  (length {np.linalg.norm(normalize(_probe)[0]):.4f})")
print(f"  [0,0] -> {normalize(_probe)[2]}  (no divide-by-zero)")

# %%
# Take two real document vectors and deliberately break the unit-norm assumption.
a_raw = V[0] * 3.0          # same direction, three times as long
b_raw = V[2] * 0.4          # same direction, much shorter

def report(u, v, label):
    dot = float(u @ v)
    cos = float(normalize(u[None, :])[0] @ normalize(v[None, :])[0])
    l2 = float(np.linalg.norm(u - v))
    identity = 1 - 0.5 * l2 ** 2
    ok = abs(identity - cos) < 1e-5
    print(f"{label:<22} dot {dot:+.4f}   cos {cos:+.4f}   "
          f"L2 {l2:.4f}   1-½L2² {identity:+.4f}   {'MATCH' if ok else 'MISMATCH'}")

print("scaled (not unit length):")
report(a_raw, b_raw, "  a*3.0, b*0.4")
print("\nnormalized:")
report(normalize(a_raw[None, :])[0], normalize(b_raw[None, :])[0], "  unit a, unit b")

print("\nCosine never moved — it only ever looked at direction. dot and L2 did.")
print("And the identity only holds in the second case. State that precondition")
print("out loud whenever you write it on a whiteboard.")

# %% [markdown]
# ### Why your vector database cares
#
# If normalized vectors make inner product equal to cosine, an engine can skip
# the norms and the division entirely — so it stores unit vectors and computes
# inner product, and calls the result cosine.
#
# The failure mode is the other direction: choose inner product on
# **un-normalized** vectors and long documents win everything, because magnitude
# becomes a popularity prior you never asked for.

# %%
docs = embed.cached_corpus_embeddings(c)
_, queries = embed.cached_query_embeddings(c)

run_norm = indexes.ExactIndex(docs).search(queries, K).to_run(c, qids)

# now un-normalize the documents by scaling each one by its text length
lengths = np.array([len(t) for t in c.texts], dtype=np.float32)
scale = (lengths / lengths.mean())[:, None]
run_bad = indexes.ExactIndex(docs * scale).search(queries, K).to_run(c, qids)

s_norm = metrics.evaluate(run_norm, c.qrels, ks=(K,))
s_bad = metrics.evaluate(run_bad, c.qrels, ks=(K,))
print(f"unit vectors, inner product : ndcg@{K} {s_norm[f'ndcg@{K}']:.4f}")
print(f"length-scaled, inner product: ndcg@{K} {s_bad[f'ndcg@{K}']:.4f}   "
      f"({s_bad[f'ndcg@{K}'] - s_norm[f'ndcg@{K}']:+.4f})")
print("\nNo error. No warning. Just quietly worse search.")

# %% [markdown]
# ## 3 · Pooling: one vector per token, but you need one per text
#
# A transformer gives you a vector per **token**. Something has to collapse them.
# The three usual choices, all from the same forward pass:
#
# * **mean** — average every token vector. The robust default.
# * **CLS** — take the special first token. Only meaningful if the model was
#   *trained* to put a summary there.
# * **max** — element-wise maximum. Occasionally used, rarely well.

# %%
dp = embed.cached_poolings(c, "corpus")
qp = embed.cached_poolings(c, "queries")

print(f"{'pooling':<8}{'ndcg@10':>10}{'recall@10':>12}{'vs mean':>10}")
print("-" * 40)
pool_scores = {}
for p in ["mean", "cls", "max"]:
    run = indexes.ExactIndex(dp[p]).search(qp[p], K).to_run(c, qids)
    s = metrics.evaluate(run, c.qrels, ks=(K,))
    pool_scores[p] = s
    delta = s[f"ndcg@{K}"] - pool_scores["mean"][f"ndcg@{K}"]
    print(f"{p:<8}{s[f'ndcg@{K}']:>10.4f}{s[f'recall@{K}']:>12.4f}"
          f"{(f'{delta:+.4f}' if p != 'mean' else '—'):>10}")

# %% [markdown]
# **`all-MiniLM-L6-v2` was trained with mean pooling.** Using CLS costs ~0.05
# nDCG and max costs ~0.18 — same weights, same text, same forward pass, just a
# different way of collapsing it.
#
# Nothing raises an exception. This is why "read the model card" is real advice
# and not a platitude: **pooling is dictated by how the model was trained, not by
# your preference.**

# %% [markdown]
# ## 4 · TASK: Matryoshka truncation
#
# An MRL-trained model packs meaning front-loaded, coarse to fine, so the first
# `d` dimensions are themselves a usable embedding. Truncate, re-normalize, done.
#
# Re-normalizing is not optional: chopping dimensions off a unit vector leaves it
# shorter than 1, and your index assumes unit length.

# %%
def truncate(vectors, dim):
    """Keep the first `dim` dimensions and restore unit length."""
    # YOUR CODE HERE — two lines.
    raise NotImplementedError("implement truncate")


# %% [markdown]
# ### CHECKPOINT 2

# %%
try:
    got = truncate(docs, 128)
    assert got.shape == (len(docs), 128), f"wrong shape {got.shape}"
    norms = np.linalg.norm(got, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), \
        f"not unit length: {norms.min():.4f}..{norms.max():.4f} (did you re-normalize?)"
    print("CP2 PASS — using your implementation")
except NotImplementedError:
    print("CP2 — not implemented yet, using the reference")
    truncate = lambda v, d: embed.normalize(v[:, :d])
except AssertionError as e:
    print(f"CP2 FAIL — {e}\n   using the reference")
    truncate = lambda v, d: embed.normalize(v[:, :d])

# %% [markdown]
# ### The experiment that matters
#
# Truncate an **MRL-trained** model and a **non-MRL** model over the same
# fractions. If MRL were doing nothing, the two curves would look the same.

# %%
mrl_docs = embed.cached_corpus_embeddings(c, model_name=embed.MRL_MODEL)
_, mrl_queries = embed.cached_query_embeddings(c, model_name=embed.MRL_MODEL)

def retention_curve(dv, qv, fractions=(1, 2, 4, 8, 16, 32)):
    full = dv.shape[1]
    rows = []
    for f in fractions:
        d = full // f
        if d < 8:
            continue
        run = indexes.ExactIndex(truncate(dv, d)).search(truncate(qv, d), K).to_run(c, qids)
        nd = metrics.evaluate(run, c.qrels, ks=(K,))[f"ndcg@{K}"]
        rows.append({"dim": d, "frac": f, "ndcg": nd})
    base = rows[0]["ndcg"]
    for r in rows:
        r["retained"] = r["ndcg"] / base
    return rows


with stopwatch("MRL model"):
    mrl_rows = retention_curve(mrl_docs, mrl_queries)
with stopwatch("non-MRL model (MiniLM)"):
    mini_rows = retention_curve(docs, queries)

print(f"\n{'':>6}{'MRL: static-retrieval-mrl-en-v1':>36}{'non-MRL: all-MiniLM-L6-v2':>34}")
print(f"{'frac':>6}{'dim':>8}{'ndcg@10':>11}{'retained':>11}"
      f"{'dim':>10}{'ndcg@10':>11}{'retained':>11}")
print("-" * 76)
for a, b in zip(mrl_rows, mini_rows):
    print(f"{'1/' + str(a['frac']):>6}{a['dim']:>8}{a['ndcg']:>11.4f}"
          f"{a['retained'] * 100:>10.1f}%{b['dim']:>10}{b['ndcg']:>11.4f}"
          f"{b['retained'] * 100:>10.1f}%")

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot([r["frac"] for r in mrl_rows], [r["retained"] * 100 for r in mrl_rows],
           "o-", color="#00b894", label="MRL-trained")
ax[0].plot([r["frac"] for r in mini_rows], [r["retained"] * 100 for r in mini_rows],
           "s--", color="#ff4d6d", label="not MRL-trained")
ax[0].set_xscale("log", base=2)
ax[0].set_xlabel("truncated to 1/N of full width")
ax[0].set_ylabel("% of full-width nDCG retained")
ax[0].set_title("MRL degrades gracefully. Naive truncation doesn't.")
ax[0].axhline(95, ls=":", c="#9aa0c4", lw=1)
ax[0].legend(fontsize=8.5)
ax[0].grid(alpha=0.25)

ax[1].plot([r["dim"] for r in mrl_rows], [r["ndcg"] for r in mrl_rows],
           "o-", color="#00b894", label="MRL-trained")
ax[1].plot([r["dim"] for r in mini_rows], [r["ndcg"] for r in mini_rows],
           "s--", color="#ff4d6d", label="not MRL-trained")
ax[1].set_xscale("log", base=2)
ax[1].set_xlabel("dimensions")
ax[1].set_ylabel(f"nDCG@{K} (absolute)")
ax[1].set_title("...but absolute quality is a different question")
ax[1].legend(fontsize=8.5)
ax[1].grid(alpha=0.25)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Read both panels, not just the first
#
# **Left:** MRL is doing real work. At one eighth of the width it keeps ~90% of
# its quality; naive truncation keeps ~68%. The dimensions really are ordered by
# importance.
#
# **Right, and be honest about it:** MiniLM at full width *outscores* the MRL
# model at full width on this corpus. MRL is a property of how a model was
# trained, not a quality ranking. The right order of operations is **pick the
# best model for your task, then check whether it truncates well** — not the
# other way round.

# %%
FULL = mrl_docs.shape[1]
N = 10_000_000
print(f"what truncation buys at {N:,} vectors (float32):\n")
print(f"{'dim':>6}{'bytes/vec':>12}{'total':>12}{'retained':>11}")
print("-" * 41)
for r in mrl_rows:
    print(f"{r['dim']:>6}{r['dim'] * 4:>12}{bytes_human(r['dim'] * 4 * N):>12}"
          f"{r['retained'] * 100:>10.1f}%")
print(f"\nThe production pattern: truncated vectors for the wide first pass,")
print(f"full-width rescore on the survivors. Cheap recall, exact ordering.")

# %% [markdown]
# ## 5 · Bi-encoder vs cross-encoder
#
# Everything so far used a **bi-encoder**: query and document embedded
# separately, so the corpus can be precomputed. A **cross-encoder** reads the
# pair together with attention across both — better judgement, nothing cacheable.

# %%
reranker = rerank.CrossEncoderReranker()
bi = indexes.ExactIndex(docs)
reranker.score("warm up", ["the first call loads the model"])

# Don't hardcode an example and hope. Score a batch of queries where the
# bi-encoder retrieved a relevant document but buried it — that is the situation
# reranking exists for — then show the biggest win AND the overall record.
def survey(limit=30):
    rows = []
    for i in range(limit):
        gold_i = {d for d, g in c.qrels[qids[i]].items() if g > 0}
        cand_i = bi.search(queries[i][None, :], 10).ids[0].tolist()
        ranks = [r for r, row in enumerate(cand_i, 1) if c.doc_ids[row] in gold_i]
        if not ranks or min(ranks) < 3:
            continue                            # nothing buried to rescue
        s = reranker.score(qtexts[i], [c.texts[j][:1200] for j in cand_i])
        order_i = np.argsort(-s).tolist()
        new = min(order_i.index(r - 1) + 1 for r in ranks)
        rows.append({"i": i, "old": min(ranks), "new": new,
                     "gain": min(ranks) - new, "cand": cand_i,
                     "gold": gold_i, "scores": s, "order": order_i})
    return rows


with stopwatch(f"survey {30} queries", quiet=True) as sw_survey:
    surveyed = survey()

better = sum(1 for r in surveyed if r["gain"] > 0)
worse = sum(1 for r in surveyed if r["gain"] < 0)
same = len(surveyed) - better - worse
print(f"across {len(surveyed)} queries with a buried relevant document:")
print(f"  cross-encoder promoted it in {better}, demoted it in {worse}, "
      f"left it in {same}")
print(f"  ({sw_survey.seconds:.1f}s for {len(surveyed) * 10} pairs)\n")

best = max(surveyed, key=lambda r: r["gain"])
i, cand, gold, order = best["i"], best["cand"], best["gold"], np.array(best["order"])
qid = qids[i]

with stopwatch("", quiet=True) as sw:
    ce_scores = reranker.score(qtexts[i], [c.texts[j][:1200] for j in cand])
order = np.argsort(-ce_scores)
print("the clearest win in that batch:")

print(f'query: "{qtexts[i]}"')
print(f"relevant: {sorted(gold)}\n")
print(f"{'bi-encoder rank':<17}{'cross-encoder rank':<20}title")
print("-" * 92)
for new_rank, idx in enumerate(order, start=1):
    row = cand[idx]
    mark = "*" if c.doc_ids[row] in gold else " "
    moved = idx + 1 - new_rank
    arrow = f"({moved:+d})" if moved else "( =)"
    print(f"{idx + 1:<17}{str(new_rank) + ' ' + arrow:<20}{mark}{c.titles[row][:60]}")
print("\n(* = judged relevant)")

old = min(r for r, row in enumerate(cand, 1) if c.doc_ids[row] in gold)
new = min(list(order).index(r - 1) + 1 for r, row in enumerate(cand, 1)
          if c.doc_ids[row] in gold)
print(f"\nThe relevant document moved from bi-encoder rank {old} to rank {new}.")
print("The bi-encoder found it; the cross-encoder noticed it mattered.")
print(f"\nBut look at the record above: it demoted the relevant document on")
print(f"{worse} of {len(surveyed)} queries. Reranking is a strong average bet, not a")
print("guarantee — which is exactly why Lab 4 puts a confidence interval on it.")

print(f"\ncross-encoder: 10 pairs in {sw.seconds * 1000:.0f} ms "
      f"= {10 / sw.seconds:.0f} pairs/sec")
print(f"bi-encoder   : the entire {len(c):,}-document corpus was already encoded,")
print(f"               and searching it took well under a millisecond.")
print(f"\nThat asymmetry is the funnel: retrieve wide and cheap with the bi-encoder,")
print(f"then spend the cross-encoder only on the handful that might change places.")
print("Lab 3 and Lab 4 measure what that reordering is actually worth.")

# %% [markdown]
# ## What to take away
#
# 1. **Similarity is geometry.** Paraphrases cluster because the model put them
#    near each other, not because anyone labelled the pairs.
# 2. **Normalize, then the metric barely matters.** Cosine, dot and L2 rank
#    identically on unit vectors — and the identity
#    `cos = 1 - ½‖u-v‖²` needs that precondition. Inner product on
#    un-normalized vectors is a silent bug that ranks by length.
# 3. **Pooling is not a free choice.** You measured the cost of getting it wrong
#    on a model that never complains.
# 4. **MRL truncation is real, and it is not a quality ranking.** Measure the
#    retention curve on *your* corpus, and pick the model first.
# 5. **Bi-encoder to retrieve, cross-encoder to rerank.** You felt the cost
#    difference directly — that is the entire reason the funnel exists.
#
# ### Into Module 02
#
# Every search in this lab was **exact** — a full scan of all 5,183 vectors. That
# stops being possible somewhere between here and a million documents, and the
# next module is about what you give up to keep going.
