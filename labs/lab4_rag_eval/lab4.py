# %% [markdown]
# # Lab 4 — Evaluated RAG
#
# **25 minutes.** You will:
#
# 1. Implement `ndcg_at_k` — the log discount is worth typing once by hand
# 2. Score four retrieval configs on a golden set: dense → hybrid → +rerank → +chunking
# 3. Watch chunking move every metric
# 4. Triage real failures into the four failure modes
# 5. Prove one improvement with a bootstrap confidence interval
#
# **New corpus: NFCorpus.** SciFact had binary judgements and ~1 relevant document
# per query. NFCorpus has **graded** judgements (1 and 2) and **38 relevant
# documents per query** — which is what makes nDCG interesting and makes recall@10
# nearly useless. That contrast is the point.
#
# Both checkpoints are must-dos.

# %%
import sys
from collections import Counter
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "wslib").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from wslib import chunking as chunk_mod
from wslib import corpus as corpus_mod
from wslib import embed, fusion, indexes, metrics, rerank, sparse
from wslib.timing import stopwatch

K = 10
CANDIDATES = 100
RERANK_DEPTH = 30          # cross-encoder cost is linear in this
RERANK_CHARS = 1200       # truncate documents for reranking (see section 4)
TOP_K_CONTEXT = 4          # chunks actually handed to the generator

c = corpus_mod.load("nfcorpus")
docs = embed.cached_corpus_embeddings(c)
qids, qvecs = embed.cached_query_embeddings(c)
jq = c.judged_queries()
qtexts = [jq[q] for q in qids]

print(c.summary())

# The cross-encoder truncates at 512 tokens anyway, so passing whole abstracts
# just wastes compute. Title plus opening is what actually gets attended to.
rerank_texts = [t[:RERANK_CHARS] for t in c.texts]

# %% [markdown]
# ## 1 · What a golden set actually is
#
# No magic. Queries, and documents judged against them.

# %%
grades = Counter(g for rel in c.qrels.values() for g in rel.values())
print(f"grade distribution: {dict(sorted(grades.items()))}")
print("  2 = highly relevant, 1 = relevant, absent = not judged\n")

qid0 = qids[0]
rel0 = {d: g for d, g in c.qrels[qid0].items() if g > 0}
print(f'query [{qid0}]: "{jq[qid0]}"')
print(f"  {len(rel0)} relevant documents; grades {Counter(rel0.values())}")
for d, g in list(sorted(rel0.items(), key=lambda x: -x[1]))[:3]:
    print(f"    grade {g}  {c.titles[c.pos(d)][:70]}")

# Worth noticing while you read those titles: NFCorpus judgements come from
# NutritionFacts.org's citation links, so they are BROAD. The query "deafness"
# is judged relevant to documents about ayurvedic medicine. That is not a bug in
# the loader (the id mapping is verified, and our nDCG matches published BEIR
# figures) — it is what a real golden set looks like. Judgement quality is
# another variable in your measurement, not a given.

rel_counts = [sum(1 for g in c.qrels[q].values() if g > 0) for q in jq]
print(f"\nrelevant per query: min {min(rel_counts)} · "
      f"median {sorted(rel_counts)[len(rel_counts) // 2]} · max {max(rel_counts)}")
print(f"=> recall@{K} is capped near {min(1.0, K / np.mean(rel_counts)):.2f} by "
      f"construction. Reporting it as a headline number would be a mistake.")

# %% [markdown]
# ## 2 · TASK: implement nDCG@k
#
# $$\mathrm{DCG@k} = \sum_{i=1}^{k} \frac{2^{g_i} - 1}{\log_2(i + 1)}
# \qquad \mathrm{nDCG@k} = \frac{\mathrm{DCG@k}}{\mathrm{IDCG@k}}$$
#
# Two details that trip people up:
#
# * **Gain is exponential in the grade.** `2**g - 1` makes a grade-2 document
#   worth 3× a grade-1, not 2×. That's a deliberate choice about how much you care
#   about the difference.
# * **IDCG comes from the judgements, not from your results.** It's the DCG of the
#   best ranking that *could* exist for this query.

# %%
def my_ndcg_at_k(ranked, relevant, k):
    """ranked   : list of doc ids, best first
    relevant : {doc_id: grade}, grade 0 or missing means not relevant
    returns  : float in [0, 1]
    """
    # YOUR CODE HERE
    #   1. DCG over ranked[:k]      gain (2**g - 1), discount log2(rank + 1), rank is 1-based
    #   2. IDCG: sort the grades in `relevant` descending, take k, same formula
    #   3. return DCG / IDCG, and 0.0 if IDCG is 0
    raise NotImplementedError("implement my_ndcg_at_k")


# %% [markdown]
# ### CHECKPOINT 1 — hand-checkable

# %%
_ranked = ["d1", "d2", "d3", "d4", "d5"]
_rel = {"d1": 3, "d3": 2, "d5": 1}
# DCG@3  = 7/log2(2) + 0 + 3/log2(4)          = 7.0 + 1.5      = 8.5
# IDCG@3 = 7/log2(2) + 3/log2(3) + 1/log2(4)                   = 9.392789
_want = 8.5 / (7.0 + 3.0 / np.log2(3) + 0.5)

try:
    got = my_ndcg_at_k(_ranked, _rel, 3)
    assert abs(got - _want) < 1e-9, f"expected {_want:.10f}, got {got:.10f}"
    print(f"CP1 PASS — {got:.10f}, using your implementation")
    ndcg_at_k = my_ndcg_at_k
except NotImplementedError:
    print("CP1 — not implemented yet, using the reference")
    ndcg_at_k = metrics.ndcg_at_k
except AssertionError as e:
    print(f"CP1 FAIL — {e}\n   using the reference")
    ndcg_at_k = metrics.ndcg_at_k

print(f"  perfect order      -> {ndcg_at_k(['d1', 'd3', 'd5'], _rel, 3):.4f}")
print(f"  reversed order     -> {ndcg_at_k(['d5', 'd3', 'd1'], _rel, 3):.4f}")
print(f"  nothing relevant   -> {ndcg_at_k(['zz'], _rel, 3):.4f}")

# %% [markdown]
# ### Why graded relevance matters
#
# Same positions, swapped grades. Binary judgements cannot see this difference —
# NFCorpus can.

# %%
print(f"grade-2 doc at rank 1: {ndcg_at_k(['d3', 'd5'], {'d3': 2, 'd5': 1}, 2):.4f}")
print(f"grade-1 doc at rank 1: {ndcg_at_k(['d5', 'd3'], {'d3': 2, 'd5': 1}, 2):.4f}")
print("\nSame two documents, both retrieved, only the order differs.")

# %% [markdown]
# ## 3 · Four configurations, one golden set
#
# Retrieve once per strategy, then score everything the same way.

# %%
dense = indexes.ExactIndex(docs)
bm25 = sparse.BM25Index(c.texts)

with stopwatch("dense retrieval"):
    dense_res = [(r.ids[0], r.scores[0])
                 for r in (dense.search(qvecs[i][None, :], CANDIDATES)
                           for i in range(len(qids)))]
with stopwatch("bm25 retrieval"):
    bm25_res = [bm25.search(t, CANDIDATES) for t in qtexts]

runs = {}
runs["dense"] = {qids[i]: c.ids(dense_res[i][0][:K]) for i in range(len(qids))}
runs["bm25"] = {qids[i]: c.ids(bm25_res[i][0][:K]) for i in range(len(qids))}

# RRF k=1 — Lab 3 showed k=60 is not always right; NFCorpus has many relevant
# documents per query, so this time the damping should actually help. Check both.
for kc in (1, 60):
    runs[f"hybrid rrf k={kc}"] = {
        qids[i]: c.ids(fusion.rrf([dense_res[i], bm25_res[i]], k_const=kc)[:K])
        for i in range(len(qids))
    }

for name, run in runs.items():
    s = metrics.evaluate(run, c.qrels, ks=(K,))
    print(f"  {name:<18} ndcg@{K} {s[f'ndcg@{K}']:.4f}  recall@{K} {s[f'recall@{K}']:.4f}")

# %% [markdown]
# **Note what just happened with `k`.** On SciFact (≈1 relevant document per
# query) `k=1` won. Here, with 38 relevant documents per query, the damping that
# hurt us there should help. The right value is a property of your corpus, not of
# the algorithm.

# %% [markdown]
# ## 4 · Reranking
#
# The cross-encoder reads each (query, document) pair with full attention. Far too
# slow for the corpus, entirely affordable on a few dozen candidates.

# %%
BEST_RRF = 60
hybrid_cands = [fusion.rrf([dense_res[i], bm25_res[i]], k_const=BEST_RRF)[:RERANK_DEPTH]
                for i in range(len(qids))]

reranker = rerank.CrossEncoderReranker()

# Feel the cost before we hide it behind a cache. This is the number that
# decides whether you rerank 30 candidates or 300 in production.
sample = [rerank_texts[j] for j in hybrid_cands[0][:16]]
reranker.score(qtexts[0], sample[:1])          # warm up: load the model first
with stopwatch("", quiet=True) as sw_one:
    reranker.score(qtexts[0], sample)
rate = 16 / max(sw_one.seconds, 1e-9)
print(f"cross-encoder throughput: {rate:.0f} pairs/sec single-threaded")
print(f"  this corpus, depth {RERANK_DEPTH}, {len(qids)} queries "
      f"= {len(qids) * RERANK_DEPTH:,} pairs "
      f"= {len(qids) * RERANK_DEPTH / rate / 60:.0f} min")
print("  ...which is why the scores are precomputed. Compare that to the 0.04s")
print("  the bi-encoder took for the whole corpus.")

with stopwatch(f"rerank {len(qids)} x {RERANK_DEPTH}"):
    reranked = reranker.rerank_all(
        qtexts, hybrid_cands, rerank_texts,
        cache_tag=f"nfcorpus-h{BEST_RRF}-d{RERANK_DEPTH}-t{RERANK_CHARS}")

runs["hybrid + rerank"] = {qids[i]: c.ids(reranked[i][:K]) for i in range(len(qids))}
s = metrics.evaluate(runs["hybrid + rerank"], c.qrels, ks=(K,))
print(f"\n  hybrid + rerank    ndcg@{K} {s[f'ndcg@{K}']:.4f}  "
      f"recall@{K} {s[f'recall@{K}']:.4f}")
print("\nRecall is unchanged by construction — reranking reorders the same")
print("candidates, it never retrieves anything new. Only nDCG can see it.")

# %% [markdown]
# ## 5 · Chunking
#
# Now retrieve over **chunks** instead of whole documents, then pool back to
# documents so the judgements still apply. The pooling rule is a retrieval
# decision: a document is as good as its best chunk.

# %%
for strategy, size in [("sentence", 512), ("fixed", 256)]:
    ch = chunk_mod.chunk_corpus(c.texts, strategy, size, 0)
    cvecs = embed.cached_chunk_embeddings(c, ch)
    cidx = indexes.ExactIndex(cvecs)
    print(f"\n{ch.summary()}")

    run = {}
    for i, qid in enumerate(qids):
        chunk_ids = cidx.search(qvecs[i][None, :], CANDIDATES * 2).ids[0]
        parents = chunk_mod.pool_to_parents(chunk_ids, ch.parent, top_k=K)
        run[qid] = c.ids(parents)
    runs[f"chunked {strategy}-{size}"] = run
    s = metrics.evaluate(run, c.qrels, ks=(K,))
    print(f"  ndcg@{K} {s[f'ndcg@{K}']:.4f}  recall@{K} {s[f'recall@{K}']:.4f}")

# %% [markdown]
# ## 6 · The leaderboard

# %%
board = {name: metrics.evaluate(run, c.qrels, ks=(5, K)) for name, run in runs.items()}
print(metrics.leaderboard(board, ks=(5, K)))

winner = max(board, key=lambda n: board[n][f"ndcg@{K}"])
print(f"\nhighest nDCG@{K}: {winner}")

# %%
fig, ax = plt.subplots(figsize=(8.5, 4.2))
names = list(board)
y = np.arange(len(names))
nd = [board[n][f"ndcg@{K}"] for n in names]
rc = [board[n][f"recall@{K}"] for n in names]
ax.barh(y - 0.2, nd, height=0.38, color="#00b894", label=f"nDCG@{K}")
ax.barh(y + 0.2, rc, height=0.38, color="#6c63ff", label=f"recall@{K}")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9); ax.invert_yaxis()
ax.legend(fontsize=8.5); ax.grid(axis="x", alpha=0.25)
ax.set_title("NFCorpus — note how differently the two metrics rank these")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 7 · TASK: triage the failures
#
# "The RAG hallucinated" is not a diagnosis. Find where the first relevant
# document actually landed, and the failure mode follows mechanically:
#
# | where the first relevant document is | failure mode | fix at |
# |---|---|---|
# | not in the candidate pool at all | **context missing** | retrieval |
# | in the pool, but below rank k | **context noise** | ranking / reranking |
# | in the top k, but below TOP_K_CONTEXT | **context overflow** | assembly |
# | inside the context we sent | **retrieval succeeded** | nothing to fix here |
#
# Note the honest name for the fourth bucket. We have not generated an answer for
# these queries, so we cannot call them generator failures — all we know is that
# retrieval put the right document in front of the model. If the answer is still
# wrong, *then* it is the generator's fault.

# %%
def classify(first_rel_rank, k=K, top_k_context=TOP_K_CONTEXT):
    """first_rel_rank: 1-based rank of the first relevant doc, or None if absent.
    returns one of: 'context missing', 'context noise', 'context overflow',
                    'retrieval succeeded'
    """
    # YOUR CODE HERE — four branches, in order.
    raise NotImplementedError("implement classify")


def reference_classify(first_rel_rank, k=K, top_k_context=TOP_K_CONTEXT):
    if first_rel_rank is None:
        return "context missing"
    if first_rel_rank > k:
        return "context noise"
    if first_rel_rank > top_k_context:
        return "context overflow"
    return "retrieval succeeded"


# %% [markdown]
# ### CHECKPOINT 2

# %%
_cases = [(None, "context missing"), (73, "context noise"),
          (7, "context overflow"), (2, "retrieval succeeded")]
try:
    for rank, want in _cases:
        got = classify(rank)
        assert got == want, f"rank {rank}: expected {want!r}, got {got!r}"
    print("CP2 PASS — using your implementation")
except NotImplementedError:
    print("CP2 — not implemented yet, using the reference")
    classify = reference_classify
except AssertionError as e:
    print(f"CP2 FAIL — {e}\n   using the reference")
    classify = reference_classify

# %% [markdown]
# ### Triage the whole query set
#
# Run it against the winning configuration's full candidate pool.

# %%
final_pool = reranked           # the reranked hybrid candidates
diagnosis = {}
for i, qid in enumerate(qids):
    gold = {d for d, g in c.qrels[qid].items() if g > 0}
    rank = None
    for r, row in enumerate(final_pool[i], start=1):
        if c.doc_ids[row] in gold:
            rank = r
            break
    diagnosis[qid] = classify(rank)

counts = Counter(diagnosis.values())
total = len(diagnosis)
print(f"{'failure mode':<22}{'queries':>9}{'share':>8}   fix at")
print("-" * 62)
FIX = {
    "context missing": "retrieval — chunking, hybrid, deeper k",
    "context noise": "ranking — rerank, tighter top-k",
    "context overflow": "assembly — reorder, raise TOP_K_CONTEXT",
    "retrieval succeeded": "nothing — the right document was in the prompt",
}
ORDER = ["context missing", "context noise", "context overflow", "retrieval succeeded"]
for mode in ORDER:
    n = counts.get(mode, 0)
    print(f"{mode:<22}{n:>9}{n / total * 100:>7.1f}%   {FIX[mode]}")

broken = sum(counts.get(m, 0) for m in ORDER[:3])
print(f"\n{broken} of {total} queries ({broken / total * 100:.0f}%) never got the right")
print("document in front of the model. For those, no amount of prompt engineering")
print("helps — and all three causes are fixable in the retrieval layer you spent")
print("three modules learning to tune.")
print(f"\nThe other {total - broken} had a good document in the context. If those")
print("answers are still wrong, THAT is when 'the model hallucinated' is a")
print("diagnosis rather than a shrug.")

# %% [markdown]
# ### Look at one actual failure

# %%
missing = [q for q, d in diagnosis.items() if d == "context missing"]
if missing:
    qid = missing[0]
    i = qids.index(qid)
    gold = {d: g for d, g in c.qrels[qid].items() if g > 0}
    print(f'query [{qid}]: "{jq[qid]}"')
    print(f"  {len(gold)} relevant documents, none in the top {RERANK_DEPTH}\n")
    print("  what we returned instead:")
    for r, row in enumerate(final_pool[i][:3], start=1):
        print(f"    {r}. {c.titles[row][:68]}")
    print("\n  what we should have found:")
    for d in list(gold)[:3]:
        print(f"       {c.titles[c.pos(d)][:68]}")
    print("\n  No prompt engineering fixes this. The document was never retrieved.")

# %% [markdown]
# ## 8 · TASK-adjacent: prove the improvement
#
# The leaderboard is a claim. Back it with an interval.

# %%
def compare(base_name, cand_name, seed=11):
    a = metrics.per_query(runs[base_name], c.qrels, "ndcg", K)
    b = metrics.per_query(runs[cand_name], c.qrels, "ndcg", K)
    boot = metrics.bootstrap_delta(a, b, n_resamples=2000, seed=seed)
    lead = f"{cand_name} - {base_name}"
    print(f"  {metrics.format_delta(boot, f'Δ nDCG@{K} ({lead})')}")
    return boot


print("the change that WON the leaderboard:")
b_rerank = compare("hybrid rrf k=60", "hybrid + rerank")

print("\na smaller-looking change:")
b_chunk = compare("dense", "chunked sentence-512")

print("\nthe one nobody should ship:")
b_fixed = compare("dense", "chunked fixed-256")

# %% [markdown]
# ### Sit with that for a second
#
# The configuration that **topped the leaderboard** — hybrid + rerank, the biggest
# single nDCG jump in the table — has a confidence interval that touches zero. On
# 323 queries we cannot actually establish it.
#
# Meanwhile sentence chunking, which looks like a smaller effect, is comfortably
# significant.
#
# Neither result is a reason to abandon reranking or to celebrate chunking. It is
# a reason to know **which of your numbers you are entitled to say out loud.** The
# per-query win/loss counts are the tell: a change can improve the mean while
# regressing a third of your queries, and that is a product decision, not a
# rounding error.

# %%
for name, boot in [("hybrid + rerank", b_rerank),
                   ("chunked sentence-512", b_chunk),
                   ("chunked fixed-256", b_fixed)]:
    verdict = "can claim it" if boot["significant"] else "CANNOT claim it"
    print(f"  {name:<24} {int(boot['improved']):>4} better / "
          f"{int(boot['regressed']):>4} worse   -> {verdict}")

# %% [markdown]
# ## 9 · Assemble the context and answer
#
# Everything so far was retrieval. This is the part that faces the user — and the
# only part most demos show.

# %%
def assemble(query, doc_rows, top_k=TOP_K_CONTEXT, max_chars=600):
    """Order for 'lost in the middle': best first, second-best LAST.

    Models attend most strongly to the start and end of a long context, so the
    weakest chunks belong in the middle where attention is thinnest.
    """
    picked = list(doc_rows)[:top_k]
    if len(picked) > 2:
        reordered = [picked[0]] + picked[2:] + [picked[1]]
    else:
        reordered = picked

    blocks = []
    for n, row in enumerate(reordered, start=1):
        blocks.append(f"[{n}] {c.titles[row]}\n{c.bodies[row][:max_chars]}")
    return reordered, "\n\n".join(blocks)


def answer(query, context, sources):
    """Generate, or fall back to extractive so nobody needs an API key."""
    import os
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return ("[extractive fallback — no OPENAI_API_KEY set]\n"
                f"Best-supported passage:\n\n{context.split(chr(10) + chr(10))[0]}\n\n"
                f"Cited sources: {', '.join(sources)}")
    try:
        from openai import OpenAI
        client = OpenAI(base_url=os.environ.get("LLM_BASE_URL") or None)
        prompt = (
            "Answer the question using ONLY the numbered sources below. "
            "Cite them as [1], [2]. If they do not contain the answer, say so.\n\n"
            f"Sources:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        r = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return r.choices[0].message.content
    except Exception as exc:  # noqa: BLE001
        return f"[generation failed: {type(exc).__name__}: {exc}]"


good = [q for q, d in diagnosis.items() if d == "retrieval succeeded"]
qid = good[0] if good else qids[0]
i = qids.index(qid)

rows, context = assemble(jq[qid], reranked[i])
gold = {d for d, g in c.qrels[qid].items() if g > 0}

print(f'query: "{jq[qid]}"\n')
print("context sent to the model:")
for n, row in enumerate(rows, start=1):
    mark = "*" if c.doc_ids[row] in gold else " "
    print(f"  [{n}]{mark} {c.titles[row][:66]}")
print("  (* = judged relevant)\n")
print(answer(jq[qid], context, [c.doc_ids[r] for r in rows]))

# %% [markdown]
# ## What to take away
#
# 1. **Pick the metric your corpus can support.** recall@10 is nearly meaningless
#    here — 38 relevant documents cannot fit in 10 slots. nDCG@10 is the number to
#    report, and knowing *why* is the skill.
# 2. **Reranking cannot change recall.** It reorders the candidates you already
#    had. If a metric doesn't move when you add a reranker, check whether that
#    metric is even capable of seeing it.
# 3. **Chunking is a retrieval decision.** It changes what gets indexed, what gets
#    retrieved, and how you have to pool results back to documents.
# 4. **Most RAG failures are retrieval failures.** You measured the split. Log
#    your retrieved document ids alongside every answer or you cannot do this
#    triage at all.
# 5. **A leaderboard without an interval is a hunch in a table.** The config that
#    topped this leaderboard has a confidence interval touching zero — we cannot
#    establish it on 323 queries — while a smaller-looking chunking change is
#    solid. Report what you can defend.
# 6. **Reranking is expensive in a way bi-encoders are not.** You measured the
#    throughput: the whole corpus embeds in a fraction of a second, and a few
#    thousand cross-encoder pairs take minutes. That asymmetry is exactly why the
#    funnel has the shape it has.
#
# ### Take this home
#
# Swap `corpus_mod.load(...)` for your own documents and a golden set in the same
# `{query: {doc_id: grade}}` shape. Every metric, index, and fusion function in
# `wslib` works unchanged. Fifty judged queries is an afternoon, and it will
# change how your team argues about search.
