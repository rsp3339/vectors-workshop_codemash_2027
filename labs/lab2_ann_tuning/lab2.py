# %% [markdown]
# # Lab 2 — Tuning an ANN index to a recall budget
#
# **35 minutes.** You will:
#
# 1. Build an **exact** index and treat it as ground truth
# 2. Implement `recall_at_k` yourself — four lines, and it demystifies the metric
# 3. Sweep `efSearch` and **plot the recall/QPS curve** you saw modelled on the slides
# 4. Pay the build-time cost of `M` and `efConstruction`
# 5. Trade memory for accuracy with IVF-PQ
# 6. **Challenge:** hit recall@10 ≥ 0.95 at the lowest latency you can
#
# Corpus: **BEIR SciFact** — 5,183 scientific abstracts, 300 judged queries.
# Embeddings are precomputed, so nothing here waits on a model.
#
# Cells marked `TASK` are yours to write. Every `CHECKPOINT` cell will fall back
# to a reference implementation if you're still working, so you can never be
# blocked — but write it yourself first.

# %%
import sys
from pathlib import Path

# make wslib importable whether you launched from labs/ or from this folder
ROOT = Path.cwd()
if not (ROOT / "wslib").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from wslib import corpus as corpus_mod
from wslib import embed, indexes, metrics
from wslib.timing import bytes_human, stopwatch, time_queries

np.set_printoptions(precision=3, suppress=True)

c = corpus_mod.load("scifact")
docs = embed.cached_corpus_embeddings(c)
qids, queries = embed.cached_query_embeddings(c)

print(c.summary())
print(f"corpus vectors {docs.shape} · query vectors {queries.shape}")
print(f"raw vectors in RAM: {bytes_human(docs.nbytes)}")

# %% [markdown]
# ## 1 · Exact search is the yardstick
#
# Brute force gets perfect index recall by definition — it *is* the right answer.
# Everything approximate gets scored against it.
#
# Note our vectors are unit-norm, so inner product **is** cosine. That's the
# Module 01 identity doing real work: it lets us build ground truth with an
# inner-product index and compare it against an L2-metric HNSW graph, because on
# unit vectors the two produce identical rankings.

# %%
with stopwatch("build exact index"):
    exact = indexes.ExactIndex(docs)

truth = indexes.ground_truth(docs, queries, k=100)
print(f"ground truth shape {truth.shape}")

t_exact = time_queries(lambda q: exact.search(q[None, :], 10), list(queries))
print(f"exact search: {t_exact}")

# What does exact search actually achieve against the human judgements?
# This is the CEILING. No index can beat it — the best it can do is match it.
run_exact = exact.search(queries, 10).to_run(c, qids)
ceiling = metrics.evaluate(run_exact, c.qrels, ks=(10,))
print("\nexact search vs human judgements (the ceiling):")
for k, v in ceiling.items():
    print(f"  {k:12} {v:.4f}")

# %% [markdown]
# ### First honest observation: at this scale, exact search is fine
#
# Look at that latency. 5,183 vectors × 384 dimensions is about 8 MB — it fits in
# cache, and a full scan takes a fraction of a millisecond. **An index barely
# helps here**, and you should expect HNSW to be maybe 10× faster rather than the
# 1000× the slides implied at a million vectors.
#
# That is not a flaw in the lab, it's the lesson from the "brute force doesn't
# scale" slide arriving from the other direction: below roughly 100K vectors,
# adding an ANN index buys you complexity and costs you recall for very little
# speed. We use a small corpus so the whole lab runs in 35 minutes — but keep the
# *shape* of the curves, not the absolute numbers.

# %% [markdown]
# **Now stop and look at this.** Exact search — perfect index recall, no
# approximation at all — gets recall@10 ≈ 0.79 against the golden set.
#
# The other 21% isn't the index's fault. It's the *embedding model* failing to
# put the right document near the query. Keep that number in mind: it's the
# difference between "my index is good" and "my search is good", and we'll come
# back to it at the end.

# %% [markdown]
# ## 2 · TASK: implement index recall@k
#
# Two different recalls live in this lab, and confusing them is the classic
# mistake:
#
# | | measured against | answers |
# |---|---|---|
# | **index recall@k** | exact search | "is my approximation losing results?" |
# | **retrieval recall@k** | human judgements | "is my search any good?" |
#
# You're implementing the first one. For each query, what fraction of the exact
# top-k did the approximate search also return?

# %%
def my_recall_at_k(approx_ids: np.ndarray, truth_ids: np.ndarray, k: int) -> float:
    """Mean overlap between approximate and exact top-k, across all queries.

    approx_ids, truth_ids: (n_queries, >=k) arrays of row indices, best-first.
    Returns a float in [0, 1].
    """
    # YOUR CODE HERE — set overlap per row, then average.
    raise NotImplementedError("implement my_recall_at_k")


# %% [markdown]
# ### CHECKPOINT 1

# %%
def _resolve(fn, reference):
    """Use the attendee's function, or fall back so nobody stalls."""
    try:
        probe_a = np.array([[1, 2, 3], [4, 5, 6]])
        probe_b = np.array([[1, 2, 9], [4, 7, 8]])
        got = fn(probe_a, probe_b, 3)
        want = reference(probe_a, probe_b, 3)
        assert abs(got - want) < 1e-9, f"expected {want}, got {got}"
        print("CP1 PASS — using your implementation")
        return fn
    except NotImplementedError:
        print("CP1 — not implemented yet, using the reference so you can continue")
        return reference
    except AssertionError as e:
        print(f"CP1 FAIL — {e}\n   using the reference so you can continue")
        return reference


recall_at_k = _resolve(my_recall_at_k, indexes.recall_vs_truth)

# hand-checkable cases
a = np.array([[1, 2, 3]])
print("identical      ->", recall_at_k(a, a, 3), "(expect 1.0)")
print("2 of 3 shared  ->", recall_at_k(a, np.array([[1, 2, 9]]), 3), "(expect ~0.667)")
print("nothing shared ->", recall_at_k(a, np.array([[7, 8, 9]]), 3), "(expect 0.0)")

# %% [markdown]
# ## 3 · Build HNSW and see what the approximation costs

# %%
with stopwatch("build HNSW (M=16, efConstruction=200)") as sw_build:
    hnsw = indexes.HNSWIndex(docs, M=16, ef_construction=200)

print(hnsw.describe())
print(f"index memory: {bytes_human(hnsw.total_bytes)} "
      f"({hnsw.bytes_per_vector:.0f} B/vector vs {docs.shape[1] * 4} B raw)")

for ef in (10, 32, 64, 128):
    hnsw.ef = ef
    ids = hnsw.search(queries, 10).ids
    r = recall_at_k(ids, truth, 10)
    t = time_queries(lambda q: hnsw.search(q[None, :], 10), list(queries))
    print(f"  efSearch={ef:4d}  index recall@10 {r:.4f}  {t.median_ms:6.3f} ms  "
          f"{t.qps:7.0f} QPS")

# %% [markdown]
# ## 4 · TASK: sweep efSearch and plot the curve
#
# This is the plot that lets you answer "what will 95% recall cost me?" without
# guessing. Fill in the sweep.

# %%
EF_VALUES = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]

def sweep_ef(index, queries, truth, ef_values, k=10):
    """Return a list of dicts: ef, recall, median_ms, qps."""
    rows = []
    for ef in ef_values:
        # YOUR CODE HERE:
        #   1. set index.ef = ef
        #   2. search all queries, k=k
        #   3. recall_at_k(ids, truth, k)
        #   4. time_queries(...) for latency and QPS
        #   5. append a dict with those numbers
        raise NotImplementedError("implement sweep_ef")
    return rows


# %% [markdown]
# ### CHECKPOINT 2

# %%
def reference_sweep(index, queries, truth, ef_values, k=10):
    rows = []
    for ef in ef_values:
        index.ef = ef
        ids = index.search(queries, k).ids
        r = recall_at_k(ids, truth, k)
        t = time_queries(lambda q: index.search(q[None, :], k), list(queries))
        rows.append({"ef": ef, "recall": r, "median_ms": t.median_ms, "qps": t.qps})
    return rows


try:
    sweep = sweep_ef(hnsw, queries, truth, EF_VALUES)
    print("CP2 PASS — using your sweep")
except NotImplementedError:
    print("CP2 — not implemented yet, using the reference")
    sweep = reference_sweep(hnsw, queries, truth, EF_VALUES)

print(f"\n{'ef':>5} {'recall@10':>11} {'ms':>8} {'QPS':>9}")
for r in sweep:
    print(f"{r['ef']:>5} {r['recall']:>11.4f} {r['median_ms']:>8.3f} {r['qps']:>9.0f}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4))

ax[0].plot([r["recall"] for r in sweep], [r["qps"] for r in sweep], "o-", color="#00b894")
ax[0].set_yscale("log")
ax[0].set_xlabel("index recall@10")
ax[0].set_ylabel("QPS (log, 1 thread)")
ax[0].set_title("The trade-off curve")
ax[0].grid(alpha=0.25)
for r in sweep:
    if r["ef"] in (8, 32, 128, 512):
        ax[0].annotate(f"ef={r['ef']}", (r["recall"], r["qps"]),
                       textcoords="offset points", xytext=(6, 6), fontsize=8)

ax[1].plot([r["ef"] for r in sweep], [r["recall"] for r in sweep], "o-", color="#6c63ff")
ax[1].axhline(0.95, ls="--", c="#ff4d6d", lw=1, label="95% target")
ax[1].set_xscale("log")
ax[1].set_xlabel("efSearch (log)")
ax[1].set_ylabel("index recall@10")
ax[1].set_title("Diminishing returns")
ax[1].legend()
ax[1].grid(alpha=0.25)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Read the curve
#
# Find the cheapest `efSearch` that clears each target. The gap between them is
# the whole point of the lab.

# %%
for target in (0.80, 0.90, 0.95, 0.99, 0.999):
    ok = [r for r in sweep if r["recall"] >= target]
    if ok:
        best = min(ok, key=lambda r: r["median_ms"])
        print(f"recall >= {target:.3f}  ->  ef={best['ef']:4d}  "
              f"{best['median_ms']:6.3f} ms  {best['qps']:7.0f} QPS")
    else:
        print(f"recall >= {target:.3f}  ->  not reached with ef <= {max(EF_VALUES)}")

# %% [markdown]
# ## 5 · M and efConstruction are build-time decisions
#
# `efSearch` you can change per query. These two you cannot — changing them
# means rebuilding the index. Watch the build time and the memory move.

# %%
def recall_at_ef(index, ef, k=10):
    """Index recall at a given efSearch — used to compare builds fairly."""
    index.ef = ef
    return recall_at_k(index.search(queries, k).ids, truth, k)


print(f"{'M':>4} {'efC':>5} {'build s':>9} {'B/vec':>7} {'total':>9} "
      f"{'r@ef=32':>9} {'r@ef=128':>10}")
for M, efC in [(8, 100), (16, 100), (16, 200), (32, 200), (32, 400)]:
    with stopwatch(quiet=True) as sw:
        idx = indexes.HNSWIndex(docs, M=M, ef_construction=efC)
    r32 = recall_at_ef(idx, 32)
    r128 = recall_at_ef(idx, 128)
    print(f"{M:>4} {efC:>5} {sw.seconds:>9.2f} {idx.bytes_per_vector:>7.0f} "
          f"{bytes_human(idx.total_bytes):>9} {r32:>9.4f} {r128:>10.4f}")

# %% [markdown]
# **What to notice:** raising `efConstruction` improves recall at a *fixed*
# `efSearch` and costs you nothing at query time — it only costs build time.
# Raising `M` improves recall too, but it is permanently in your memory bill.

# %% [markdown]
# ## 6 · IVF-PQ: trading memory for accuracy
#
# HNSW stores full vectors plus edges. IVF-PQ stores *compressed* vectors and
# only searches a few partitions. Different bet entirely.

# %%
nlist = int(np.sqrt(len(docs)))          # the usual rule of thumb
print(f"nlist = sqrt(N) = {nlist}\n")

with stopwatch("train + build IVF-PQ"):
    ivfpq = indexes.IVFPQIndex(docs, nlist=nlist, m=48, nbits=8)
print(ivfpq.describe())
print(f"memory: {bytes_human(ivfpq.total_bytes)} "
      f"({ivfpq.bytes_per_vector:.0f} B/vector, {ivfpq.compression:.0f}x smaller than raw)")
print(f"  vs HNSW: {bytes_human(hnsw.total_bytes)} ({hnsw.bytes_per_vector:.0f} B/vector)")

print(f"\n{'nprobe':>7} {'recall@10':>11} {'ms':>8} {'QPS':>9}")
ivf_rows = []
for nprobe in (1, 2, 4, 8, 16, 32, nlist):
    ivfpq.nprobe = nprobe
    ids = ivfpq.search(queries, 10).ids
    r = recall_at_k(ids, truth, 10)
    t = time_queries(lambda q: ivfpq.search(q[None, :], 10), list(queries))
    ivf_rows.append({"nprobe": nprobe, "recall": r, "median_ms": t.median_ms, "qps": t.qps})
    print(f"{nprobe:>7} {r:>11.4f} {t.median_ms:>8.3f} {t.qps:>9.0f}")

# %% [markdown]
# ### Which half of that loss was partitioning, and which was compression?
#
# `IVFFlat` partitions exactly like `IVFPQ` but stores **uncompressed** vectors.
# Run the same nprobe sweep on it and the two losses separate cleanly.

# %%
ivfflat = indexes.IVFFlatIndex(docs, nlist=nlist)
print(f"{'nprobe':>7} {'IVFFlat':>10} {'IVF-PQ':>10}   (index recall@10)")
for nprobe in (1, 4, 8, 32, nlist):
    ivfflat.nprobe = nprobe
    ivfpq.nprobe = nprobe
    rf = recall_at_k(ivfflat.search(queries, 10).ids, truth, 10)
    rp = recall_at_k(ivfpq.search(queries, 10).ids, truth, 10)
    print(f"{nprobe:>7} {rf:>10.4f} {rp:>10.4f}")

print(f"\nIVFFlat memory: {bytes_human(ivfflat.total_bytes)} "
      f"({ivfflat.bytes_per_vector:.0f} B/vec)")
print(f"IVF-PQ  memory: {bytes_human(ivfpq.total_bytes)} "
      f"({ivfpq.bytes_per_vector:.0f} B/vec)")

# %% [markdown]
# **There it is.** At `nprobe = nlist` — searching *every* partition, so no
# partitioning loss remains — IVFFlat reaches **1.0000** and IVF-PQ stops around
# **0.73**.
#
# * **Partitioning loss is recoverable.** Turn nprobe up and you buy it all back.
# * **Quantization loss is not.** No query-time parameter recovers information
#   the codebooks threw away at build time.
#
# That asymmetry is the whole reason production systems do a **compressed wide
# pass followed by an exact rescore** — you get the memory win from compression
# and the ordering back from the full-precision vectors.

# %%
fig, ax = plt.subplots(figsize=(6.5, 4.4))
ax.plot([r["recall"] for r in sweep], [r["qps"] for r in sweep],
        "o-", label=f"HNSW ({hnsw.bytes_per_vector:.0f} B/vec)", color="#00b894")
ax.plot([r["recall"] for r in ivf_rows], [r["qps"] for r in ivf_rows],
        "s-", label=f"IVF-PQ ({ivfpq.bytes_per_vector:.0f} B/vec)", color="#ffd166")
ax.axhline(t_exact.qps, ls=":", c="#9aa0c4", lw=1.2,
           label=f"exact ({t_exact.qps:.0f} QPS, recall 1.0)")
ax.set_yscale("log")
ax.set_xlabel("index recall@10")
ax.set_ylabel("QPS (log, 1 thread)")
ax.set_title("Same axes, different bets")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7 · CHALLENGE — post your number
#
# **Hit index recall@10 ≥ 0.95 at the lowest median latency you can.**
#
# Anything is fair: `M`, `efConstruction`, `efSearch`, IVF-PQ parameters, a
# different `nlist`, `m`, `nbits`. Post `(config, recall, ms)` and we'll build a
# leaderboard.

# %%
TARGET = 0.95

def score_config(index, label, k=10):
    ids = index.search(queries, k).ids
    r = recall_at_k(ids, truth, k)
    t = time_queries(lambda q: index.search(q[None, :], k), list(queries))
    ok = "OK " if r >= TARGET else "   "
    print(f"{ok}{label:<44} recall {r:.4f}  {t.median_ms:6.3f} ms  "
          f"{bytes_human(index.total_bytes):>9}")
    # keep the results as scored: the index object may be retuned afterwards
    return {"label": label, "recall": r, "ms": t.median_ms, "bytes": index.total_bytes,
            "run": index.search(queries, k).to_run(c, qids)}


results = []
hnsw.ef = 64
results.append(score_config(hnsw, "HNSW M=16 efC=200 ef=64"))

# YOUR TURN — try to beat that latency while staying above 0.95.
# Some directions worth trying:
#   indexes.HNSWIndex(docs, M=32, ef_construction=400)   # denser graph, lower ef
#   indexes.IVFFlatIndex(docs, nlist=128)                # partition without compressing
#   indexes.IVFPQIndex(docs, nlist=64, m=96, nbits=8)    # finer sub-vectors

# %% [markdown]
# ## 8 · The number that matters
#
# Take your winning config and score it against the **human judgements** —
# the thing users actually experience.

# %%
best = min((r for r in results if r["recall"] >= TARGET),
           key=lambda r: r["ms"], default=results[0])
print(f"best config: {best['label']}  (index recall {best['recall']:.4f})\n")

comparison = {
    "exact (ceiling)": metrics.evaluate(run_exact, c.qrels, ks=(10,)),
    best["label"]: metrics.evaluate(best["run"], c.qrels, ks=(10,)),
}
hnsw.ef = 8
comparison["hnsw ef=8 (sloppy)"] = metrics.evaluate(
    hnsw.search(queries, 10).to_run(c, qids), c.qrels, ks=(10,))
ivfpq.nprobe = 4
comparison["ivf_pq nprobe=4"] = metrics.evaluate(
    ivfpq.search(queries, 10).to_run(c, qids), c.qrels, ks=(10,))

print(metrics.leaderboard(comparison, ks=(10,)))

# %% [markdown]
# ## What to take away
#
# 1. **`efSearch` is your production dial.** `M` and `efConstruction` are
#    architecture decisions you make once.
# 2. **The curve has a cliff.** Compare the cost of 0.90 with the cost of 0.99
#    from section 4. Pick a recall *budget* from the product requirement, then
#    read the parameter off the curve — never the other way round.
# 3. **Partitioning loss is recoverable, quantization loss is not.** nprobe buys
#    back the first; nothing buys back the second.
# 4. **Index recall is not retrieval quality.** You can push index recall to
#    1.000 and still only reach recall@10 ≈ 0.79 against the golden set, because
#    that ceiling belongs to the *embedding model*.
#
# That last point is the bridge into Module 03: if the index is already perfect
# and search is still missing a fifth of the answers, tuning the index harder is
# not the fix. **Retrieving differently is.**
#
# ### Your measured curve won't match the slides
#
# The deck's curve came from a smooth model. Yours came from 5,183 real vectors.
# Where do they disagree, and why? (Hint: corpus size, dimensionality, and how
# clustered SciFact abstracts actually are.)
