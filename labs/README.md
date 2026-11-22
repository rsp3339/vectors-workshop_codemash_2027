# Workshop labs — Meaning as Geometry

Hands-on labs for *Semantic Search from First Principles to Production*.

CPU only. No GPU, no API key, no server, no cloud account.

```
labs/
  wslib/                  shared plumbing — corpus, embeddings, indexes, metrics
  lab1_geometry/          embeddings, pooling, MRL, cross-encoder  <- built
  lab2_ann_tuning/        recall/latency tuning        <- built
  lab3_hybrid/            BM25 + RRF fusion            <- built
  lab4_rag_eval/          golden set, nDCG, RAG triage  <- built
  scripts/                prepare_data · verify_setup · build_notebooks
                          workshop_config · make_data_archive
  tests/                  hand-checked metric tests
  data/                   downloaded corpus + cached embeddings (gitignored)
```

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # 3.11 or 3.12
pip install -r requirements.txt
python scripts/prepare_data.py --poolings                # SciFact + poolings + MRL (labs 1-3)
python scripts/prepare_data.py --dataset nfcorpus --all  # NFCorpus + chunks + rerank (lab 4)
python scripts/verify_setup.py                          # prints SETUP OK
```

**Linux / Windows:** install the CPU torch wheel first or pip will pull several GB
of CUDA you will never use:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

`prepare_data.py` is the only step that needs the network, and the only slow one:
about a minute for SciFact, and roughly 8 minutes for `--dataset nfcorpus --all`
(chunk embeddings plus ~9,700 cross-encoder pairs). After that every lab runs
from cache in **5-13 seconds**.

### Known issue: faiss + torch OpenMP (read this before debugging a crash)

pip-installed `faiss-cpu` and `torch` each bundle their own `libomp`. A process
using both — which Lab 4 does — segfaults (exit 139) or deadlocks. Measured on
macOS + Python 3.12:

| `OMP_NUM_THREADS` | result |
|---|---|
| unset | hangs |
| **1** | **works** |
| 2 | hangs |
| 4 | hangs |

`wslib/__init__.py` sets `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` at
import time, which is the only thing that works — the Python APIs
(`torch.set_num_threads`, `faiss.omp_set_num_threads`) run too late, after the
runtimes have initialised. **So always `import wslib` before faiss or torch.**
Single-threaded is no real loss: the cross-encoder still does ~35 pairs/sec, and
Lab 2's QPS figures claim one thread, so now they are honest.

## Corpus

**BEIR SciFact** — 5,183 scientific abstracts, 300 judged queries, ~1.1 relevant
documents per query. Small enough to embed on a laptop, real enough that the
numbers mean something.

Reference numbers with `all-MiniLM-L6-v2`, exact search:

| metric | value |
|---|---|
| recall@10 | 0.7883 |
| nDCG@10 | 0.6484 |
| MRR@10 | 0.6068 |

That nDCG matches published BEIR figures for this model, which is how you know
the pipeline is wired correctly. **It is also the ceiling** — no index can beat
exact search, so every lab number should be read against it.

> **Judgements are binary.** SciFact marks relevant documents grade 1, so nDCG
> is position-sensitive but has no grading to exploit. Everything in `wslib`
> handles graded relevance; pass `dataset="nfcorpus"` for genuinely graded
> (0/1/2) judgements if you want to demonstrate that in Lab 4.

## Running the labs

Labs are authored as `# %%`-delimited Python so they diff cleanly and can be run
top-to-bottom in CI. Generate notebooks for attendees:

```bash
python scripts/build_notebooks.py         # lab*.py -> lab*.ipynb (+ Colab bootstrap)
python scripts/build_notebooks.py --links # just print the Colab URLs
```

Each notebook gets a two-cell bootstrap that clones the repo, installs
dependencies and fetches the data cache **on Colab**, and is a no-op on a local
Jupyter install (it walks up from the notebook to find `wslib/`). It ends by
running `verify_setup.py`, so the first thing an attendee sees is `SETUP OK`.

### Publishing for attendees

Everything publication-related lives in **`scripts/workshop_config.py`** — set it
once and both the bootstrap cells and the Colab links follow.

```bash
# 1. push labs/ to a public GitHub repo, then:
#    edit REPO in scripts/workshop_config.py   (or export WORKSHOP_REPO)

# 2. build the ~113 MB data cache archive so Colab users skip prepare_data
python scripts/make_data_archive.py
gh release create workshop-data workshop-data.tar.gz --notes 'lab data cache'
#    paste the asset's download URL into DATA_URL in workshop_config.py

# 3. regenerate the notebooks with those values baked in
python scripts/build_notebooks.py
python tests/test_colab_bootstrap.py   # a bad DATA_URL fails at the download
```

Step 3 prints the four Colab links for the setup slide and the pre-work email.
Until `REPO` is set, the notebooks build fine but refuse to bootstrap in Colab
with an explicit instructor message rather than an obscure failure.

Run one as a script to check it still works:

```bash
MPLBACKEND=Agg python lab2_ann_tuning/lab2.py
```

Cells marked `TASK` are for attendees to write; each `CHECKPOINT` falls back to a
reference implementation and says so, so nobody is ever blocked by an earlier
step.

## wslib

| module | what's in it |
|---|---|
| `corpus` | BEIR download + parse, `judged_queries()`, id↔row mapping |
| `embed` | model loading, unit-norm encode, on-disk `.npy` cache, MRL truncation |
| `indexes` | `ExactIndex`, `HNSWIndex`, `IVFFlatIndex`, `IVFPQIndex`, `ground_truth` |
| `sparse` | `BM25Index` (Okapi BM25, plain tokenizer) |
| `fusion` | `rrf`, `linear_blend` (raw or min-max), `scale_scores` |
| `metrics` | `recall_at_k`, `reciprocal_rank`, `ndcg_at_k`, `evaluate`, `bootstrap_delta` |
| `timing` | warm-up-aware `time_queries` (median + p95 + QPS), `stopwatch` |
| `chunking` | `chunk_corpus` (whole/fixed/sentence), `pool_to_parents` |
| `rerank` | `CrossEncoderReranker` with an on-disk score cache |
| `embed` (poolings) | `encode_poolings` — mean/CLS/max from one forward pass |
| `quiet` | fd-level suppression of faiss's C++ warnings |

Two design decisions the labs depend on:

**Everything is unit-norm.** So inner product *is* cosine, and ranking by L2 is
identical to ranking by cosine — that's the Module 01 identity, and it's why we
can build ground truth with `IndexFlatIP` and score an L2-metric HNSW graph
against it without cheating.

**Two different recalls.** `indexes.recall_vs_truth` measures the *index* against
exact search ("is my approximation losing results?"). `metrics.recall_at_k`
measures *retrieval* against human judgements ("is my search any good?").
Conflating them is the classic mistake, and Lab 2 makes the distinction
explicitly.

```bash
python tests/test_metrics.py      # 25 hand-checked assertions
```

## Lab 2 — what the real numbers show

Measured on this corpus, not modelled:

**The cliff is real.** Cheapest `efSearch` clearing each target:

| target | efSearch | median | QPS |
|---|---|---|---|
| 0.90 | 12 | 0.015 ms | 65,000 |
| 0.95 | 24 | 0.024 ms | 41,000 |
| 0.99 | 48 | 0.037 ms | 27,000 |
| 0.999 | 192 | 0.122 ms | 8,200 |

**Partitioning loss is recoverable, quantization loss is not.** Same `nlist`,
same nprobe sweep, only compression differs:

| nprobe | IVFFlat | IVF-PQ |
|---|---|---|
| 1 | 0.4847 | 0.4393 |
| 8 | 0.8713 | 0.6860 |
| 71 (all cells) | **1.0000** | **0.7293** |

At 48 B/vector versus 1536, that's a 32× memory win for ~27 points of recall
you cannot buy back at query time — which is exactly why production systems do a
compressed wide pass and then rescore with full-precision vectors.

**A perfect index does not fix a mediocre retriever.** HNSW at `ef=64` reaches
0.9937 index recall and scores *identically to exact search* against the
judgements (recall@10 0.7883). The missing 21% belongs to the embedding model,
not the index — which is the bridge into Module 03.

### Caveat to state out loud

At 5K vectors the whole corpus is ~8 MB and fits in cache, so exact search runs
in 0.12 ms and HNSW is only ~10× faster, not 1000×. The corpus is small so the
lab fits in 35 minutes. **Read the shape of the curves, not the absolute
numbers** — and note this is the "don't add an index below 100K vectors" lesson
arriving from the other direction.


## Lab 3 — what the real numbers show

The measured results **contradict the received wisdom**, and the lab now teaches
what actually happened rather than what the slides predicted.

**The retriever split is real.** Of 300 queries: dense wins 72, BM25 wins 71,
157 tie. Hybrid isn't a preference, it's arithmetic.

**Every fusion method beats both single retrievers** — that's the reliable win.
But which fusion method wins depends on what you're robust to:

| method | nDCG@10 | after BM25 ×100 | after outlier | tuning |
|---|---|---|---|---|
| dense only | 0.6484 | — | — | — |
| bm25 only | 0.6523 | — | — | — |
| linear raw (α=0.99) | 0.7038 | **0.6575** ✗ | — | 1 weight, fitted |
| linear min-max (α=0.5) | **0.7079** | 0.7079 ✓ | **0.6719** ✗ | 1 weight, fitted |
| RRF k=60 | 0.6793 | 0.6793 ✓ | 0.6793 ✓ | none |
| RRF k=1 | 0.7006 | 0.7006 ✓ | 0.7006 ✓ | 1 constant |

Three findings worth knowing before you teach this:

1. **RRF does not simply win.** A tuned min-max blend scores slightly higher
   here. The honest claim is that RRF gets within a whisker with nothing fitted,
   and is the only method that survives *both* perturbations.
2. **`k=60` is not optimal on SciFact — `k=1` is**, worth ~2 nDCG points. The
   mechanism: `k` damps the head of each list, which is right when several
   documents are relevant, and wrong when there is ~1 relevant document per query
   and a retriever's rank-1 confidence is the signal you wanted.
3. **Adding a third, weaker retriever makes RRF worse** (0.6793 → 0.6217 with a
   title-only BM25 scoring 0.4135 alone). RRF gives every list an equal vote —
   that's what makes it tuning-free, and it means a weak retriever drags the
   fusion toward its mistakes. The fix is retriever *selection*, not weights.

Min-max normalisation being scale-invariant is why the lab adds a second
perturbation: injecting one inflated top-1 score per query. Min-max maps scores
using that query's own maximum, so an outlier compresses everything else toward
zero — it drops 0.036 nDCG. RRF doesn't move.

## Lab 4 corpus — NFCorpus

Lab 4 uses **NFCorpus** rather than SciFact, because it has genuinely graded
judgements and enough relevant documents per query for nDCG to be interesting.

| | SciFact | NFCorpus |
|---|---|---|
| documents | 5,183 | 3,633 |
| judged queries | 300 | 323 |
| grades | binary (1) | **graded (1 and 2)** |
| relevant per query | 1.1 | 38.2 (median 16, max 475) |
| exact nDCG@10 | 0.6484 | 0.3188 |
| exact recall@10 | 0.7883 | 0.1589 |

Both nDCG figures match published BEIR results for `all-MiniLM-L6-v2`, which is
how we know the pipeline is right.

**That 0.1589 is correct, not broken.** With ~38 relevant documents per query,
recall@10 is capped near 0.26 by construction — you cannot retrieve 38 documents
in 10 slots. NFCorpus therefore demonstrates the "which metric when" lesson
concretely: recall@k is the wrong headline metric on a corpus like this, and
nDCG@k is the right one. `prepare_data.py` says so in its output.


## Lab 4 — what the real numbers show

On NFCorpus, graded judgements, 323 queries:

| config | recall@10 | nDCG@10 | MRR@10 |
|---|---|---|---|
| dense | 0.1589 | 0.3188 | 0.5068 |
| bm25 | 0.1522 | 0.3071 | 0.5085 |
| hybrid rrf k=1 | **0.1698** | 0.3415 | 0.5442 |
| hybrid rrf k=60 | 0.1639 | 0.3426 | 0.5467 |
| hybrid + rerank | 0.1668 | **0.3526** | **0.5702** |
| chunked sentence-512 | 0.1610 | 0.3345 | 0.5270 |
| chunked fixed-256 | 0.1552 | 0.3090 | 0.4822 |

**The `k` advantage disappears, and that's the point.** On SciFact (~1 relevant
document per query) `k=1` beat `k=60` by ~2 nDCG points. Here, with 38 relevant
per query, the difference is +0.0011 for `k=60`, 95% CI [−0.0072, +0.0095] — not
significant. The right value is a property of the corpus, not of RRF.

**Sentence chunking helps, fixed-size chunking hurts.** Same corpus, same
retriever: `sentence-512` cuts 0% of chunks mid-sentence and gains nDCG over
whole documents; `fixed-256` cuts 84% and loses. That's the chunking slide,
measured.

**Reranking cannot move recall.** It reorders existing candidates — recall@10
barely budges while nDCG@10 gains 0.010 and MRR gains 0.024. If a metric doesn't
respond to a reranker, ask whether that metric can see ordering at all.

### The significance result is the best thing in this lab

| change | Δ nDCG@10 | 95% CI | verdict |
|---|---|---|---|
| hybrid + rerank | +0.0100 | [−0.0030, +0.0231] | **cannot claim it** |
| chunked sentence-512 | +0.0157 | [+0.0026, +0.0288] | can claim it |
| chunked fixed-256 | −0.0098 | [−0.0246, +0.0055] | cannot claim it |

**The configuration that topped the leaderboard is not statistically
established**, while a smaller-looking chunking change is. Reranking wins
102 queries and loses 90. This is the workshop's thesis landing on the workshop's
own results, and it's worth teaching exactly as it came out.

### Failure triage, computed not invented

Where the first relevant document actually landed, across all 323 queries:

| failure mode | queries | share |
|---|---|---|
| context missing | 67 | 20.7% |
| context noise | 27 | 8.4% |
| context overflow | 16 | 5.0% |
| retrieval succeeded | 213 | 65.9% |

**110 of 323 queries (34%) never got a judged-relevant document in front of the
model.** That is an upper bound on retrieval failure, not a count of it. Only 6
of the 67 "context missing" queries have a judged document closer to the query
(by the same embedding model) than the one ranked first. Lab 4 shows the clearest
of those ("dietary scoring") beside a suspect judgement ("okra", whose only
relevant document is about steam-cooking greens). Read failures before fixing
them. Note the fourth bucket's honest name — we
never generated answers for those queries, so "generator failure" would be a
claim we cannot support; all we know is retrieval did its job.

### A word about judgement quality

NFCorpus qrels come from NutritionFacts.org citation links, so they are *broad*:
the query "deafness" is judged relevant to documents about ayurvedic medicine.
The id mapping is verified and our nDCG matches published BEIR figures, so this
is the dataset, not a bug. Lab 4 says so explicitly — judgement quality is
another variable in your measurement, not a given.


## Lab 1 — what the real numbers show

**The paraphrase hook holds, and it is checked rather than asserted.** Three
paraphrase pairs, none sharing a content word with its partner
(cancer/malignancy, exercise/physical activity, vitamin D/cholecalciferol). Every
sentence's nearest neighbour is its own paraphrase; pairs score 0.81 / 0.86 /
0.48 against a mean of 0.13 for everything else. A keyword index scores all three
pairs at zero.

**The normalization identity, failing and then holding:**

| | dot | cos | L2 | 1−½L2² | |
|---|---|---|---|---|---|
| a×3.0, b×0.4 | +0.3437 | +0.2864 | 2.9108 | **−3.2363** | MISMATCH |
| unit a, unit b | +0.2864 | +0.2864 | 1.1946 | **+0.2864** | MATCH |

Cosine never moves — it only ever looked at direction. The identity needs the
unit-norm precondition, and the lab makes that visible rather than stating it.

**The silent bug, measured:** inner product on length-scaled (un-normalized)
vectors takes nDCG@10 from **0.6484 to 0.2190**. No error, no warning.

**Pooling costs real points on a model that never complains:**

| pooling | nDCG@10 | recall@10 | vs mean |
|---|---|---|---|
| mean (what it was trained with) | 0.6484 | 0.7883 | — |
| CLS | 0.5993 | 0.7366 | −0.0491 |
| max | 0.4696 | 0.6003 | −0.1788 |

**MRL truncation is real — and is not a quality ranking:**

| truncated to | MRL model | retained | non-MRL (MiniLM) | retained |
|---|---|---|---|---|
| full | 0.5935 (1024d) | 100% | 0.6484 (384d) | 100% |
| 1/2 | 0.5759 | 97.0% | 0.6158 | 95.0% |
| 1/4 | 0.5644 | 95.1% | 0.5572 | 85.9% |
| 1/8 | 0.5350 | **90.1%** | 0.4425 | **68.2%** |
| 1/16 | 0.4587 | 77.3% | 0.2836 | 43.7% |
| 1/32 | 0.3093 | 52.1% | 0.1052 | 16.2% |

The retention curves diverge sharply — MRL genuinely orders dimensions by
importance. **But MiniLM at full width outscores the MRL model at full width on
this corpus**, so the lab says plainly: pick the best model for your task first,
*then* check whether it truncates well. Never the other way round.

**Bi- vs cross-encoder, honestly.** Across 8 queries where the bi-encoder
retrieved a relevant document but buried it, the cross-encoder promoted it in 6,
demoted it in 1, left it in 1. The lab shows the clearest win (rank 10 → 4) and
then states the failure rate — reranking is a strong average bet, not a
guarantee, which is exactly why Lab 4 puts an interval on it.

### The MRL model choice

`sentence-transformers/static-retrieval-mrl-en-v1` rather than
`nomic-embed-text-v1.5`, for three practical reasons: no `trust_remote_code`
(some corporate environments block it) and no `einops`; a fraction of the
download; and being a static model it encodes the whole corpus in ~9s versus
~72s for MiniLM.
