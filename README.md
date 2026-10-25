# Meaning as Geometry — Semantic Search from First Principles to Production

Hands-on labs for the CodeMash 2027 half-day precompiler.

Embeddings, vector search, hybrid retrieval, and RAG — built, tuned, and
**measured**. CPU only: no GPU, no API key, no server, no cloud account.

## Open a lab in Colab

Nothing to install. Run the first cell and it fetches the code and data for you.

| | Lab | What you do |
|---|---|---|
| 1 | [**Meaning as geometry**](https://colab.research.google.com/github/rsp3339/vectors-workshop_codemash_2027/blob/main/labs/lab1_geometry/lab1.ipynb) | Normalize, compare pooling, truncate MRL embeddings and measure the loss |
| 2 | [**Tuning to a budget**](https://colab.research.google.com/github/rsp3339/vectors-workshop_codemash_2027/blob/main/labs/lab2_ann_tuning/lab2.ipynb) | Build exact ground truth, sweep HNSW and IVF to hit a recall target |
| 3 | [**Hybrid search**](https://colab.research.google.com/github/rsp3339/vectors-workshop_codemash_2027/blob/main/labs/lab3_hybrid/lab3.ipynb) | Implement RRF in five lines, then break score-based fusion on purpose |
| 4 | [**Evaluated RAG**](https://colab.research.google.com/github/rsp3339/vectors-workshop_codemash_2027/blob/main/labs/lab4_rag_eval/lab4.ipynb) | Score four configurations, triage failures, learn what you can claim |

## Or run them locally

```bash
cd labs
python3.12 -m venv .venv && source .venv/bin/activate      # 3.11 or 3.12
pip install -r requirements.txt
python scripts/prepare_data.py --poolings                  # SciFact  (labs 1-3)
python scripts/prepare_data.py --dataset nfcorpus --all    # NFCorpus (lab 4)
python scripts/verify_setup.py                             # prints SETUP OK
```

**Linux / Windows:** install the CPU torch wheel first, or pip pulls several GB of
CUDA you will never use:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then open the notebooks in Jupyter or VS Code, or run a lab as a script:

```bash
MPLBACKEND=Agg python lab2_ann_tuning/lab2.py
```

## What's here

```
labs/
  wslib/            corpus · embeddings · indexes · BM25 · fusion
                    chunking · reranking · metrics · timing
  lab1_geometry/    lab2_ann_tuning/    lab3_hybrid/    lab4_rag_eval/
  scripts/          prepare_data · verify_setup · build_notebooks
                    workshop_config · make_data_archive
  tests/            hand-checked metric assertions
```

Corpora are public BEIR benchmarks — **SciFact** (binary judgements, ~1 relevant
document per query) and **NFCorpus** (graded judgements, 38 per query) — chosen so
the same metrics behave differently on each. Exact-search baselines match
published BEIR figures for `all-MiniLM-L6-v2`, which is how you know the pipeline
is wired correctly.

Full detail, including every measured result, is in [`labs/README.md`](labs/README.md).

## A note on the results

Several labs contradict the standard advice. A widely-cited default parameter
costs accuracy on one corpus and helps on another; a "reliable" technique tops the
leaderboard and still fails a significance test. Those aren't mistakes in the
labs — they are the point. Measure your own corpus.

## License

Code is provided for workshop use. The BEIR corpora retain their original
licenses; see the [BEIR project](https://github.com/beir-cellar/beir).
