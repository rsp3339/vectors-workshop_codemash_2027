"""
Corpus loading — one shared dataset for all four labs.

Default is BEIR **SciFact**: 5,183 scientific abstracts and 300 judged test
queries. It is small enough to embed on a laptop CPU and real enough that the
numbers mean something, and its queries mix literal terminology with paraphrase,
which is exactly the contrast Lab 3 needs.

Everything is cached under labs/data/ after the first download, so the labs run
offline afterwards.

Note on grades: SciFact judgements are **binary** (every relevant document is
grade 1). nDCG still works and is still position-sensitive, but the graded part
of the metric has nothing to chew on. Pass dataset="nfcorpus" for genuinely
graded (0/1/2) judgements if you want to demonstrate that.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlopen

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATASETS = {
    "scifact": {"split": "test", "docs": 5183, "queries": 300, "graded": False},
    "nfcorpus": {"split": "test", "docs": 3633, "queries": 323, "graded": True},
}


@dataclass
class Corpus:
    """A retrieval dataset, flattened into parallel lists.

    doc_ids[i] identifies the document whose text is texts[i]; that index is the
    integer id used by every index we build, which keeps faiss, BM25 and the
    qrels all talking about the same thing.
    """

    name: str
    doc_ids: List[str]
    titles: List[str]
    bodies: List[str]
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]
    graded: bool = False
    _pos: Dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._pos = {d: i for i, d in enumerate(self.doc_ids)}

    # ── text ──────────────────────────────────────────────────────────────

    @property
    def texts(self) -> List[str]:
        """What we actually embed: title and body together.

        BEIR convention, and it matters — the title carries a lot of the signal
        in an abstract-length document.
        """
        return [
            (t + ". " + b).strip() if t else b
            for t, b in zip(self.titles, self.bodies)
        ]

    def __len__(self) -> int:
        return len(self.doc_ids)

    def pos(self, doc_id: str) -> Optional[int]:
        """Row index for a document id, or None if it isn't in the corpus."""
        return self._pos.get(doc_id)

    def ids(self, positions) -> List[str]:
        """Map index rows back to document ids."""
        return [self.doc_ids[i] for i in positions if 0 <= i < len(self.doc_ids)]

    def doc(self, doc_id: str) -> str:
        i = self._pos[doc_id]
        return self.texts[i]

    # ── judged queries ────────────────────────────────────────────────────

    def judged_queries(self) -> Dict[str, str]:
        """Only the queries that have at least one relevant document.

        BEIR ships some queries with no positive judgements; including them
        would silently drag every mean down.
        """
        return {
            qid: q
            for qid, q in self.queries.items()
            if qid in self.qrels and any(g > 0 for g in self.qrels[qid].values())
        }

    def summary(self) -> str:
        jq = self.judged_queries()
        rel = [sum(1 for g in self.qrels[q].values() if g > 0) for q in jq]
        return (
            f"{self.name}: {len(self):,} documents · {len(jq)} judged queries · "
            f"{sum(rel) / max(len(rel), 1):.1f} relevant per query · "
            f"{'graded' if self.graded else 'binary'} judgements"
        )


# ── download & parse ───────────────────────────────────────────────────────


def _fetch(name: str, data_dir: Path) -> Path:
    """Download the BEIR zip once and unpack it into data_dir."""
    target = data_dir / name
    if (target / "corpus.jsonl").exists():
        return target

    data_dir.mkdir(parents=True, exist_ok=True)
    url = BEIR_URL.format(name=name)
    print(f"downloading {name} from BEIR ...", file=sys.stderr)

    with urlopen(url, timeout=120) as resp:
        blob = resp.read()

    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(data_dir)

    if not (target / "corpus.jsonl").exists():
        raise RuntimeError(f"unexpected archive layout for {name} under {data_dir}")
    print(f"  unpacked to {target}", file=sys.stderr)
    return target


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load(
    dataset: str = "scifact",
    data_dir: Path | str = DATA_DIR,
    max_docs: Optional[int] = None,
) -> Corpus:
    """Load a BEIR dataset, downloading it on first use.

    max_docs truncates the corpus for a fast smoke test. It keeps every document
    that appears in the qrels first, so the golden set stays intact and recall
    numbers remain meaningful — just easier.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; try {sorted(DATASETS)}")

    data_dir = Path(data_dir)
    root = _fetch(dataset, data_dir)
    split = DATASETS[dataset]["split"]

    # queries
    queries = {r["_id"]: r["text"] for r in _read_jsonl(root / "queries.jsonl")}

    # qrels
    qrels: Dict[str, Dict[str, int]] = {}
    with (root / "qrels" / f"{split}.tsv").open(encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader, None)          # query-id  corpus-id  score
        for row in reader:
            if len(row) < 3:
                continue
            qid, did, score = row[0], row[1], int(row[2])
            qrels.setdefault(qid, {})[did] = score

    # keep only queries for this split
    queries = {q: t for q, t in queries.items() if q in qrels}

    # corpus
    doc_ids: List[str] = []
    titles: List[str] = []
    bodies: List[str] = []
    for r in _read_jsonl(root / "corpus.jsonl"):
        doc_ids.append(r["_id"])
        titles.append(r.get("title", "") or "")
        bodies.append(r.get("text", "") or "")

    corpus = Corpus(
        name=dataset,
        doc_ids=doc_ids,
        titles=titles,
        bodies=bodies,
        queries=queries,
        qrels=qrels,
        graded=DATASETS[dataset]["graded"],
    )

    if max_docs is not None and max_docs < len(corpus):
        corpus = _truncate(corpus, max_docs)
    return corpus


def _truncate(c: Corpus, max_docs: int) -> Corpus:
    """Shrink the corpus while keeping every judged document."""
    must_keep = {d for rel in c.qrels.values() for d, g in rel.items() if g > 0}
    keep_idx = [i for i, d in enumerate(c.doc_ids) if d in must_keep]
    if len(keep_idx) > max_docs:
        raise ValueError(
            f"max_docs={max_docs} is below the {len(keep_idx)} judged documents"
        )
    seen = set(keep_idx)
    for i in range(len(c.doc_ids)):
        if len(seen) >= max_docs:
            break
        seen.add(i)
    order = sorted(seen)
    return Corpus(
        name=f"{c.name}[{max_docs}]",
        doc_ids=[c.doc_ids[i] for i in order],
        titles=[c.titles[i] for i in order],
        bodies=[c.bodies[i] for i in order],
        queries=c.queries,
        qrels=c.qrels,
        graded=c.graded,
    )
