"""
ABOUTME: Runs Labs 2-4 end to end and checks that what each lab prints agrees with
ABOUTME: the numbers it measured, so on-screen claims cannot drift from the data.

Needs the prepared data cache (see README). Takes about 30 seconds.
Run with:  python tests/test_lab_output.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

LABS = Path(__file__).resolve().parent.parent

FAILED = []


def check(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label:<58} {detail}")
    if not ok:
        FAILED.append(label)


def run_lab(folder, script):
    env = dict(os.environ, MPLBACKEND="Agg")
    env.pop("OPENAI_API_KEY", None)   # Lab 4 must take its offline path, not a paid API
    p = subprocess.run([sys.executable, str(LABS / folder / script)],
                       cwd=LABS, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:], sep="\n")
        raise SystemExit(f"{script} exited {p.returncode}")
    return p.stdout


def block(text, start, end):
    """The slice of `text` from the line containing `start` up to `end`."""
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start))
    return text[i:j if j >= 0 else len(text)]


# ── Lab 2: the challenge winner is the config scored against judgements ────
out = run_lab("lab2_ann_tuning", "lab2.py")
m = re.search(r"best config: (.+?)  \(index recall", out)
best_label = m.group(1) if m else None
check("lab2 prints the best challenge config", best_label is not None)
board = out[out.find("best config:"):]
check("lab2 scores that config against the judgements",
      best_label is not None and re.search(rf"^{re.escape(best_label)}\s+0\.\d{{4}}",
                                           board, re.M) is not None,
      f"looking for a leaderboard row named {best_label!r}")

UNITS = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}
entries = [(label.strip(), float(ms), float(size) * UNITS[unit])
           for label, ms, size, unit in re.findall(
               r"^OK (.+?)\s+recall \d\.\d{4}\s+([\d.]+) ms\s+([\d.]+) (B|KB|MB|GB)$", out, re.M)]
smallest = min(entries, key=lambda e: (e[2], e[1]))[0] if entries else None
check("lab2 challenge winner is the smallest index clearing the target",
      len(entries) >= 2 and best_label == smallest,
      f"printed {best_label!r}, smallest qualifying {smallest!r} of {len(entries)}")
lab2_text = (LABS / "lab2_ann_tuning" / "lab2.py").read_text()
check("lab2 text states the smallest-index rule, not the latency one",
      "smallest index" in lab2_text and "lowest latency" not in lab2_text
      and "lowest median latency" not in lab2_text)

exact = re.search(r"exact search:\s+([\d.]+) ms median", out)
target_ms = dict(re.findall(r"recall >= (0\.95|0\.99)0\s+->\s+ef=\s*\d+\s+([\d.]+) ms", out))
speedups = [float(exact.group(1)) / float(ms) for ms in target_ms.values()] \
    if exact and len(target_ms) == 2 else []
for doc in ("lab2_ann_tuning/lab2.py", "README.md"):
    quoted = re.search(r"(\d+)–(\d+)× faster", (LABS / doc).read_text())
    lo, hi = map(int, quoted.groups()) if quoted else (None, None)
    check(f"{doc} quotes the measured HNSW speedup at recall 0.95-0.99",
          bool(quoted and speedups) and all(lo <= round(s) <= hi for s in speedups),
          f"quoted {lo}–{hi}x vs measured "
          + ", ".join(f"{s:.1f}x" for s in speedups))


# ── Lab 3: the BM25 candidate-count remark matches the counts ──────────────
out = run_lab("lab3_hybrid", "lab3.py")
m = re.search(r"bm25  returns (\d+) candidates/query \((\d+) of (\d+) queries "
              r"matched fewer than (\d+) documents\)", out)
check("lab3 reports how many BM25 lists came back short", m is not None)
if m:
    avg, short, total, cap = map(int, m.groups())
    check("lab3 short-list count is consistent with the average",
          (short == 0) == (avg == cap), f"avg {avg}, short {short}/{total}")

mm = re.search(r"^linear min-max \(a=0\.5\)\s+[\d.]+\s+([\d.]+)", out, re.M)
r60 = re.search(r"^RRF k=60\s+[\d.]+\s+([\d.]+)", out, re.M)
gap = round((float(mm.group(1)) - float(r60.group(1))) * 100) if mm and r60 else None
for doc in ("lab3_hybrid/lab3.py", "README.md"):
    text = (LABS / doc).read_text()
    quoted = re.search(r"~(\d+) nDCG points behind", text)
    check(f"{doc} quotes the measured min-max lead over RRF k=60",
          bool(quoted) and int(quoted.group(1)) == gap,
          f"quoted {quoted and quoted.group(1)} vs measured {gap}")
    check(f"{doc} does not call untuned RRF within a whisker",
          "whisker" not in text)


# ── Lab 4 ──────────────────────────────────────────────────────────────────
out = run_lab("lab4_rag_eval", "lab4.py")

# RRF k on NFCorpus: the comparison is tested, not asserted
check("lab4 bootstraps RRF k=60 against k=1",
      re.search(r"Δ nDCG@10 \(hybrid rrf k=60 - hybrid rrf k=1\) [+-]\d\.\d{4}\s+95% CI",
                out) is not None)

# reranker cost is compared with a time the lab actually measured
m_dense = re.search(r"^dense retrieval: ([\d.]+)s", out, re.M)
m_claim = re.search(r"dense retrieval searched all ([\d,]+) documents for (\d+) "
                    r"queries in ([\d.]+)s", out)
check("lab4 compares reranking with measured dense retrieval",
      m_claim is not None)
if m_dense and m_claim:
    check("lab4 quoted dense time matches the stopwatch",
          m_dense.group(1) == m_claim.group(3),
          f"stopwatch {m_dense.group(1)}s, quoted {m_claim.group(3)}s")

# failure examples separate a retrieval miss from a doubtful judgement
genuine = block(out, "a genuine miss", "a suspect judgement")
suspect = block(out, "a suspect judgement", "the change that WON")
check("lab4 shows a genuine retrieval miss", bool(genuine))
check("lab4 shows a suspect judgement", bool(suspect))
sim = r"closest relevant document has similarity ([+-]?\d\.\d{3})"
g_sim = re.search(sim, genuine)
s_sim = re.search(sim, suspect)
check("lab4 genuine miss is closer to its judgements than the suspect one",
      bool(g_sim and s_sim) and float(g_sim.group(1)) > float(s_sim.group(1)),
      f"{g_sim and g_sim.group(1)} vs {s_sim and s_sim.group(1)}")
top = re.search(r"our top result has similarity ([+-]?\d\.\d{3})", genuine)
check("lab4 genuine miss: judged document beats what we returned",
      bool(g_sim and top) and float(g_sim.group(1)) > float(top.group(1)),
      f"judged {g_sim and g_sim.group(1)} vs returned {top and top.group(1)}")
check("lab4 does not blame retrieval for the suspect judgement",
      bool(suspect) and "No prompt engineering fixes this" not in suspect)
check("lab4 triage summary counts judged-relevant documents",
      re.search(r"never got a judged-relevant\s+document", out) is not None)

context = out[out.find("context sent to the model:"):]
starred = re.findall(r"^  \[(\d)\]\*", context, re.M)
check("lab4 fallback does not call the top passage best-supported",
      "Best-supported" not in context and "Top-ranked passage (no answer generated)" in context)
check("lab4 fallback does not claim sources were cited",
      "Cited sources" not in context and "Context sources:" in context)
if "1" in starred:
    check("lab4 adds no unjudged note when passage [1] is judged",
          "is unjudged" not in context)
else:
    note = re.search(r"passage \[1\] is unjudged.*?Judged relevant: (.*)$", context, re.S | re.M)
    check("lab4 says when the top passage is unjudged, and which ones are judged",
          note is not None and re.findall(r"\[(\d)\]", note.group(1)) == starred,
          f"stars {starred}, note {note and note.group(1)!r}")

measured = re.search(r"cross-encoder throughput: (\d+) pairs/sec", out)
claimed = re.search(r"~([\d,]+) pairs/sec",
                    (LABS / "wslib" / "__init__.py").read_text())
rates = (float(measured.group(1)), float(claimed.group(1).replace(",", ""))) \
    if measured and claimed else None
check("wslib's quoted cross-encoder rate is within 1.5x of lab4's",
      bool(rates) and max(rates) / min(rates) <= 1.5,
      f"measured {measured and measured.group(1)} vs quoted {claimed and claimed.group(1)}")

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1)
print("all lab output tests passed")
