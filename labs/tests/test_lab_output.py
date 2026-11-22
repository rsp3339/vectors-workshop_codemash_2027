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
