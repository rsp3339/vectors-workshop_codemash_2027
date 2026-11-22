#!/usr/bin/env python3
"""
Turn `# %%`-delimited lab scripts into .ipynb files for Colab.

The labs are authored as plain .py so they diff cleanly, run in CI, and can be
executed top-to-bottom to check they still work. Attendees get notebooks with a
Colab bootstrap cell prepended.

    python scripts/build_notebooks.py                  # all labs
    python scripts/build_notebooks.py lab2             # just one
    python scripts/build_notebooks.py --repo github.com/me/vecws
    python scripts/build_notebooks.py --links          # print Colab URLs only

Publication targets live in scripts/workshop_config.py — set them once there.

Cell markers:
    # %%              -> code cell
    # %% [markdown]   -> markdown cell (leading "# " stripped from each line)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import workshop_config as cfg

ROOT = Path(__file__).resolve().parent.parent


# ── py -> cells ────────────────────────────────────────────────────────────


def split_cells(source: str) -> List[Dict]:
    cells: List[Dict] = []
    kind = "code"
    buf: List[str] = []

    def flush():
        if not buf:
            return
        text = "\n".join(buf).strip("\n")
        if not text.strip():
            return
        if kind == "markdown":
            lines = [
                ln[2:] if ln.startswith("# ") else ("" if ln.strip() == "#" else ln)
                for ln in text.split("\n")
            ]
            text = "\n".join(lines).strip("\n")
        cells.append({"kind": kind, "text": text})

    for line in source.split("\n"):
        if line.strip().startswith("# %%"):
            flush()
            buf = []
            kind = "markdown" if "[markdown]" in line else "code"
            continue
        buf.append(line)
    flush()
    return cells


def to_notebook(cells: List[Dict]) -> Dict:
    def mk(cell):
        lines = cell["text"].split("\n")
        src = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        if cell["kind"] == "markdown":
            return {"cell_type": "markdown", "metadata": {}, "source": src}
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src,
        }

    return {
        "cells": [mk(c) for c in cells],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ── the Colab bootstrap ────────────────────────────────────────────────────


def bootstrap(repo: str, branch: str, subdir: str, data_url: str) -> str:
    """First two cells of every notebook: get the code, get the data, verify.

    Written so that running it locally in Jupyter is a no-op — it detects that
    wslib is already importable and does nothing.
    """
    configured = cfg.is_configured(repo)
    slug = cfg.slug(repo)
    url = cfg.clone_url(repo)

    unconfigured_note = "" if configured else (
        "\n# ⚠️  This notebook was generated before the repo URL was set.\n"
        "#     Instructor: set REPO in scripts/workshop_config.py and rebuild.\n"
    )

    return f'''# %% [markdown]
# ### Setup — run this cell first
#
# On **Colab** this fetches the code and the precomputed data cache, then checks
# your environment. On a **local** Jupyter or VS Code install it detects that
# everything is already present and does nothing.
#
# It takes about a minute the first time and nothing thereafter.

# %%{unconfigured_note}
REPO_SLUG = "{slug}"
BRANCH    = "{branch}"
SUBDIR    = "{subdir}"
DATA_URL  = "{data_url}"

import os, subprocess, sys

# Note: importlib.util.find_spec("google.colab") RAISES ModuleNotFoundError when
# there is no `google` package at all, which is every local machine. Ask by
# importing instead.
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except Exception:
    IN_COLAB = False


def sh(*args):
    """Run a command, surfacing its output only if it fails."""
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-2000:], sep="\\n")
        raise RuntimeError(" ".join(args) + f" -> exit {{p.returncode}}")
    return p.stdout


def find_labs_root():
    """Directory containing wslib/, searching cwd and its ancestors."""
    here = os.path.abspath(os.getcwd())
    for _ in range(4):
        if os.path.isdir(os.path.join(here, "wslib")):
            return here
        here = os.path.dirname(here)
    return None


root = find_labs_root()

if root is None and not IN_COLAB:
    raise SystemExit(
        "Could not find wslib/ from " + os.getcwd() + ".\\n"
        "Open this notebook from inside the labs/ directory, or run it in Colab."
    )

if root is None:
    if "REPLACE_ME" in REPO_SLUG:
        raise SystemExit(
            "This notebook has no repository configured yet.\\n"
            "Instructor: set REPO in scripts/workshop_config.py and re-run "
            "scripts/build_notebooks.py."
        )

    print(f"cloning {{REPO_SLUG}} ...")
    sh("git", "clone", "--depth", "1", "--branch", BRANCH,
       f"https://github.com/{{REPO_SLUG}}.git", "/content/workshop")
    root = os.path.join("/content/workshop", SUBDIR) if SUBDIR else "/content/workshop"

    print("installing dependencies (~40s) ...")
    sh(sys.executable, "-m", "pip", "install", "-q",
       "faiss-cpu", "sentence-transformers", "rank-bm25")

os.chdir(root)
if root not in sys.path:
    sys.path.insert(0, root)
print(f"labs root: {{root}}")

# %% [markdown]
# ### Data cache
#
# The corpora and their embeddings are precomputed so no lab opens with a
# progress bar. This pulls the prebuilt archive if one is published, and builds
# it locally otherwise.

# %%
import glob, os, sys, subprocess

need = not glob.glob(os.path.join("data", "emb_*.npy"))

if not need:
    print("data cache present")
elif DATA_URL:
    print("downloading prebuilt data cache ...")
    os.makedirs("data", exist_ok=True)
    subprocess.run(["curl", "-fsSL", "-o", "/tmp/wsdata.tar.gz", DATA_URL], check=True)
    subprocess.run(["tar", "-xzf", "/tmp/wsdata.tar.gz"], check=True)
    print("  done")
else:
    print("no prebuilt cache configured — building it now (this is the slow path)")
    subprocess.run([sys.executable, "scripts/prepare_data.py", "--poolings"], check=True)

# OMP_NUM_THREADS must be set before faiss or torch loads; importing wslib does
# that for us, so import it before anything else touches those libraries.
import wslib  # noqa: F401
subprocess.run([sys.executable, "scripts/verify_setup.py"])
'''


# ── build ──────────────────────────────────────────────────────────────────


def build(py_path: Path, out_path: Path, boot: str) -> Dict:
    src = py_path.read_text(encoding="utf-8")
    cells = split_cells(boot + "\n" + src)
    out_path.write_text(json.dumps(to_notebook(cells), indent=1), encoding="utf-8")
    n_md = sum(1 for c in cells if c["kind"] == "markdown")
    return {"code": len(cells) - n_md, "markdown": n_md}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("labs", nargs="*", help="filter, e.g. lab2")
    ap.add_argument("--repo", default=cfg.REPO)
    ap.add_argument("--branch", default=cfg.BRANCH)
    ap.add_argument("--subdir", default=cfg.SUBDIR)
    ap.add_argument("--data-url", default=cfg.DATA_URL)
    ap.add_argument("--links", action="store_true", help="print Colab URLs only")
    args = ap.parse_args()

    found = sorted(ROOT.glob("lab*/lab*.py"))
    if not found:
        print("no lab*/lab*.py files found", file=sys.stderr)
        return 1
    wanted = [p for p in found
              if not args.labs
              or any(w in p.stem or w in p.parent.name for w in args.labs)]
    if not wanted:
        print(f"nothing matched {args.labs}", file=sys.stderr)
        return 1

    configured = cfg.is_configured(args.repo)
    boot = bootstrap(args.repo, args.branch, args.subdir, args.data_url)

    if not args.links:
        for py in wanted:
            stats = build(py, py.with_suffix(".ipynb"), boot)
            print(f"{py.name} -> {py.with_suffix('.ipynb').name}  "
                  f"({stats['code']} code, {stats['markdown']} markdown)")

    print()
    if not configured:
        print("REPO is not set yet — notebooks will refuse to bootstrap in Colab.")
        print("Set it in scripts/workshop_config.py (or pass --repo) and rebuild.\n")

    print(f"repo   : {cfg.slug(args.repo)}  (branch {args.branch})")
    print(f"data   : {args.data_url or 'not published — Colab will build the cache'}")
    print("\nColab links for the setup slide and the pre-work email:")
    for py in wanted:
        rel = py.with_suffix(".ipynb").relative_to(ROOT)
        path = f"{args.subdir}/{rel}" if args.subdir else str(rel)
        print(f"  {py.parent.name:<18} {cfg.colab_url(path, args.repo, args.branch)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
