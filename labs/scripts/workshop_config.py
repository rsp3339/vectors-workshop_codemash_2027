"""
The one place the published locations live.

Set REPO once here (or export WORKSHOP_REPO) and every notebook's Colab bootstrap
plus every generated Colab link picks it up. Nothing else in the labs needs to
know where they are published.

    python scripts/build_notebooks.py            # uses the values below
    WORKSHOP_REPO=github.com/me/vecws python scripts/build_notebooks.py
    python scripts/build_notebooks.py --repo https://github.com/me/vecws
"""

from __future__ import annotations

import os

# ── set these ──────────────────────────────────────────────────────────────

# GitHub repository holding this labs/ directory. Owner/name is enough; a full
# https:// URL works too. Leave as the placeholder and the bootstrap cell will
# tell attendees it isn't configured yet rather than failing obscurely.
REPO = os.environ.get("WORKSHOP_REPO", "rsp3339/vectors-workshop_codemash_2027")

BRANCH = os.environ.get("WORKSHOP_BRANCH", "main")

# Directory inside the repo that contains wslib/ — "labs" if you commit this
# folder as-is, "" if you publish its contents at the repo root.
SUBDIR = os.environ.get("WORKSHOP_SUBDIR", "labs")

# Optional: a prebuilt data cache (corpora + embeddings + rerank scores) so Colab
# users skip the ~8 minute prepare_data step. Upload the archive produced by
# scripts/make_data_archive.py as a GitHub Release asset and paste its URL here.
# Leave blank and the bootstrap builds the cache instead.
DATA_URL = os.environ.get(
    "WORKSHOP_DATA_URL",
    "https://github.com/rsp3339/vectors-workshop_codemash_2027/releases/download/"
    "workshop-data/workshop-data.tar.gz",
)


# ── derived ────────────────────────────────────────────────────────────────


def slug(repo: str = None) -> str:
    """owner/name, whatever form REPO was given in."""
    r = (repo or REPO).strip().rstrip("/")
    for prefix in ("https://github.com/", "http://github.com/", "github.com/", "git@github.com:"):
        if r.startswith(prefix):
            r = r[len(prefix):]
    return r.removesuffix(".git")


def clone_url(repo: str = None) -> str:
    return f"https://github.com/{slug(repo)}.git"


def is_configured(repo: str = None) -> bool:
    return "REPLACE_ME" not in slug(repo)


def colab_url(notebook_path: str, repo: str = None, branch: str = None) -> str:
    """Colab link for a notebook path relative to the repo root.

    Colab's /github/ route opens a notebook straight from a public repo, so
    attendees need no Google Drive setup and no account beyond being signed in.
    """
    path = notebook_path.lstrip("/")
    return (
        f"https://colab.research.google.com/github/{slug(repo)}/blob/"
        f"{branch or BRANCH}/{path}"
    )
