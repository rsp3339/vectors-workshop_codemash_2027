"""
ABOUTME: Checks requirements.txt pins every package to an exact version, that the installed
ABOUTME: environment matches those pins, and that the docs promise only the tested Python.

Needs the labs environment installed from requirements.txt. Takes a second.
Run with:  python tests/test_requirements.py
"""

import importlib.metadata
import re
import sys
from pathlib import Path

LABS = Path(__file__).resolve().parent.parent

FAILED = []


def check(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label:<58} {detail}")
    if not ok:
        FAILED.append(label)


lines = [line.split("#")[0].strip()
         for line in (LABS / "requirements.txt").read_text().splitlines()]
lines = [line for line in lines if line]

for line in lines:
    pinned = re.fullmatch(r"([A-Za-z0-9_.-]+)==([0-9][0-9A-Za-z.]*)", line)
    check(f"requirements.txt pins {line.split('=')[0].split('>')[0]} exactly",
          pinned is not None, line)
    if pinned:
        name, version = pinned.groups()
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        check(f"installed {name} matches its pin", installed == version,
              f"pinned {version}, installed {installed}")

torch_pin = next((line for line in lines if line.startswith("torch==")), None)
for doc in ("../README.md", "README.md", "requirements.txt"):
    commands = re.findall(r"pip install (torch\S*) --index-url", (LABS / doc).read_text())
    check(f"{Path(doc).name} ({doc}) installs the pinned CPU torch",
          bool(commands) and all(c == torch_pin for c in commands),
          f"pinned {torch_pin}, commands {commands}")

for doc in ("../README.md", "README.md", "scripts/verify_setup.py"):
    check(f"{Path(doc).name} ({doc}) promises only Python 3.12",
          "3.11" not in (LABS / doc).read_text())

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1)
print("all requirements tests passed")
