"""
ABOUTME: Checks each notebook's Colab bootstrap: a missing data archive fails at the
ABOUTME: download rather than in tar, and setup installs the pinned packages, openai included.

Needs no data cache and no network. Takes a few seconds.
Run with:  python tests/test_colab_bootstrap.py
"""

import http.server
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

LABS = Path(__file__).resolve().parent.parent

FAILED = []


def check(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label:<58} {detail}")
    if not ok:
        FAILED.append(label)


def data_cell(notebook):
    """Source of the code cell that fetches the data cache."""
    cells = json.loads(notebook.read_text())["cells"]
    for c in cells:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and "DATA_URL:" in src:
            return src
    raise SystemExit(f"no data cell in {notebook}")


class NotFound(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"<html>Not Found</html>"
        self.send_response(404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


server = http.server.HTTPServer(("127.0.0.1", 0), NotFound)
threading.Thread(target=server.serve_forever, daemon=True).start()
missing_url = f"http://127.0.0.1:{server.server_port}/workshop-data.tar.gz"

home = os.getcwd()
for nb in sorted(LABS.glob("lab*/lab*.ipynb")):
    failed_step = None
    with tempfile.TemporaryDirectory() as empty, tempfile.TemporaryFile() as err:
        # The cell's subprocesses write to fd 1 and 2 directly, so capture at
        # the descriptor level rather than through sys.stdout / sys.stderr.
        sys.stdout.flush()
        saved = os.dup(1), os.dup(2)
        os.dup2(err.fileno(), 1)
        os.dup2(err.fileno(), 2)
        os.chdir(empty)
        try:
            exec(data_cell(nb), {"DATA_URL": missing_url, "__name__": "__main__"})
        except subprocess.CalledProcessError as e:
            failed_step = e.cmd[0]
        finally:
            sys.stdout.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.chdir(home)
        err.seek(0)
        output = err.read().decode(errors="replace")
    check(f"{nb.name}: a missing archive fails at the download",
          failed_step == "curl", f"failed at {failed_step}")
    named = [line for line in output.splitlines() if "404" in line]
    check(f"{nb.name}: the error names the HTTP 404",
          bool(named), named[0] if named else output.strip()[-80:])

server.shutdown()

pins = {line.split("#")[0].strip()
        for line in (LABS / "requirements.txt").read_text().splitlines()} - {""}

for nb in sorted(LABS.glob("lab*/lab*.ipynb")):
    cells = json.loads(nb.read_text())["cells"]
    setup = next("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    install = re.search(r'"pip", "install", "-q",(.*?)\)', setup, re.S)
    packages = re.findall(r'"([^"]+)"', install.group(1)) if install else []
    check(f"{nb.name}: Colab setup installs openai for Lab 4's generation step",
          any(p.split("==")[0] == "openai" for p in packages), ", ".join(packages))
    check(f"{nb.name}: Colab setup installs the versions requirements.txt pins",
          bool(packages) and all(p in pins for p in packages),
          ", ".join(p for p in packages if p not in pins))

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1)
print("all Colab bootstrap tests passed")
