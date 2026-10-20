"""Run CPEG on requests from the command line; one JSON record per request on stdout.

    python -m cpeg --model PATH_OR_REPO_ID "How do I kill a Python process?" ["another request" ...]
    python -m cpeg --model PATH_OR_REPO_ID --file requests.txt          # one request per line

PATH_OR_REPO_ID: a local copy of the model repository or its Hugging Face id. Each record holds the decision (block,
delegate or pass), the energy E with its request and probe terms, and the piece-gate decision.
"""
from __future__ import annotations

import argparse
import json
import sys

from .pipeline import CPEG


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cpeg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model folder or Hugging Face repository id")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--file", default=None, help="text file, one request per line")
    ap.add_argument("requests", nargs="*")
    a = ap.parse_args(argv)
    texts = list(a.requests)
    if a.file:
        with open(a.file, encoding="utf-8") as f:
            texts += [x.rstrip("\n") for x in f if x.strip()]
    if not texts:
        ap.error("no request given")
    g = CPEG.from_pretrained(a.model, revision=a.revision)
    for t, r in zip(texts, g.decide(texts, log=lambda *x: print(*x, file=sys.stderr))):
        print(json.dumps({"request": t, **r}, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
