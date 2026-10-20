"""Gate1 pieces: the units plus clause splits of long requests, request-with-preceding-context pairs and de-framed
requests, and the piece-score features built on them.

Every piece is a substring of the input, a mechanical decoding, a deletion of matched frame spans, or input substrings
joined in their original order. The whole prompt is scored only when it is a single unit.
"""
from __future__ import annotations

import re

import numpy as np

from . import base as V1, units as V2

MAX_REQ, MAX_PAIR, MAX_DEFRAME, MAX_CLAUSE, MAT_HEAD, MAT_TAIL, MAX_PIECES = 5, 3, 2, 4, 2, 2, 16
CLAUSE_MIN = 12
TAU = 0.26094                                                                 # piece-classifier operating threshold ("hot")
CLAUSE = re.compile(r"(?<=[,;:])\s+|\s+(?=(?:and then|(?<!and )then|so that|in order to|but|while|because|which|after|before)\b)",
                    re.I)
FEATS = V2.FEATS + ["pair_max", "deframed_max", "clause_max", "all_max", "top2", "hot"]


def clauses(unit: str) -> list[str]:
    """Split a long request at , ; : and connectives; fragments under 4 words join the previous one."""
    if len(unit.split()) < CLAUSE_MIN:
        return []
    out: list[str] = []
    for c in (x.strip() for x in CLAUSE.split(unit)):
        if not c:
            continue
        if out and len(c.split()) < V1.MIN_WORDS:
            out[-1] = out[-1] + " " + c
        else:
            out.append(c)
    if len(out) >= 2 and len(out[0].split()) < V1.MIN_WORDS:
        out[:2] = [out[0] + " " + out[1]]
    return out if len(out) >= 2 else []


def deframe(unit: str) -> str | None:
    """A mixed request with its matched frame spans deleted, if 4+ words remain."""
    r = re.sub(r"\s+", " ", V2.FRAME.sub(" ", unit)).strip(" ,.;:")
    return r if len(r.split()) >= V1.MIN_WORDS and r != unit else None


def units(text: str) -> dict:
    t, sents, seq, frags, payload = V2.split(text)
    base = V2.units(text)
    req_all = [u for r, u, _ in seq if r == "request"] + [u for _, u, _ in payload]
    req = list(dict.fromkeys(req_all))[:MAX_REQ]
    pairs = []
    for i, (r, u, _) in enumerate(seq):
        if r != "request":
            continue
        prev = next((seq[j][1] for j in range(i - 1, -1, -1) if seq[j][0] != "frame"), None)
        if prev is not None and len(seq) > 2:                                   # a pair never spans every unit of the prompt
            pairs.append(prev + " " + u)
    pairs = list(dict.fromkeys(pairs))[:MAX_PAIR]
    defr = list(dict.fromkeys(d for r, u, m in seq if r == "request" and m for d in [deframe(u)] if d))[:MAX_DEFRAME]
    cl = list(dict.fromkeys(c for u in req_all for c in clauses(u)))[:MAX_CLAUSE]
    mat_u = list(dict.fromkeys([u for r, u, _ in seq if r == "material"] +
                               [f for f in frags if len(f.split()) >= V1.MIN_WORDS and f not in sents]))
    mat = mat_u if len(mat_u) <= MAT_HEAD + MAT_TAIL else mat_u[:MAT_HEAD] + mat_u[-MAT_TAIL:]
    pieces = list(dict.fromkeys([base["core"]] + req + pairs + defr + cl + mat))[:MAX_PIECES]
    keep = set(pieces)
    return {**base, "req": [p for p in req if p in keep], "mat": [p for p in mat if p in keep],
            "pairs": [p for p in pairs if p in keep], "deframed": [p for p in defr if p in keep],
            "clauses": [p for p in cl if p in keep], "pieces": pieces}


def features(u: dict, cal: dict[str, float]) -> dict:
    f = V2.features(u, cal)
    mx = lambda k: max((cal[p] for p in u[k]), default=0.0)                    # noqa: E731
    s = sorted((cal[p] for p in u["pieces"]), reverse=True)
    f.update({"pair_max": mx("pairs"), "deframed_max": mx("deframed"), "clause_max": mx("clauses"), "all_max": s[0],
              "top2": float(np.mean(s[:2])), "hot": float(np.mean([x >= TAU for x in s]))})
    return f


def _boost():
    """The Gate1 head: gradient-boosted trees, fixed seed."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=15, random_state=42)
