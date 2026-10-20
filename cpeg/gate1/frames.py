"""Gate1 final parser and feature set: frame sentences are scored (frame spans deleted) and, when no request is found,
a core of the non-frame units and de-framed frames in input order. The pieces are scored by the piece classifier
(calibrated) and by the independent substance classifier S (DeBERTa-v3-base); 25 features in all (FEATS).
"""
from __future__ import annotations

import numpy as np

from . import pieces as V3, units as V2

S_ENC = "microsoft/deberta-v3-base"                                           # substance classifier S
S_LEN, S_HOT = 256, 0.5                                                       # S max tokens; S 'hot' piece cut-off
S_FEATS = ["s_core", "s_max", "s_top2", "s_mean", "s_hot"]
MAX_FRAME, S_REQ, S_FRAME, S_PAIR = 3, 3, 2, 2
FEATS = V3.FEATS + ["frame_max"] + S_FEATS + ["s_frame_max"]


def units(text: str) -> dict:
    t, sents, seq, frags, payload = V2.split(text)
    u = V3.units(text)
    frames = list(dict.fromkeys(d for r, x, _ in seq if r == "frame" for d in [V3.deframe(x)] if d))[:MAX_FRAME]
    core = u["core"]
    if u["n_req"] == 0:                                                        # no request -> frames join the core
        parts = [x if r != "frame" else V3.deframe(x) for r, x, _ in seq]
        parts = [x for x in parts if x] + [m for m in u["mat"] if m not in sents and m not in parts]
        if parts:
            core = " ".join(dict.fromkeys(parts))
    pieces = list(dict.fromkeys([core] + u["req"] + frames + u["pairs"] + u["deframed"] + u["clauses"] + u["mat"]))[:V3.MAX_PIECES]
    keep = set(pieces)
    return {**u, "core": core, "core_is_whole": core == t, "frames": [f for f in frames if f in keep], "pieces": pieces,
            "req": [p for p in u["req"] if p in keep], "mat": [p for p in u["mat"] if p in keep],
            "pairs": [p for p in u["pairs"] if p in keep], "deframed": [p for p in u["deframed"] if p in keep],
            "clauses": [p for p in u["clauses"] if p in keep]}


def s_sel(u: dict) -> list[str]:
    """The pieces S scores: the core, up to three requests, two frames and two pairs."""
    return list(dict.fromkeys([u["core"]] + u["req"][:S_REQ] + u["frames"][:S_FRAME] + u["pairs"][:S_PAIR]))


def features(u: dict, cal: dict[str, float], s: dict[str, float]) -> dict:
    v = [s[p] for p in s_sel(u)]
    top = sorted(v, reverse=True)
    return {**V3.features(u, cal), "frame_max": max((cal[p] for p in u["frames"]), default=0.0),
            "s_core": s[u["core"]], "s_max": top[0], "s_top2": float(np.mean(top[:2])), "s_mean": float(np.mean(v)),
            "s_hot": float(np.mean([x >= S_HOT for x in v])),
            "s_frame_max": max((s[p] for p in u["frames"][:S_FRAME]), default=0.0)}
