"""Gate1 units: a deterministic parser splits the prompt, rules tag each unit as frame, request or material, and the
core request is reassembled from the input's own units.

No new text is written: every unit is a substring of the input or a mechanical decoding of one, and the core is those
units joined in their original order.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np

from . import base as V1

MAX_REQ, MAT_HEAD, MAT_TAIL, MAX_PIECES = 8, 3, 3, 16
SAFE_Y, HARM_Y = 0.2, 0.8                                                     # confident labels for the head fit
FRAME = re.compile(
    r"\b(you are|you're|act as|acting as|pretend|role[- ]?play|imagine|let's play|stay in character|from now on|"
    r"(ignore|disregard)\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|all)\b|developer mode|\bDAN\b|jailbreak|"
    r"hypothetical(ly)?|in a fictional|fictional (world|story|universe)|for (educational|research) purposes|"
    r"no (restrictions|filters|limits|rules)|without (any )?(restrictions|censorship|filters|limits)|"
    r"you (must|will|should) (always|never)|respond only as|your (new )?(role|task|name) is)", re.I)
REQ_START = re.compile(
    r"^\W*(write|tell|give|explain|describe|list|provide|create|generate|make|show|help|teach|how|what|why|where|which|"
    r"who|when|can you|could you|would you|will you|please|i want|i need|i'd like|i would like)\b", re.I)
REQ_ANY = re.compile(r"\?\s*$|\bstep[- ]by[- ]step\b|\binstructions?\b|\bsteps\b", re.I)
FEATS = ["core", "req_max", "req_mean", "mat_max", "n_req", "n_mat", "n_frame", "has_frame", "mixed", "has_decoded",
         "has_joined", "core_is_whole"]


def role(unit: str) -> tuple[str, bool]:
    """(role, mixed) for one sentence unit."""
    fr = bool(FRAME.search(unit))
    rq = bool(REQ_START.search(unit) or REQ_ANY.search(unit))
    if rq:
        return "request", fr
    return ("frame" if fr else "material"), False


def split(text: str) -> tuple[str, list[str], list[tuple[str, str, bool]], list[str], list[tuple[str, str, bool]]]:
    """Normalised text, sentences, (role, unit, mixed) in input order, quoted fragments, decoded/joined payloads."""
    t = V1.ZW.sub("", unicodedata.normalize("NFKC", str(text))).strip()
    sents = []
    for line in t.splitlines():
        for s in V1.SENT.split(line.strip()):
            s = s.strip()
            if not s:
                continue
            if sents and len(s.split()) < V1.MIN_WORDS:
                sents[-1] = sents[-1] + " " + s
            else:
                sents.append(s)
    seq = []                                                                   # (role, text, mixed) in input order
    for s in sents:
        w = s.split()
        parts = [s] if len(w) <= V1.WIN_OVER else [" ".join(w[i:i + V1.WIN]) for i in range(0, len(w) - V1.WIN + V1.STRIDE, V1.STRIDE)]
        r, mixed = role(s)
        seq += [(r if len(parts) == 1 else ("material" if r != "frame" else "frame"), p, mixed) for p in parts]
    frags = [next(g for g in m.groups() if g is not None).strip() for m in V1.QUOTED.finditer(t)]
    frags = [f for f in frags if f]
    payload = [("request", d, False) for d in V1._decoded(t)]
    if len(frags) >= 2 and sum(len(f.split()) for f in frags) >= V1.MIN_WORDS:
        payload.append(("request", " ".join(frags), False))
    return t, sents, seq, frags, payload


def units(text: str) -> dict:
    """Tagged units, the core, and the pieces to score (never the whole prompt unless it is one unit)."""
    t, sents, seq, frags, payload = split(text)
    quoted = [("material", f, False) for f in frags if len(f.split()) >= V1.MIN_WORDS and f not in sents]
    req = [u for r, u, _ in seq if r == "request"] + [u for _, u, _ in payload]
    mat = [u for r, u, _ in seq if r == "material"] + [u for _, u, _ in quoted]
    frames = [u for r, u, _ in seq if r == "frame"]
    core_parts = req or mat or ([seq[0][1]] if seq else [t])
    core = " ".join(dict.fromkeys(core_parts))
    req_scored = list(dict.fromkeys(req))[:MAX_REQ]
    mat_u = list(dict.fromkeys(mat))
    mat_scored = mat_u if len(mat_u) <= MAT_HEAD + MAT_TAIL else mat_u[:MAT_HEAD] + mat_u[-MAT_TAIL:]
    pieces = list(dict.fromkeys([core] + req_scored + mat_scored))[:MAX_PIECES]
    return {"core": core, "req": req_scored, "mat": mat_scored, "n_req": len(req), "n_mat": len(mat), "n_frame": len(frames),
            "mixed": any(m for _, _, m in seq), "has_decoded": any(V1._decoded(t)), "has_joined": len(frags) >= 2,
            "core_is_whole": core == t, "pieces": pieces}


def features(u: dict, cal: dict[str, float]) -> dict:
    rq = [cal[p] for p in u["req"]]
    mt = [cal[p] for p in u["mat"]]
    return {"core": cal[u["core"]], "req_max": max(rq) if rq else 0.0, "req_mean": float(np.mean(rq)) if rq else 0.0,
            "mat_max": max(mt) if mt else 0.0, "n_req": float(np.log1p(u["n_req"])), "n_mat": float(np.log1p(u["n_mat"])),
            "n_frame": float(np.log1p(u["n_frame"])), "has_frame": float(u["n_frame"] > 0), "mixed": float(u["mixed"]),
            "has_decoded": float(u["has_decoded"]), "has_joined": float(u["has_joined"]), "core_is_whole": float(u["core_is_whole"])}
