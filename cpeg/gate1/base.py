"""Gate1 base: normalisation and sentence constants, mechanical decodings, and the three-way threshold rule.

The parser never writes new text: every piece is a substring of the input or a mechanical decoding of one.
"""
from __future__ import annotations

import base64
import binascii
import codecs
import hashlib
import re

import numpy as np

MAX_PIECES, SENT_KEEP, MIN_WORDS = 24, 8, 4
WIN_OVER, WIN, STRIDE = 190, 150, 100
LO_Q, HI_Q = 0.01, 0.97                                                       # default threshold quantiles
ZW = re.compile("[\u200b-\u200f\u2060\ufeff]")
SENT = re.compile(r"(?<=[.!?])\s+")
QUOTED = re.compile(r'```(.+?)```|"([^"\n]{1,2000})"|\u201c([^\u201d\n]{1,2000})\u201d|`([^`\n]{1,2000})`', re.S)
B64 = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
HEX = re.compile(r"\b(?:[0-9a-fA-F]{2}){8,}\b")
SPACED = re.compile(r"\b(?:\w ){3,}\w\b")
LEET = str.maketrans({"4": "a", "@": "a", "3": "e", "1": "i", "!": "i", "0": "o", "5": "s", "$": "s", "7": "t", "+": "t"})
COMMON = set("""the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she
or an will my one all would there their what so up out if about who get which go me when make can like time no just him know
take people into year your good some could them see other than then now look only come its over think also back after use two
how our work first well way even new want because any these give day most us is are was were has had been how why where tell
explain write give list describe steps step help please""".split())


def _words(t: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", t.lower())


def _common_ratio(t: str) -> float:
    w = _words(t)
    return sum(x in COMMON for x in w) / len(w) if w else 0.0


def _printable(s: str) -> bool:
    return bool(s) and sum(ch.isprintable() or ch in "\n\t" for ch in s) / len(s) >= 0.9 and len(_words(s)) >= 3


def _decoded(t: str) -> list[str]:
    """Mechanical decodings of the input (no new text): base64 / hex tokens, rot13, reversed text, leetspeak, spaced letters."""
    out = []
    for m in B64.finditer(t):
        tok = m.group(0)
        try:
            s = base64.b64decode(tok + "=" * (-len(tok) % 4), validate=True).decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            continue
        if _printable(s):
            out.append(s)
    for m in HEX.finditer(t):
        try:
            s = bytes.fromhex(m.group(0)).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        if _printable(s):
            out.append(s)
    base = _common_ratio(t)
    for cand in (codecs.encode(t, "rot13"), t[::-1]):
        r = _common_ratio(cand)
        if r >= 0.3 and r - base >= 0.15:
            out.append(cand)
    toks = t.split()
    leet = [w for w in toks if re.search(r"[a-zA-Z]", w) and re.search(r"[0-9@$!+]", w)]
    if len(leet) >= 3 and len(leet) >= 0.2 * len(toks):
        out.append(" ".join(w.translate(LEET) if w in leet else w for w in toks))
    if SPACED.search(t):
        out.append(SPACED.sub(lambda m: m.group(0).replace(" ", ""), t))
    return out


def _sha(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def thresholds(p: np.ndarray, harmful: np.ndarray, hi_q: float = HI_Q, hi_mask: np.ndarray | None = None) -> dict:
    """lo: LO_Q of the harmful scores; hi: hi_q of the safe scores (only those under hi_mask when given)."""
    lo = float(np.quantile(p[harmful == 1], LO_Q))
    safe = (harmful == 0) if hi_mask is None else ((harmful == 0) & hi_mask)
    hi = float(np.quantile(p[safe], hi_q))
    if lo >= hi:
        lo = hi = (lo + hi) / 2
    return {"lo": round(lo, 6), "hi": round(hi, 6)}


def route(p: np.ndarray, th: dict) -> np.ndarray:
    return np.where(p >= th["hi"], "block", np.where(p < th["lo"], "pass", "delegate")).astype(object)
