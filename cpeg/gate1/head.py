"""Gate1 end to end: parse -> score pieces (piece classifier, Platt-calibrated) and S pieces (substance classifier) ->
25 features -> gradient-boosted head -> p -> block / delegate / pass by two thresholds.

The head is refitted deterministically (seed 42) from the committed feature table of its fit pool
(config/gate1_head_fit.csv.gz: confident rows, y <= 0.2 or y >= 0.8, half A), exactly as the original system did at
load time. The thresholds are the registered ones (config/gate1.json: lo 0.05784, hi 0.736825), which the threshold
rule produced on half B (lo = 1st percentile of the harmful p, hi = 95th percentile of the safe p). `thresholds="refit"`
recomputes them with that rule instead; a different scikit-learn version can move the refitted trees slightly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import base as V1, encoders as ENC, frames as V5, pieces as V3, units as V2

CONFIG = Path(__file__).resolve().parents[2] / "config"
HI_Q = 0.95                                                                   # hi: 95th percentile of the safe half-B p
FEATS = V5.FEATS                                                              # the 25 features the head reads


def fit_head(fit: pd.DataFrame):
    """(head, thresholds by the rule) from the fit-pool feature table: fit on confident half A, thresholds on half B."""
    conf = fit[(fit.y <= V2.SAFE_Y) | (fit.y >= V2.HARM_Y)]
    A, B = conf[conf.half == "A"], conf[conf.half == "B"]
    m = V3._boost().fit(A[FEATS].to_numpy(), A.harmful.to_numpy())
    th = V1.thresholds(m.predict_proba(B[FEATS].to_numpy())[:, 1], B.harmful.to_numpy(), HI_Q, None)
    return m, th


def features(texts: list[str], score_piece, score_s) -> pd.DataFrame:
    """The 25 Gate1 features per text. score_piece / score_s: list[str] -> raw sigmoid scores (np.ndarray)."""
    us = [V5.units(t) for t in texts]
    pcs = sorted({p for u in us for p in u["pieces"]})
    sps = sorted({p for u in us for p in V5.s_sel(u)})
    cfg = json.loads((CONFIG / "gate1.json").read_text(encoding="utf-8"))
    raw = np.asarray(score_piece(pcs), float) if pcs else np.zeros(0)
    cal = dict(zip(pcs, ENC.platt(cfg["piece_calibration"], raw).tolist())) if pcs else {}
    s = dict(zip(sps, np.asarray(score_s(sps), float).tolist())) if sps else {}
    return pd.DataFrame([V5.features(u, cal, s) for u in us])


class Gate1:
    """Gate1 with its two classifiers loaded from `piece_ckpt` and `s_ckpt` (torch files, see encoders.py)."""

    def __init__(self, piece_ckpt=None, s_ckpt=None, thresholds: str = "registered", config_dir: Path = CONFIG):
        cfg = json.loads((config_dir / "gate1.json").read_text(encoding="utf-8"))
        fit = pd.read_csv(config_dir / cfg["head_fit_features"], encoding="utf-8")
        self.head, th_rule = fit_head(fit)
        self.th = th_rule if thresholds == "refit" else {"lo": cfg["thresholds"]["lo"], "hi": cfg["thresholds"]["hi"]}
        self.piece_ckpt, self.s_ckpt = piece_ckpt, s_ckpt

    def p(self, feats: pd.DataFrame) -> np.ndarray:
        return self.head.predict_proba(feats[FEATS].to_numpy())[:, 1]

    def tiers(self, texts: list[str], log=print) -> tuple[np.ndarray, np.ndarray]:
        """(p, tier) per text; tier in {block, delegate, pass}."""
        f = features(texts, lambda x: ENC.predict(x, ENC.PIECE_ENC, self.piece_ckpt, log),
                     lambda x: ENC.predict(x, ENC.SUBSTANCE_ENC, self.s_ckpt, log))
        p = self.p(f)
        return p, V1.route(p, self.th)
