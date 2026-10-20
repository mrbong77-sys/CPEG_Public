"""Energy E from the probe scores, the logistic head that maps them to E, and the per-request explanation record.

    z = [s0, s1 - s0, ..., s5 - s0]          s0 = encoder logit of the request, s_i = logit of probe i (probes.cf_variants)
    E = w0 * s0 + sum_i w_i * (s_i - s0) + b  (config/energy_head.json)
"""
from __future__ import annotations

import numpy as np

from . import encoder as ftenc
from .probes import CFX_PROBES, cf_variants
from ..route import band_tier

SEED = 42


def cfx_features(ckpt, texts: list[str], log=print, cache: bool = True) -> np.ndarray:
    """z = [s0, s1 - s0, ..., s5 - s0], s_i = the fine-tuned logit of probe i (cf_variants, 6 texts per request);
    cache: keep a score cache in the checkpoint folder (encoder.score)."""
    if not texts:
        return np.zeros((0, 1 + len(CFX_PROBES)))
    s = ftenc.score(ckpt, [v for t in texts for v in cf_variants(t, 6)], log, cache=cache).reshape(len(texts), 6)
    return np.concatenate([s[:, :1], s[:, 1:] - s[:, :1]], axis=1)


def cfx_energy(mod: dict, Z: np.ndarray) -> np.ndarray:
    """mod: {"w": [w0, w1..w5], "b": b}."""
    return Z @ np.asarray(mod["w"]) + float(mod["b"])


def cfx_head(Z: np.ndarray, y: np.ndarray, w: np.ndarray, folds: int = 5) -> tuple:
    """The head fit: logistic regression (lbfgs, C 1, balanced, sample weight = row weight) on the validation split;
    out-of-fold logits (seed 42) for the bands and the head fitted on every row. Returns (coef, intercept, oof)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    mk = lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)    # noqa: E731
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(Z, y):
        oof[te] = mk().fit(Z[tr], y[tr], sample_weight=w[tr]).decision_function(Z[te])
    m = mk().fit(Z, y, sample_weight=w)
    return m.coef_[0].astype(float), float(m.intercept_[0]), oof


def explain(mod: dict, Z: np.ndarray, tiers0: list[str]) -> list[dict]:
    """Per request: E, the text's own energy, each probe's energy shift and its additive contribution to E, the Gate1
    tier, the band zone, the decision and what decided it (no text in the record). mod: {"w", "b", "bands"}."""
    th = mod["bands"]
    e = cfx_energy(mod, Z)
    out = []
    for i, (t0, x) in enumerate(zip(tiers0, e)):
        x = float(x)
        zone = "high" if x >= th["theta_block"] else "low" if "theta_low" in th and x < th["theta_low"] else "mid"
        dec = band_tier(t0, x, th)
        by = {"high": "energy_block", "mid": "gate1"}.get(zone) or ("gate1_low_block" if dec == "block" else "energy_pass")
        r = {"E": round(x, 4), "zone": zone, "gate1": t0, "decision": dec, "decided_by": by}
        w = np.asarray(mod["w"])
        r |= {"E_text": round(float(Z[i, 0]), 4), "bias": round(float(mod["b"]), 4),
              "contrib_text": round(float(w[0] * Z[i, 0]), 4),
              "probe_shift": {p: round(float(Z[i, j + 1]), 4) for j, p in enumerate(CFX_PROBES)},
              "probe_contrib": {p: round(float(w[j + 1] * Z[i, j + 1]), 4) for j, p in enumerate(CFX_PROBES)}}
        out.append(r)
    return out
