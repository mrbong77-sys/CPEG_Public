"""Band routing: the energy E and the Gate1 tier -> block / delegate / pass, and the rule that set the thresholds.

Deployed bands of CPEG (config/operating_point.json):
    theta_block = 3.7741 (98th percentile of the generic benign validation energy)
    theta_low = theta_del = theta_release = 0.2209 (95th percentile of the same energies)

band_tier with these bands reduces to:
    E >= theta_block                   -> block   (whatever Gate1 says)
    theta_low <= E < theta_block       -> Gate1 block stays block; Gate1 pass or delegate -> delegate
    E < theta_low                      -> pass    (Gate1 has no say; a Gate1 block is not kept)

A delegation forwards the request to the backend LLM with DELIBERATION_SYSTEM_PROMPT. In every reported metric a
delegation counts as not blocked.
"""
from __future__ import annotations

import numpy as np

DELIBERATION_SYSTEM_PROMPT = (
    "Answer the request below carefully and responsibly, weighing any ethical dimensions. If it is a survey, rating, "
    "or multiple-choice item, give the answer in exactly the format the request asks for. Refuse only if answering "
    "would cause real, concrete harm — workable help for weapons, malware, intrusion, or fraud; verbatim copyrighted "
    "text; or private personal data.")


def band_tier(tier: str, e: float, th: dict) -> str:
    """Bands: E >= theta_block blocks; below theta_low the energy decides alone (pass) unless g1_low_block is set;
    theta_del <= E < theta_block turns a pass into a delegate; E < theta_release turns a Gate1 block into a delegate;
    otherwise the Gate1 tier stands."""
    if e >= th["theta_block"]:
        return "block"
    if "theta_low" in th and e < th["theta_low"]:                            # below the uncertain band the
        if th.get("g1_low_block") and tier == "block":                        # (option, off in CPEG) a Gate1 block still blocks
            return "block"
        return "pass"                                                         # energy is sure: Gate1 has no say
    if e >= th["theta_del"] and tier == "pass":
        return "delegate"
    if tier == "block" and "theta_release" in th and e < th["theta_release"]:
        return "delegate"
    return tier


def energy_only(e: float, th: dict) -> str:
    """Ablation "energy gate only" (no piece gate): block when E >= theta_block, otherwise pass."""
    return "block" if e >= th["theta_block"] else "pass"


# ------------------------------------------------------------------- operating-point rule (how the bands were set)

# The rule as configured for CPEG: theta_block at the 98th percentile of the generic benign validation energy; theta_del
# chosen among the 95th / 97th / 98th percentiles; Gate1 counts only inside [theta_low, theta_block) with theta_low at
# the 95th percentile, and inside the band a Gate1 block stands (theta_release := theta_low).
FINAL_RULE = {"q_block": (0.98,), "q_del": (0.95, 0.97, 0.98), "q_low": 0.95, "release_at_harm_min": True,
              "release_harm_q": 0.0, "prefer_release": True, "no_harm_release": True, "val_or_tiebreak": True,
              "val_or_release": True, "band_gate1_block": True}
GRID = {"q_block": (0.99, 0.995, 0.999), "q_del": (0.95, 0.97, 0.99), "q_release": (0.0, 0.01, 0.02, 0.05)}


def select(e_ben_val, e_harm_val, E_eval, tier0, lab, grid=GRID, prefer_release: bool = False,
           no_harm_release: bool = False, e_ben_or_val=None, val_or_release: bool = False,
           release_at_harm_min: bool = False, release_harm_q: float = 0.0):
    """Operating point on the routing texts the energy gate was trained on: fewest benign items blocked (then
    delegated) while no more harmful items pass than with Gate1 alone and at most 5 fewer harmful items are blocked."""
    base_pass = int(((np.array(tier0) == "pass") & (lab == 1)).sum())
    base_block = int(((np.array(tier0) == "block") & (lab == 1)).sum())
    rows = []
    for qb in grid["q_block"]:
        for qd in grid["q_del"]:
            for qr in (("harm_min",) if release_at_harm_min else grid["q_release"]):
                th = {"theta_del": float(np.quantile(e_ben_val, qd)), "theta_block": float(np.quantile(e_ben_val, qb))}
                th["theta_block"] = max(th["theta_block"], th["theta_del"])
                if qr == "harm_min":                                          # below every validation harmful E
                    th["theta_release"] = min(float(np.quantile(e_harm_val, release_harm_q)), th["theta_del"])  # q 0 = min
                elif qr > 0:
                    th["theta_release"] = min(float(np.quantile(e_harm_val, qr)), th["theta_del"])
                t = np.array([band_tier(a, float(x), th) for a, x in zip(tier0, E_eval)])
                t0 = np.array(tier0)
                m = {"q_block": qb, "q_del": qd, "q_release": qr, **{k: round(v, 4) for k, v in th.items()}, "th": th,
                     "benign_block": int(((t == "block") & (lab == 0)).sum()), "benign_delegate": int(((t == "delegate") & (lab == 0)).sum()),
                     "harmful_pass": int(((t == "pass") & (lab == 1)).sum()), "harmful_block": int(((t == "block") & (lab == 1)).sum()),
                     "harmful_released": int(((t0 == "block") & (t != "block") & (lab == 1)).sum())}
                m["ok"] = m["harmful_pass"] <= base_pass and m["harmful_block"] >= base_block - 5
                if no_harm_release:
                    m["ok"] = m["ok"] and m["harmful_released"] == 0
                if e_ben_or_val is not None and len(e_ben_or_val):
                    eo = np.asarray(e_ben_or_val)
                    m["val_or_energy_block"] = round(float((eo >= th["theta_block"]).mean()), 5)
                    m["val_or_released"] = round(float((eo < th["theta_release"]).mean()), 5) if "theta_release" in th else 0.0
                rows.append(m)
    ok = [r for r in rows if r["ok"]] or rows
    if e_ben_or_val is not None and val_or_release:
        best = min(ok, key=lambda r: (r["benign_block"], r.get("val_or_energy_block", 0), -r.get("val_or_released", 0),
                                      r["harmful_pass"]))
    elif e_ben_or_val is not None:
        best = min(ok, key=lambda r: (r["benign_block"], r.get("val_or_energy_block", 0),
                                      -(0 if r["q_release"] == "harm_min" else r["q_release"]), r["harmful_pass"]))
    elif prefer_release:
        best = min(ok, key=lambda r: (r["benign_block"], -(0 if r["q_release"] == "harm_min" else r["q_release"]),
                                      r["harmful_pass"], r["benign_delegate"]))
    else:
        best = min(ok, key=lambda r: (r["benign_block"], r["benign_delegate"], r["harmful_pass"]))
    return best, rows, {"gate1_harmful_pass": base_pass, "gate1_harmful_block": base_block}


def bands(v: dict, e_harm, e_sel, t_sel, l_sel, it: dict = FINAL_RULE) -> tuple:
    """The operating-point rule for one rule config: v = the validation energies {"benign": generic benign,
    "benign_or": over-refusal benign}; e_harm = energies of the benchmark harmful training rows; (e_sel, t_sel, l_sel) =
    energies, Gate1 tiers and labels of the routing texts the rule is chosen on. Returns (bands, chosen row, grid, base)."""
    g_ = {**GRID, **{k: it[k] for k in ("q_block", "q_del", "q_release") if k in it}}
    best, grid, base = select(v["benign"], e_harm, e_sel, t_sel, l_sel,
                              grid=g_, prefer_release=it.get("prefer_release", False),
                              no_harm_release=it.get("no_harm_release", False),
                              e_ben_or_val=v.get("benign_or") if it.get("val_or_tiebreak") else None,
                              val_or_release=it.get("val_or_release", False),
                              release_at_harm_min=it.get("release_at_harm_min", False),
                              release_harm_q=it.get("release_harm_q", 0.0))
    th = dict(best["th"])                                                     # unrounded thresholds
    th.update({k: best[k] for k in ("q_block", "q_del", "q_release")})
    if it.get("q_low"):                                                       # Gate1 only in [theta_low, theta_block)
        th["theta_low"] = min(float(np.quantile(v["benign"], it["q_low"])), th["theta_block"])
        th["q_low"] = it["q_low"]
        if it.get("band_gate1_block"):                                        # inside the band a Gate1 block
            th["theta_release"] = th["theta_low"]                             # stands (no release to delegation)
    return th, best, grid, base
