"""The advice-label rule of the energy-gate training pool.

Requests for personal legal, financial (incl. accounting) or medical (incl. mental-health) advice are label 0: they are
to be answered by the backend model (a delegation or a pass is right, a block is an over-refusal). They are selected by
each dataset's own category, in every source group of the pool:

    MM-SafetyBench    scenarios 10-Legal_Opinion, 11-Financial_Advice, 12-Health_Consultation   (eval.units.MMSB_ADVICE)
    AIR-Bench 2024    cate-idx 2.6.22-2.6.25 (regulated-industry advice); 2.6.26 stays harmful  (eval.units.AIR_ADVICE)
    do-not-answer     legal / financial / medical / mental-health advice harms                    (dna)
    SORRY-Bench       medical / financial / legal advice categories                               (sorry)
    SALAD-Bench       O17 law advice, O20 medical advice                                           (salad)
    JailBreakV-28K    Health Consultation, Tailored Unlicensed Advice (incl. RedTeam-2K)           (jbv)

The same advice requests wrapped in a jailbreak template, an encoding or a persuasion rewrite (JailBreakV-28K
jailbreak_query; SORRY-Bench SORRY_WRAPPED mutations) leave the pool, so that the label of a jailbreak wrapper stays
harmful. Matching is by stripped text.

    rel, drop = delegate_texts(load_sources(), benchmark_advice_texts(eval_rows))
    pool, dropped = drop_texts(pool, drop)        # harmful rows whose text is a wrapped advice request leave the pool
    pool, relabelled = relabel_advice(pool, rel)  # harmful rows whose text is an advice request become 0

Pool rows are dicts {"text", "label", "source", ...}. Counts of the CPEG pool: config/energy_pool_composition.json.
The loaders read the public datasets with `datasets` / `huggingface_hub` (not needed for the pool functions).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..eval.units import is_advice

SORRY_REPO = "sorry-bench/sorry-bench-202406"
SORRY_ADVICE = ("medical advice", "financial advice", "legal consulting advice", "legal advice")
# SORRY-Bench mutations that wrap the request (encodings, persuasion rewrites): dropped; plain-language ones: relabelled
SORRY_WRAPPED = ("ascii", "atbash", "caesar", "morse", "authority_endorsement", "evidence-based_persuasion",
                 "expert_endorsement", "logical_appeal", "misrepresentation", "role_play")
DNA_ADVICE = ("Legal Advice", "Financial Advice", "Medical Advice", "Consult advice on psychotic disorders",
              "Advice of coping strategies")
SALAD_ADVICE = ("O17:", "O20:")
JBV = "JailbreakV-28K/JailBreakV-28k"
JBV_ADVICE = ("Health Consultation", "Tailored Unlicensed Advice")


def _hf(repo: str, **kw):
    from datasets import load_dataset
    return load_dataset(repo, **kw)


def dna() -> list[str]:
    """do-not-answer questions whose specific harm is legal / financial / medical / mental-health advice."""
    return [x["question"] for x in _hf("LibrAI/do-not-answer", split="train") if str(x["specific_harms"]).startswith(DNA_ADVICE)]


def _sorry_rows() -> list[tuple[str, str, str]]:
    """(mutation, category, first turn) of every question*.jsonl of SORRY-Bench ('' = the base file)."""
    from huggingface_hub import hf_hub_download, list_repo_files
    out = []
    for f in list_repo_files(SORRY_REPO, repo_type="dataset"):
        m = re.fullmatch(r"question(?:_(.+))?\.jsonl", f.rsplit("/", 1)[-1])
        if not m:
            continue
        for x in Path(hf_hub_download(SORRY_REPO, f, repo_type="dataset")).read_text(encoding="utf-8").split("\n"):
            if x.strip():
                r = json.loads(x)
                if r.get("turns"):
                    out.append((m.group(1) or "", str(r.get("category")), r["turns"][0]))
    return out


def _sorry_cats() -> set[str]:
    """SORRY-Bench category ids (1-based, in the order of meta_info.py) whose name names an advice category."""
    from huggingface_hub import hf_hub_download
    meta = Path(hf_hub_download(SORRY_REPO, "meta_info.py", repo_type="dataset")).read_text(encoding="utf-8")
    names = re.findall(r'"([^"\n]{3,80})"', meta)
    return {str(i + 1) for i, n in enumerate(names) if any(w in n.lower() for w in SORRY_ADVICE)}


def sorry(kind: str) -> list[str]:
    """SORRY-Bench advice-category questions over every mutation: 'plain' (relabelled) or 'wrapped' (dropped)."""
    cats = _sorry_cats()
    return [t for m, c, t in _sorry_rows() if c in cats and (m in SORRY_WRAPPED) == (kind == "wrapped")]


def salad() -> list[str]:
    """SALAD-Bench base set, 3-category O17 (law advice) and O20 (medical advice)."""
    return [x["question"] for x in _hf("OpenSafetyLab/Salad-Data", name="base_set", split="train")
            if str(x["3-category"]).startswith(SALAD_ADVICE)]


def jbv(kind: str) -> list[str]:
    """JailBreakV-28K advice policies: 'plain' = RedTeam-2K questions and the 28K set's redteam_query (relabelled),
    'wrapped' = the 28K set's jailbreak_query (dropped)."""
    if kind == "plain":
        out = [x["question"] for x in _hf(JBV, name="RedTeam_2K", split="RedTeam_2K") if x["policy"] in JBV_ADVICE]
        return out + [x["redteam_query"] for x in _hf(JBV, name="JailBreakV_28K", split="JailBreakV_28K")
                      if x["policy"] in JBV_ADVICE]
    return [x["jailbreak_query"] for x in _hf(JBV, name="JailBreakV_28K", split="JailBreakV_28K") if x["policy"] in JBV_ADVICE]


SOURCES = {"relabel": {"do-not-answer legal / financial / medical / mental-health advice": dna,
                       "sorry-bench medical / financial / legal advice, plain-language mutations": lambda: sorry("plain"),
                       "salad-bench O17 law / O20 medical advice": salad,
                       "jailbreakv health consultation / tailored unlicensed advice": lambda: jbv("plain")},
           "drop": {"jailbreakv advice in a jailbreak template": lambda: jbv("wrapped"),
                    "sorry-bench advice, encoded / persuasion mutations": lambda: sorry("wrapped")}}


def load_sources(loaders: dict = SOURCES) -> dict:
    """{"relabel": {name: [texts]}, "drop": {name: [texts]}}: the advice texts of the training sources, deduplicated."""
    return {kind: {name: sorted({t.strip() for t in fn() if isinstance(t, str) and t.strip()}) for name, fn in srcs.items()}
            for kind, srcs in loaders.items()}


def benchmark_advice_texts(rows: list[dict]) -> set[str]:
    """Stripped texts of the advice units among evaluation rows built by eval.units.build (MM-SafetyBench 10-12,
    AIR-Bench 2.6.22-2.6.25); their texts in the pool (benchmark halves, routing texts) take label 0 as well."""
    return {r["text"].strip() for r in rows if is_advice(r)}


def delegate_texts(sources: dict, bench_texts: set[str] = frozenset()) -> tuple[set[str], set[str]]:
    """(texts to relabel 0, texts to drop). A text in both sets is relabelled, not dropped."""
    rel = set(bench_texts) | {t.strip() for v in sources["relabel"].values() for t in v}
    drop = {t.strip() for v in sources["drop"].values() for t in v} - rel
    return rel, drop


def drop_texts(pool: list[dict], texts: set[str]) -> tuple[list[dict], dict]:
    """Harmful rows whose text is in `texts` leave the pool; the order of the rest is unchanged; counts per source."""
    kept, gone = [], {}
    for r in pool:
        if r["label"] == 1 and "label_orig" not in r and r["text"].strip() in texts:
            gone[r["source"]] = gone.get(r["source"], 0) + 1
        else:
            kept.append(r)
    return kept, dict(sorted(gone.items()))


def relabel_advice(pool: list[dict], texts: set[str]) -> tuple[list[dict], dict]:
    """Harmful rows whose text is in `texts` become 0 (label_orig 1 kept, so the validation split drawn on the original
    labels stays the same); order and every other row unchanged; counts per source."""
    out, n = [], {}
    for r in pool:
        if r["label"] == 1 and r["text"].strip() in texts:
            r = {**r, "label": 0, "label_orig": 1}
            n[r["source"]] = n.get(r["source"], 0) + 1
        out.append(r)
    return out, dict(sorted(n.items()))


def apply_rule(pool: list[dict], rel: set[str], drop: set[str]) -> tuple[list[dict], dict]:
    """The rule on a pool: drop the wrapped advice rows, then relabel the advice rows. Returns (pool, counts)."""
    pool, gone = drop_texts(pool, drop)
    pool, n = relabel_advice(pool, rel)
    return pool, {"dropped": gone, "relabelled": n, "n": len(pool),
                  "labels": {str(k): sum(r["label"] == k for r in pool) for k in (0, 1)}}
