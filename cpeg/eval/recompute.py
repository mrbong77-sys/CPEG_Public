"""Recompute the paper's Tables 1-4, the Figure 3 data and the figures quoted in the text from unit counts (no model,
no GPU, no benchmark text).

    python -m cpeg.eval.recompute            # prints each check and exits 1 if any check fails

Inputs (results/): table1..table4 CSV (the paper's Tables 1-4 as printed), fig3_data.json (Figure 3),
unit_counts.json (per system and benchmark, on all units and on the units untrained for CPEG: harmful units not
blocked and benign units blocked, with their unit counts), paired_tests.json (exact McNemar comparisons of the
before-the-patch and no-probes ablations with CPEG) and config/energy_pool_composition.json.

Checks
1. Table 1: harmful, benign, untrained harmful and untrained benign units of each benchmark against the unit counts.
2. Table 2: every row (CPEG and the five ablations: mean APR over 7 harmful and mean ORR over 9 benign benchmarks,
   WildJailbreak and WildGuardTest APR / ORR, two-turn APR) from the counts on all units.
3. Table 3: every row (six benchmarks on the units untrained for CPEG, their mean, XSTest on all units), the row
   "CPEG, all units" and the no-benchmark-items row (§).
4. Table 4: every row (six benign benchmarks on the untrained units and their mean, the advice requests on the
   untrained units (‡), XSTest and OR-Bench-Hard on all units (†)), the row "CPEG, all units" and the § row.
5. Mean columns of Tables 3 and 4 within 0.1 point of the mean of the row's rounded cells.
6. Figure 3: the unit counts of CPEG and before the patch, and the means of every system and ablation.
7. Paired tests: exact two-sided McNemar p and Holm-adjusted p per benchmark, pooled counts and p, and the
   discordant counts against the unit counts (blocked by the ablation − blocked by CPEG = a_only − b_only).
8. Figures quoted in the text (Sections 5.1-5.5 and 6), each from the unit counts, the paired tests or the pool
   composition.
Table 5 (reasoning items blocked, time and memory) needs the per-item records and the latency measurements; it is
recomputed by the release check of the records repository.
"""
from __future__ import annotations

import csv
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from .metrics import holm, mcnemar

ROOT = Path(__file__).resolve().parents[2]
RES, CFG = ROOT / "results", ROOT / "config"
HARM = ["harmbench", "xstest", "mmsafetybench", "airbench", "wildguardtest", "wildjailbreak", "taxonomy"]
BEN = ["xstest", "advice", "wildguardtest", "wildjailbreak", "orbench", "ethics_cs", "llm_ethics", "ggb", "morables"]
SIX_H = [b for b in HARM if b != "xstest"]                       # Table 3: XSTest has no untrained harmful unit
SIX_B = ["wildguardtest", "wildjailbreak", "ethics_cs", "llm_ethics", "ggb", "morables"]   # Table 4 mean
NAME = {"harmbench": "HarmBench", "xstest": "XSTest", "mmsafetybench": "MM-SafetyBench", "airbench": "AIR-Bench",
        "wildguardtest": "WildGuardTest", "wildjailbreak": "WildJailbreak", "taxonomy": "Two-turn attacks",
        "advice": "Advice requests", "orbench": "OR-Bench-Hard", "ethics_cs": "ETHICS-cs", "llm_ethics": "LLM Ethics",
        "ggb": "GGB", "morables": "MORABLES"}
ABLATIONS = ["Energy gate only", "No probes", "8B judge", "Before the patch", "No benchmark items"]
OPEN_GUARDS = ["Llama Guard 3 8B", "Shieldstral-1.0-3B", "NeMo Guardrails", "WildGuard 7B", "Qwen3Guard-8B strict",
               "Qwen3Guard-8B loose"]
MARKS = "†‡§"


def pct(x: float) -> str:
    """Percent, rounded half-up to one decimal, as in the tables (0.0265 -> 2.7)."""
    return str((Decimal(repr(x)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def bare(s: str) -> str:
    """A row or column label without its footnote mark."""
    return s.rstrip(MARKS).strip()


def table(name: str, res: Path = RES) -> list[dict]:
    with open(res / f"{name}.csv", encoding="utf-8", newline="") as fh:
        return [{bare(k): v for k, v in r.items()} for r in csv.DictReader(fh)]


def num(c: str):
    try:
        return float(c.replace(",", ""))
    except ValueError:
        return None


def close(a: float, b: float, rel: float = 1e-12) -> bool:
    return abs(a - b) <= rel * max(1.0, abs(b))


class Counts:
    """unit_counts.json: rate(system, side, bench, part) and means."""

    def __init__(self, uc: dict):
        self.s = uc["systems"]

    def get(self, sysname, side, b, part="all"):
        return self.s[sysname][part][side].get(b)

    def rate(self, sysname, side, b, part="all"):
        x = self.get(sysname, side, b, part)
        return None if not x else x[0] / x[1]

    def mean(self, sysname, side, benches, part="all"):
        v = [self.rate(sysname, side, b, part) for b in benches]
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None

    def blocked(self, sysname, side, b, part="all"):
        k, n = self.get(sysname, side, b, part)
        return n - k if side == "apr" else k


def cell(x) -> str:
    return "—" if x is None else pct(x)


def main(res: Path = RES, cfg: Path = CFG, quiet: bool = False) -> int:
    ok, out = True, []
    C = Counts(json.loads((res / "unit_counts.json").read_text(encoding="utf-8")))
    fig = json.loads((res / "fig3_data.json").read_text(encoding="utf-8"))
    pt = json.loads((res / "paired_tests.json").read_text(encoding="utf-8"))["paired"]

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= bool(cond)
        out.append(f"{'PASS' if cond else 'FAIL'}  {label}{'  ' + detail if detail else ''}")

    def sysname(row_label: str) -> tuple[str, str]:
        """(unit_counts system, part) of a Table 3 / 4 row."""
        lab = bare(row_label)
        return ("CPEG", "all") if lab == "CPEG, all units" else (lab, "untrained")

    # 1. Table 1
    for r in table("table1_benchmarks", res):
        b = next(k for k, v in NAME.items() if v == r["Benchmark"])
        got = []
        for part in ("all", "untrained"):
            h = C.get("CPEG", "apr", b, part)
            o = C.get("CPEG", "orr", b, part) or C.s["CPEG"][part]["advice_parts"].get(b)
            has_h, has_o = C.get("CPEG", "apr", b) is not None, (C.get("CPEG", "orr", b) or C.s["CPEG"]["all"]["advice_parts"].get(b)) is not None
            got += [str(h[1]) if h else ("0" if has_h else "—"), str(o[1]) if o else ("0" if has_o else "—")]
        want = [r[c] for c in ("Harmful units", "Benign units", "Untrained harmful", "Untrained benign")]
        check(f"Table 1 {r['Benchmark']}: units from the unit counts", got == want, "/".join(got))

    # 2. Table 2
    t2 = {r["Configuration"]: r for r in table("table2_ablations", res)}
    for lab in ABLATIONS + ["CPEG"]:
        r = t2[lab]
        got = [pct(C.mean(lab, "apr", HARM)), pct(C.mean(lab, "orr", BEN)), cell(C.rate(lab, "apr", "wildjailbreak")),
               cell(C.rate(lab, "orr", "wildjailbreak")), cell(C.rate(lab, "apr", "wildguardtest")),
               cell(C.rate(lab, "orr", "wildguardtest")), cell(C.rate(lab, "apr", "taxonomy"))]
        want = [r[c] for c in ("Mean APR", "Mean ORR", "WildJailbreak APR", "WildJailbreak ORR", "WildGuardTest APR",
                               "WildGuardTest ORR", "Two-turn attacks APR")]
        check(f"Table 2 {lab}: row from the unit counts", got == want, "/".join(got))
    check("Table 2: CPEG and the five ablations", set(t2) == set(ABLATIONS + ["CPEG"]), ", ".join(t2))

    # 3. Table 3
    t3 = table("table3_apr_untrained", res)
    for r in t3:
        s, part = sysname(r["System"])
        got = [cell(C.rate(s, "apr", b, part)) for b in SIX_H] + [cell(C.mean(s, "apr", SIX_H, part)),
                                                                 cell(C.rate(s, "apr", "xstest"))]
        want = [r[NAME[b]] for b in SIX_H] + [r["Mean"], r["XSTest"]]
        check(f"Table 3 {r['System']}: row from the unit counts", got == want, "/".join(got))
    # 4. Table 4
    t4 = table("table4_orr", res)
    for r in t4:
        s, part = sysname(r["System"])
        got = [cell(C.rate(s, "orr", b, part)) for b in SIX_B] + [cell(C.mean(s, "orr", SIX_B, part)),
                                                                 cell(C.rate(s, "orr", "advice", part)),
                                                                 cell(C.rate(s, "orr", "xstest")), cell(C.rate(s, "orr", "orbench"))]
        want = [r[NAME[b]] for b in SIX_B] + [r["Mean"], r["Advice"], r["XSTest"], r["OR-Bench-Hard"]]
        check(f"Table 4 {r['System']}: row from the unit counts", got == want, "/".join(got))
    for lab, rows in (("Table 3", t3), ("Table 4", t4)):
        names = [bare(r["System"]) for r in rows]
        check(f"{lab}: CPEG, nine other systems, CPEG on all units, no benchmark items",
              names == ["CPEG"] + [n for n in C.s if n not in ABLATIONS and n != "CPEG"] + ["CPEG, all units", "No benchmark items"],
              ", ".join(names))

    # 5. Mean columns from rounded cells
    for name, rows, cols in (("Table 3", t3, [NAME[b] for b in SIX_H]), ("Table 4", t4, [NAME[b] for b in SIX_B])):
        worst = max(abs(sum(num(r[c]) for c in cols) / len(cols) - num(r["Mean"])) for r in rows)
        check(f"{name}: Mean column within 0.1 point of the mean of the rounded cells ({len(rows)} rows)", worst <= 0.1,
              f"max diff {worst:.3f}")

    # 6. Figure 3
    for k in ("CPEG", "Before the patch"):
        same = all(fig[k]["apr"][b] == C.get(k, "apr", b) for b in HARM) and all(fig[k]["orr"][b] == C.get(k, "orr", b) for b in BEN)
        check(f"Figure 3 {k}: unit counts", same)
    for grp in ("systems", "ablations"):
        bad = [n for n, v in fig[grp].items() if not (close(v["mean_apr"], C.mean(n, "apr", HARM), 1e-12)
                                                       and close(v["mean_orr"], C.mean(n, "orr", BEN), 1e-12))]
        check(f"Figure 3 {grp}: {len(fig[grp])} means = means of the unit counts", not bad and len(fig[grp]) == (10 if grp == "systems" else 5),
              ", ".join(bad))

    # 7. Paired tests
    PAIR = {"before_patch_vs_CPEG": "Before the patch", "no_probes_vs_CPEG": "No probes"}
    for tag, comp in pt.items():
        for side, s in (("harmful", "apr"), ("benign", "orr")):
            per = comp[side]["per_bench"]
            ps = {b: mcnemar(v["a_only"], v["b_only"]) for b, v in per.items()}
            adj = holm(ps)
            pool = comp[side]["pooled"]
            pp = mcnemar(pool["a_only"], pool["b_only"])
            check(f"{tag} {side}: per-benchmark McNemar p", all(close(ps[b], per[b]["p"]) for b in per))
            check(f"{tag} {side}: Holm-adjusted p", all(close(adj[b], per[b]["p_holm"]) for b in per))
            check(f"{tag} {side}: pooled = sum of benchmarks, pooled McNemar p",
                  close(pp, pool["p"]) and pool["a_only"] == sum(v["a_only"] for v in per.values())
                  and pool["b_only"] == sum(v["b_only"] for v in per.values()),
                  f"a_only {pool['a_only']}, b_only {pool['b_only']}, p {pp:.2g}")
            a = PAIR[tag]
            bad = [b for b, v in per.items() if C.blocked(a, s, b) - C.blocked("CPEG", s, b) != v["a_only"] - v["b_only"]
                   or C.get("CPEG", s, b)[1] != v["n"]]
            check(f"{tag} {side}: discordant counts agree with the unit counts", not bad, ", ".join(bad))

    # 8. Figures quoted in the text
    def p2(a, b):
        return f"{mcnemar(a, b):.2g}"

    def pp(tag, side, b=None):
        x = pt[tag][side]["pooled"] if b is None else pt[tag][side]["per_bench"][b]
        return x["a_only"], x["b_only"]
    # 5.1 energy gate only: every block of the energy gate alone is a CPEG block, so the discordant units are the
    # difference of the block counts
    e, c = "Energy gate only", "CPEG"
    sub = all(C.blocked(e, s, b) <= C.blocked(c, s, b) for s, bs in (("apr", HARM), ("orr", BEN)) for b in bs)
    dh = {b: C.blocked(c, "apr", b) - C.blocked(e, "apr", b) for b in HARM}
    do = {b: C.blocked(c, "orr", b) - C.blocked(e, "orr", b) for b in BEN}
    n_all = sum(C.get(c, "apr", b)[1] for b in HARM) + sum(C.get(c, "orr", b)[1] for b in BEN)
    tot = sum(dh.values()) + sum(do.values())
    got = (sub, sum(dh.values()), dh["wildjailbreak"], p2(0, sum(dh.values())), sum(do.values()), do["orbench"],
           p2(0, sum(do.values())), tot, n_all, f"{100 * tot / n_all:.2f}",
           f"{100 * (C.mean(e, 'apr', HARM) - C.mean(c, 'apr', HARM)):.1f}", f"{100 * (C.mean(c, 'orr', BEN) - C.mean(e, 'orr', BEN)):.1f}")
    check("text 5.1: piece gate adds 63 harmful blocks (31 WildJailbreak; p 2.2e-19) and 37 benign (23 OR-Bench-Hard; "
          "p 1.5e-11), 100 of 12,578 units (0.80%), APR -0.8 and ORR +1.0 points",
          got == (True, 63, 31, "2.2e-19", 37, 23, "1.5e-11", 100, 12578, "0.80", "0.8", "1.0"), str(got))
    # 5.1 no probes
    t = "no_probes_vs_CPEG"
    got = (pp(t, "harmful"), p2(*pp(t, "harmful")), pp(t, "harmful", "wildjailbreak")[0], pp(t, "harmful", "airbench")[0],
           pp(t, "benign"), p2(*pp(t, "benign")))
    check("text 5.1: no probes 78 vs 12 harmful (p 5.2e-13; 44 WildJailbreak, 22 AIR-Bench), 8 vs 7 benign (p 1)",
          got == ((78, 12), "5.2e-13", 44, 22, (8, 7), "1"), str(got))
    # 5.2 before the patch
    t, b = "before_patch_vs_CPEG", "Before the patch"
    sig = {side: sorted(k for k, v in pt[t][side]["per_bench"].items() if v["p_holm"] < 0.05) for side in ("harmful", "benign")}
    got = (pp(t, "benign"), p2(*pp(t, "benign")), pp(t, "benign", "wildguardtest"), pp(t, "benign", "wildjailbreak"),
           pp(t, "harmful"), p2(*pp(t, "harmful")), pp(t, "harmful", "wildguardtest"), pp(t, "harmful", "wildjailbreak"),
           pp(t, "harmful", "airbench"), pp(t, "harmful", "taxonomy"), sig["harmful"], sig["benign"])
    check("text 5.2: before the patch 644 vs 37 benign (p 3.8e-144; WildGuardTest 445 vs 1, WildJailbreak 182 vs 0), "
          "216 vs 87 harmful (p 8.1e-14; 68 vs 21, 88 vs 55, AIR-Bench 37 vs 2, two-turn 10 vs 0); no other benchmark "
          "significant after Holm", got == ((644, 37), "3.8e-144", (445, 1), (182, 0), (216, 87), "8.1e-14", (68, 21),
                                           (88, 55), (37, 2), (10, 0), ["airbench", "taxonomy", "wildguardtest", "wildjailbreak"],
                                           ["wildguardtest", "wildjailbreak"]), str(got))
    got = tuple(cell(C.rate(s_, "apr", k)) for k in ("wildjailbreak", "wildguardtest", "airbench", "taxonomy") for s_ in (b, c))
    got += (pct(C.mean(b, "orr", BEN)), pct(C.mean(c, "orr", BEN)), pct(C.mean(b, "apr", HARM)), pct(C.mean(c, "apr", HARM)))
    check("text 5.2: APR WildJailbreak 3.8 -> 5.4, WildGuardTest 7.7 -> 13.9, AIR-Bench 0.8 -> 3.2, two-turn 0.0 -> 1.3; "
          "mean ORR 17.2 -> 3.4, mean APR 2.3 -> 3.9", got == ("3.8", "5.4", "7.7", "13.9", "0.8", "3.2", "0.0", "1.3",
                                                            "17.2", "3.4", "2.3", "3.9"), str(got))
    rest_h, rest_b = [k for k in HARM if not k.startswith("wild")], [k for k in BEN if not k.startswith("wild")]
    three = ["harmbench", "xstest", "mmsafetybench"]
    got = tuple(pct(C.mean(s_, "apr", rest_h)) for s_ in (b, c)) + tuple(pct(C.mean(s_, "apr", three)) for s_ in (b, c)) + \
        tuple(pct(C.mean(s_, "orr", rest_b)) for s_ in (b, c))
    check("text 5.2: other benchmarks: mean APR 0.9 -> 1.7 (HarmBench, XSTest, MM-SafetyBench 1.3 in both), mean ORR 0.6 -> 2.0",
          got == ("0.9", "1.7", "1.3", "1.3", "0.6", "2.0"), str(got))
    pc = json.loads((cfg / "energy_pool_composition.json").read_text(encoding="utf-8"))
    patch = {k: v for k, v in pc["by_source"].items() if k.startswith(("fit:wildjailbreak_", "fit:wildguardmix_"))}
    rel = sum(v for k, v in pc["advice_label_rule"]["relabelled_rows_by_source"].items() if k in patch)
    harm = sum(v for k, v in patch.items() if k.endswith("_harmful")) - rel
    got = (sum(patch.values()), harm, sum(patch.values()) - harm, pc["n"], pc["labels"]["1"], pc["labels"]["0"])
    check("text 3.6 / 5.2: patch rows 41,422 of 108,897 (19,750 harmful, 21,672 benign); pool 37,557 harmful, 71,340 benign",
          got == (41422, 19750, 21672, 108897, 37557, 71340) and sum(pc["by_source"].values()) == pc["n"], str(got))
    # 5.2 no benchmark items (units untrained for CPEG; XSTest and OR-Bench-Hard on all units)
    nb, u = "No benchmark items", "untrained"
    lower_all = all(C.rate(nb, "apr", k, u) > C.rate(c, "apr", k, u) for k in SIX_H)
    four = [k for k in SIX_B if k not in ("llm_ethics", "ggb")]
    mx = max(abs(C.rate(nb, "orr", k, u) - C.rate(c, "orr", k, u)) for k in four)
    got = (lower_all, pct(C.mean(nb, "apr", SIX_H, u)), pct(C.mean(c, "apr", SIX_H, u)), cell(C.rate(nb, "apr", "mmsafetybench", u)),
           cell(C.rate(c, "apr", "mmsafetybench", u)), pct(C.mean(nb, "orr", SIX_B, u)), pct(C.mean(c, "orr", SIX_B, u)),
           cell(C.rate(nb, "orr", "llm_ethics", u)), cell(C.rate(c, "orr", "llm_ethics", u)), cell(C.rate(nb, "orr", "ggb", u)),
           cell(C.rate(c, "orr", "ggb", u)), pct(mx), cell(C.rate(nb, "orr", "xstest")), cell(C.rate(nb, "orr", "orbench")),
           cell(C.rate(c, "orr", "xstest")), cell(C.rate(c, "orr", "orbench")))
    check("text 5.2: no benchmark items -> CPEG on untrained units: APR lower on all six, mean 12.4 -> 5.5 (MM-SafetyBench "
          "25.9 -> 3.3), mean ORR 9.9 -> 4.1 (LLM Ethics 25.7 -> 0.0, GGB 15.6 -> 7.8), other four at most 1.4 points; "
          "XSTest 6.8 and OR-Bench-Hard 25.2 against 1.2 and 2.8",
          got == (True, "12.4", "5.5", "25.9", "3.3", "9.9", "4.1", "25.7", "0.0", "15.6", "7.8", "1.4", "6.8", "25.2", "1.2", "2.8"),
          str(got))
    # 5.4 advice requests
    adv = C.get(c, "orr", "advice")
    parts = C.s[c]["all"]["advice_parts"]
    og = {g: C.rate(g, "orr", "advice", u) for g in OPEN_GUARDS}
    lo, hi = min(og, key=og.get), max(og, key=og.get)
    got = (adv, cell(adv[0] / adv[1]), C.get(c, "orr", "advice", u)[1], parts["mmsafetybench"][0], cell(parts["mmsafetybench"][0] / parts["mmsafetybench"][1]),
           parts["airbench"], lo, cell(og[lo]), hi, cell(og[hi]), cell(C.rate(c, "orr", "advice", u)))
    check("text 5.4: CPEG blocked 9 of 413 advice requests (2.2%); 187 untrained; 3 MM-SafetyBench (0.7% of 406), 6 of 7 "
          "AIR-Bench; open guards 5.3% (Qwen3Guard-8B loose) to 47.1% (Llama Guard 3 8B) against 2.1%",
          got == ([9, 413], "2.2", 187, 3, "0.7", [6, 7], "Qwen3Guard-8B loose", "5.3", "Llama Guard 3 8B", "47.1", "2.1"), str(got))
    # 5.5 energy gate only on the units of Tables 3 and 4
    got = (pct(C.mean(e, "apr", SIX_H, u)), pct(C.mean(e, "orr", SIX_B, u)))
    check("text 5.5: energy gate only, mean APR 6.6 (Table 3 units) and mean ORR 3.1 (Table 4 units)", got == ("6.6", "3.1"), str(got))
    # 6.1 / 6.2
    om = {g: C.mean(g, "orr", SIX_B, u) for g in OPEN_GUARDS}
    others = [n for n in C.s if n not in ABLATIONS and n != "CPEG"]
    gap = {k: 100 * (C.rate(c, "apr", k, u) - min(C.rate(n, "apr", k, u) for n in others)) for k in ("wildjailbreak", "wildguardtest")}
    got = (pct(C.mean(c, "orr", SIX_B, u)), pct(min(om.values())), pct(max(om.values())),
           pct(C.mean("Qwen3Guard-8B loose", "apr", SIX_H, u)), pct(C.mean("Llama Guard 3 8B", "apr", SIX_H, u)),
           pct(C.mean(c, "apr", SIX_H, u)), f"{gap['wildjailbreak']:.1f}", f"{gap['wildguardtest']:.1f}")
    check("text 6.1 / 6.2: untrained mean ORR 4.1 within the open guards' 3.8-23.6; Qwen3Guard-8B loose and Llama Guard 3 "
          "8B mean APR 19.9 and 31.1 against 5.5; WildJailbreak and WildGuardTest 2.3 and 1.7 points above the lowest APR",
          got == ("4.1", "3.8", "23.6", "19.9", "31.1", "5.5", "2.3", "1.7"), str(got))

    if not quiet:
        print("\n".join(out))
        print("all checks passed" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
