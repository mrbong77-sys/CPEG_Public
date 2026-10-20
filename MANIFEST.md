# Manifest

Every file of this package, where it came from in the original (private) research repository, and what was changed.
Source paths are relative to that repository's root; `<src>/` stands for its Python package directory. No logic was
changed unless stated: the package outputs were checked against the original functions on the same inputs (parser
pieces and features, probes, band routing, operating-point rule, explanation records, unit rates, paired tests,
two-turn construction and sampling, thresholds) and were identical.

Changes common to all code files: module and package names changed to `cpeg.*`; imports trimmed to the modules
included here; functions not on the inference or evaluation path removed (data collection, training-pool assembly,
benchmark runs, caching to the original repository layout, commit helpers, latency harnesses); comments and docstrings
reworded without internal document references, dates or names; identifiers renamed only where they carried the
original system name.

## Code

| File | Source | Changes |
|---|---|---|
| `cpeg/__init__.py`, `cpeg/gate1/__init__.py`, `cpeg/energy/__init__.py`, `cpeg/eval/__init__.py` | — | new (one-line docstrings) |
| `cpeg/gate1/base.py` | `<src>/c/gate1.py` | kept: parser constants, `_words`, `_common_ratio`, `_printable`, `_decoded`, `_sha`, `thresholds`, `route`. Removed: `parse`/`features` of the first Gate1 parser (not used by the final Gate1), fit pool, bench sets, feature runs, head runs, latency, Gate2 stand-in. Zero-width and curly-quote characters in two regexes written as `\u` escapes (same matches). |
| `cpeg/gate1/units.py` | `<src>/c/gate1v2.py` | kept: `FRAME`, `REQ_START`, `REQ_ANY`, `FEATS`, `SAFE_Y`/`HARM_Y`, `role`, `split`, `units`, `features`. Removed: feature/run/latency commands. |
| `cpeg/gate1/pieces.py` | `<src>/c/gate1v3.py` | kept: constants, `CLAUSE`, `FEATS`, `clauses`, `deframe`, `units`, `features`, `_boost`. Removed: feature runs, head evaluation, cost curve, latency. |
| `cpeg/gate1/frames.py` | `<src>/c/gate1v5.py` (+ `S_HOT`, `S_FEATS`, `S_LEN` from `<src>/c/gate1v4.py`) | kept: `S_ENC`, frame constants, `FEATS` (25 features), `units`, `s_sel`, `features`. Removed: training-data builders, head runs, revision config objects. |
| `cpeg/gate1/encoders.py` | `<src>/n/student.py` (`encode`, `pool`, `predict`), `<src>/student/model.py` (`hf`), `<src>/c/kbench.py` (`_cal` → `platt`), `<src>/c/gbench.py` (`_logit`, `platt` → `fit_platt`), `<src>/c/train.py` and `<src>/c/gate1v4.py` (scoring settings) | Module-level settings (max length 256, head + tail truncation, [CLS] pooling, no autocast) that the original set by context managers are passed as fixed defaults; `predict` takes the encoder name instead of a factory. Same computation. Training code not included. Release addition: `load_state` also reads the one-file safetensors form of a checkpoint (`backbone.*`, `head.*`). |
| `cpeg/gate1/head.py` | `<src>/c/e2e.py` (`v3_tiers`), `<src>/c/jevroute.py` (`_v3`), `<src>/c/gate1c2.py` (`rev_thresholds`), `<src>/c/gate1v10.py` (`HI_Q`), `<src>/c/gate1v9.py` (head `v6_boost` = boosted trees on the 25 features), `<src>/c/gate1.py` / `<src>/c/gate1v4.py` (`score_pieces`, `score_s`) | Re-assembled as one module. The original feature function of the final Gate1 also computed judge-LLM features for three Qwen3Guard sizes, which the final head (`v6_boost`) does not read; here the 25 features are computed directly (identical values for the columns the head uses) and no judge LLM is loaded. Thresholds default to the registered values from `config/gate1.json` instead of being recomputed at load time (`thresholds="refit"` recomputes them). Piece-score disk caches removed. |
| `cpeg/energy/probes.py` | `<src>/c/energy.py` (`CF_MAL`, `CF_BEN`, `CF_LEX`, `CF_PEOPLE`, `CF_PROTECT`, `CFX_PROBES`, `_cf_mask`, `_cf_object`, `cf_variants`) | `CF_VARIANTS` default set to 6 (the original default was 4 and the final system always passed 6). |
| `cpeg/energy/encoder.py` | `<src>/c/ftenc.py` | kept: `row_weights`, `batches`, `_tok`, `_det`, `train`, `score`, `_score`, `_ckey`. Removed: subprocess runner and CLI; support for [request, probe] pair inputs (used only by a configuration not reported); `balanced_rows`. `score` gained `cache=True/False` (the original always cached; outputs are the same). Added constant `FINAL_EPOCHS = 2` documenting the epochs of the CPEG encoder. |
| `cpeg/energy/head.py` | `<src>/c/energy.py` (`cfx_features`, `cfx_energy`, `cfx_head`, `explain`) | `explain` takes the probe features `Z` instead of texts (the original computed `Z` inside). `cfx_features` passes `cache` to `encoder.score`. |
| `cpeg/energy/advice.py` | `<src>/c/advice.py` (`dna`, `_sorry_rows`, `_sorry_cats`, `sorry_delegate` → `sorry`, `salad`, `jbv`, `SORRY_WRAPPED`, `JBV_ADVICE`, `DELEGATE` → `SOURCES`, `delegate_sources` → `load_sources`), `<src>/c/energy.py` (`delegate_texts`, `drop_texts`, `relabel_advice`; the drop-then-relabel step of the pool preparation → `apply_rule`; `advice_texts` → `benchmark_advice_texts`) | new module for the advice-label rule. Category lists written as named constants (`DNA_ADVICE`, `SORRY_ADVICE`, `SALAD_ADVICE`). `load_sources` returns the texts instead of writing a cache file and committing counts; `delegate_texts` takes the loaded texts and the benchmark advice texts as arguments instead of reading files; `apply_rule` returns the pool and its counts instead of writing them. Same selection, same matching (stripped text), same order (drop, then relabel). |
| `cpeg/route.py` | `<src>/c/energy.py` (`band_tier`, `cf_select` → `select`, `cf_bands` → `bands`, `CF_GRID` → `GRID`), `<src>/c/e2e_peaos.py` (`V3_DELIBERATION_SYS` → `DELIBERATION_SYSTEM_PROMPT`), `<src>/c/paperbench.py` (energy-gate-only rule → `energy_only`) | `bands` takes the rule config `FINAL_RULE` (the band-relevant keys of the final configuration) instead of a configuration number; the result keys of the first-stage baseline renamed `gate1_harmful_*`. Docstring carries the deployed band values. |
| `cpeg/pipeline.py` | `<src>/c/paperbench.py` (`_tiers`: Gate1 tiers, energy, `band_tier`, the energy-gate-only and no-probes ablations) | new glue: loads the configuration, runs Gate1 and the energy gate, returns explanation records; `energy_gate_only_tiers` and `no_probes_tiers` (the request logit with the no-probes bands, as the original scored that ablation with the encoder alone and its own bands). Release additions: `CPEG.from_pretrained` loads the model-repository layout (safetensors piece-gate files, the energy-encoder folder, the full-precision head `heads/cpeg.json`) and turns the encoder's score cache off (`score_cache`). |
| `cpeg/eval/metrics.py` | `<src>/c/paperbench.py` (`unit_rates`, `trained_flags`), `<src>/c/prepatch.py` (`mcnemar`, `holm`, `unit_blocks`, `paired`, `means`) | `trained_flags` takes the overlap flags and the extra trained units as arguments instead of reading files. Rate key `asr` → `apr` (the paper's attack pass rate); `unit_rates` also returns the counts (`not_blocked`, `blocked`, `routed_units`). New `side_rates` and `means` pool the benign advice units of MM-SafetyBench and AIR-Bench into one benchmark, `advice`, and average unrounded count ratios, as the paper's tables do (`<build>/make_displays.py`: `Data.o`, `Data.mean`); `paired` pools the same units (`pool_advice=True`). |
| `cpeg/eval/overlap.py` | `<src>/c/energy.py` (`embed`), `<src>/c/paperbench.py` (`overlap`) | `overlap_flags` takes the pool texts as an argument and returns the flags instead of writing files. |
| `cpeg/eval/units.py` | `<src>/c/paperbench.py` (`SRC`, `TAXONOMY`, `ETH_PROMPT`, `MORABLES_PROMPT`, `SAMPLE`, `_csv`, `_unit`, `harmbench_rows`, `build`, `sample`, `KSAFE_ADVICE` → `MMSB_ADVICE`, `AIR_ADVICE`, `is_advice`, `advice_label`), `<src>/c/newbench.py` (`_load` → `_gated`) | Benchmark keys renamed: `ksafe` → `mmsafetybench`, `ethics` → `llm_ethics`, `ethicscs` → `ethics_cs` (affects unit ids only; the sampling salts of the sampled benchmarks are unchanged). The advice units get label 0 inside `build` (the original relabelled them after reading, `advice_label`; same labels). The MM-SafetyBench source URL is not listed (the original fetched the benchmark's question files from a byte-identical third-party mirror); `get` supplies the files. Download/caching code removed. |
| `cpeg/eval/recompute.py` | — | new: recomputes Tables 1–4, the Figure 3 data and the figures quoted in the text from `results/unit_counts.json` and `results/paired_tests.json`. |
| `tests/test_cpeg.py` | — | new: checks without checkpoints (in memory; no files written). |
| `requirements.txt` | `pyproject.toml` (dependencies) | new list. |
| `README.md`, `MANIFEST.md` | — | new. |
| `LICENSE` | — | Apache License 2.0 of the code (the checkpoints, configuration, results and records are licensed separately, `README.md` "Licence"). |
| `cpeg/__main__.py` | — | new: command line over `CPEG.from_pretrained`. |

`<build>/` = the build script of the manuscript's tables and figures (not included).

## Configuration

| File | Source | Changes |
|---|---|---|
| `config/operating_point.json` | operating-point records (`route.json`: `bands`, `selection.cf.chosen`) of the final energy gate and of its encoder alone (no probes) | both deployed band dicts copied (keys `cpeg`, `ablation_no_probes`; `ablation_energy_gate_only` states its rule); checked against the band rule (θ_low = θ_release = θ_del = the chosen 95th percentile, θ_block = the 98th percentile, no `g1_low_block`); notes added. |
| `config/energy_head.json` | head-fit record of the final energy gate (`fit.json`: `n`, `n_val`, `head`, `heads`) and fine-tuning record of its encoder (`fit.json`: `encoder`, `heads`) | weights, intercept, validation AUROCs and recipe copied; feature order and meanings added. |
| `config/gate1.json` | `results/c/gate1v10/summary.json` (`heads.v6_boost.thresholds`), `results/c/k2/summary.json` (`operating.v32`: `cal`, `tau`) | values copied; head and classifier specification written from the code constants. Unchanged by this update. |
| `config/gate1_head_fit.csv.gz` | `results/c/gate1v10/feat_fit.csv` | columns `half`, `y`, `harmful` and the 25 head features kept; `id`, `source` and the unused judge-LLM feature columns dropped; gzip. Unchanged by this update. |
| `config/energy_pool_composition.json` | pool records of the final encoder (`data.json`: `n`, `labels`, `dropped_advice_wrapped`, `relabelled`; rows per source and held-out sets from the pool record it was copied from), `results/c/advice/delegate_sources.json`, `results/c/energy/delegate_check.json` | `by_source` = the copied pool's rows per source minus the dropped wrapped-advice rows (sum checked = 108,897); labels and per-source relabel / drop counts copied (checked: labels before the rule + relabelled = labels after); source keys `eval:ksafe`, `fit:pb_ksafe`, `eval:ethics`, `eval:ethicscs` renamed to `eval:mmsafetybench`, `fit:pb_mmsafetybench`, `eval:llm_ethics`, `eval:ethics_cs`; the rule and its category selection written out (`advice_label_rule`); description added. |

## Results

Every file of `results/` is written by the original repository's release script from the committed result files that
the manuscript's table builder (`<build>/`) reads: the builder's table function is run on those files, its tables are
written as CSV, and every cell is compared with the tables of the manuscript (all cells agreed). Label renames in
every file: the system name → `CPEG`; the ablations → their names in the paper (energy gate only, no probes, 8B judge,
before the patch, no benchmark items); `ksafe` → `mmsafetybench`, `ethics` → `llm_ethics`, `ethicscs` → `ethics_cs`,
`asr` → `apr`.

| File | Source | Changes |
|---|---|---|
| `results/table1_benchmarks.csv` … `results/table5_reasoning_cost.csv` | the paper's Tables 1–5 as the table builder writes them from the committed results (unit-level routing records, the manuscript recount, the reasoning-block and latency tables) | markdown bold removed; thousands separators removed from plain counts (cells such as `0 / 36,891 (0%)` kept as printed); footnote marks († ‡ §) kept. |
| `results/fig3_data.json` | the table builder's Figure 3 data | keys renamed (`final` → `CPEG`, `a6` → `Before the patch`, `asr` → `apr`, `map` → `systems`, ablation keys → paper names; benchmark keys as above); [count, units] pairs and named mean fields; description added. The weight-tuned defences of the figure (final answers) are not included. |
| `results/unit_counts.json` | the table builder's count layer (rate × n from the unit-level routing records, exact for n ≤ 2,000 and 4-decimal rates; checked by the builder against the recount's same-unit counts) | new: per system, all units and units untrained for CPEG, [not blocked, units] per harmful benchmark and [blocked, units] per benign benchmark, the advice units pooled (parts kept). |
| `results/paired_tests.json` | `results/c/prepatch/summary.json` (`paired`) | comparison keys → `before_patch_vs_CPEG`, `no_probes_vs_CPEG`; the benign advice units of MM-SafetyBench and AIR-Bench pooled as `advice` (counts summed; the two parts under `parts`); the pooled `advice` McNemar p and the benign-side Holm values recomputed with `cpeg.eval.metrics` over the 9 benign benchmarks; all other values copied. Replaces `prepatch_summary.json` of the earlier package (same paired values; its per-benchmark rates are in `unit_counts.json`). |

Removed with earlier updates (no longer in the paper): the composition-ablation, encoder-capacity, untrained-ASR,
reasoning-items, weight-tuned-defence (ASR, ASR as classifiers, ORR, degenerate output, reasoning), classifier
comparison, McNemar and cost tables of the first package (`t01`–`t18`); the piece-gate-only rows of the earlier
Table 2. Their content that the paper keeps is in Tables 1–5.
