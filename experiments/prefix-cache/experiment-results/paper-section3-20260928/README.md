# Executable Section 3 comparison — 2026-09-28

Completed fixed-controller versus dreaming-controller cache-policy search. Fresh $10 combined allowance; estimated actual API cost **$9.41479026** (not a provider invoice).

| Arm | Policy proposals | Controller proposals | Training reduction | Validation reduction |
| --- | ---: | ---: | ---: | ---: |
| Fixed | 96 | 0 | 0% | 0% |
| Dream | 63 | 18 | 4.7584% | 2.6110% |

Reductions measure extra token recomputation relative to LRU. Dream saved 3,584 tokens on development validation. Both selected policies passed evaluation. This is one comparison, not evidence of a reliable causal advantage or measured GPU speedup. The final test was not evaluated.

Readable files include the plan, report, selected programs, usage, controller revisions and validation metrics. `run-records.tar.gz` contains complete saved histories, prompts, evaluations, transport journals, checkpoint and frozen source. `manifest.json` records original and published hashes for every archived file. Local absolute paths in records are replaced by `<ORIGINAL_EXPERIMENT_ROOT>`; frozen source is verbatim. Runtime lock/PID files and bytecode are omitted. Credentials and datasets are excluded.

Extract into a scratch directory for inspection. This completed archive is evidence, not a command to resume paid requests. Dataset hashes and setup are documented in the experiment; API sampling is nondeterministic.
