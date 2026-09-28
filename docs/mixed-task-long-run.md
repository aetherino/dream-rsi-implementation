# Longer mixed-workload search

The September 28 continuation tests whether more proposals help under the unchanged pilot setup. It targets three paired trials of `block-v1` versus `task-v1`, with up to 100 total proposals per arm. Trial 1 resumes each 12-call pilot history, including its partially completed second cycle. Trials 2 and 3 start independently from LRU, without importing candidate histories. These are stochastic replications, not seeded deterministic API runs.

Prompts, model, task features, expression language, controller, scoring, capacities and workloads remain unchanged. The controller is fixed; there are no controller revisions or hints about inactive task conditions. New cycles still start from LRU while retaining the prior search history, as in the pilot. The cycle ceiling increases to 20 to accommodate the longer search.

## Spending limit

The cap is **$4 of new API spending**, excluding the completed pilot and all earlier experiments. The target allows at most 576 new proposals, but the budget takes priority and may stop arms before 100 total calls. Per-arm allocations are proportional to remaining proposals: $0.611111 for each continued arm and $0.694444 for each fresh arm, totaling $3.999998. Unused allocations are not transferred between arms.

Budget accounting uses the overseas realtime MiMo-v2.6-pro uncached rates verified on [MiMo pricing](https://mimo.mi.com/docs/en-US/price/pay-as-you-go) on September 28, 2026: $0.435 per million input tokens and $0.87 per million output tokens. Input-cache discounts are ignored. Each request reserves its input allowance and maximum completion cost before submission. Unknown outcomes retain that reservation and are never automatically resubmitted. This is a conservative token-cost cap at those rates, not a provider-side account spending limit.

Requests run concurrently through the regular API, with four HTTP workers; this is not the provider's discounted asynchronous batch service. Simulator evaluations run on CPU.

## Selection and checkpoints

Training-selected policies are frozen at 12, 25, 50 and 100 calls when reached. All six searches finish and their final policies are frozen before validation starts. Validation outcomes never enter discovery prompts. Validation is previously used development data, not a fresh holdout. The official final test remains unopened.

Every wave writes a checkpoint, per-request outcomes, usage, histories, prompts, milestones and a report. The output directory also stores the source snapshot and provenance hashes. Resume checks the source and prepared data and rejects changes. A process lock prevents two coordinators using the same output directory.

## Commands

From the repository root, prepare without making API calls:

```sh
.venv/bin/python -m dream_rsi.mixed_long \
  --output runs/mixed-task-long-20260928 --max-new-usd 4 --prepare-only
```

Start or resume that exact plan:

```sh
.venv/bin/python -m dream_rsi.mixed_long \
  --output runs/mixed-task-long-20260928 --resume
```

A resume uses the saved budget and settings; it does not reset spending. Inspect `report.md`, `report.json`, and `coordinator.log` in the run directory for progress. Raw run artifacts are local and ignored by Git; the historical result archives remain separately versioned. Final results should be archived after completion, with token/text data excluded.

The continuation tests cover imported partial-cycle history, independent fresh starts, checkpoint reloads, validation ordering, conservative unknown charges, tiny budgets and changed source rejection. The full suite passed 99 tests before launch.
