# Dream-enabled continuation, capped at $10

On September 28, 2026, the user requested controller dreaming in the ongoing mixed-workload search. The fixed-controller coordinator was stopped, its saved responses were reconciled without new HTTP requests, and its histories were continued with **two controller proposals between search cycles**.

A dream proposal changes the rules for selecting which search branches to open or refine. The harness evaluates it by replaying recorded training histories with outcomes hidden until selected. It adopts the proposal only when its mean replay value strictly exceeds the incumbent's value on the same complete history pool. An accepted controller selects branches in the next live cycle. Replay itself makes no model calls. Better replay scores do not establish better live search; subsequent live policy results measure that.

The current partial search cycle finishes under its original controller. Dreaming begins at its next cycle boundary. Final cycles do not spend controller calls when no subsequent discovery cycle can use them. The existing limits remain four online rounds per cycle, six replay rounds, two discovery workers, depth four, and at most twenty cycles. A cycle can end early if its controller selects no actions.

## Budget and experiment boundary

The user raised the combined new-spending cap from **$4 to $10** before the adaptive stage started. This is $10 across the fixed and adaptive stages, not an additional $10. All earlier charges and uncertain request reservations carry forward. Per-arm ceilings increase proportionally to the original allocation; the spending ledger does not reset. Saved completed API responses are evaluated locally; interrupted requests with unknown usage retain their reservation and are never resent. Requests proved never submitted are recorded as cancelled with zero charge.

The original limit of 100 calls per arm now counts both policy and controller attempts. Milestones at 12, 25, 50 and 100 calls use that same total-attempt basis. Consequently the adaptive stage can produce fewer cache policies. Current reservations and accepted controller-update counts appear in the live report.

The three paired trials still compare `block-v1` and `task-v1`. Their controller behavior is adaptive after a fixed-controller warmup. **This is a staged continuation, not a controlled fixed-versus-adaptive comparison.** The transition call, cycle, round and spending are recorded separately for every arm in `plan.json`. Discovery prompts, feature sets, model, scores, workload traces and capacities are unchanged. Controller prompts use the existing harness prompt. If necessary to fit its input allowance, redundant replay trajectories are omitted from the prompt; every historical node remains represented, and acceptance still uses all recorded histories.

All searches finish and their training winners are frozen before validation. Validation remains previously used development data. The final test stays unopened.

## Reproduction and progress

The previous run is preserved at `runs/mixed-task-long-20260928` and marked superseded. It must not be resumed alongside its continuation.

The one-time migration command, already applied, is:

```sh
.venv/bin/python -m dream_rsi.mixed_long \
  --continue-from runs/mixed-task-long-20260928 \
  --output runs/mixed-task-dream-20260928 \
  --controller-revisions 2 --prepare-only
```

Resume only the new directory:

```sh
.venv/bin/python -m dream_rsi.mixed_long \
  --output runs/mixed-task-dream-20260928 --resume
```

Inspect `report.md`, `report.json`, `coordinator.log`, and each arm's `controller-revisions-*.json`. Empty revision logs correspond to fixed-controller warmup cycles. The source snapshot, predecessor checkpoint, transition plan, usage and histories are retained locally. The fixed-stage archive is under `experiment-results/mixed-task-long-fixed-stage`.

The full suite passed 100 tests before migration. Tests exercise actual controller proposal/replay cycles, strict acceptance, checkpoint reloads, offline response reconciliation, retained unknown charges, preserved budget ceilings during migration, duplicate-migration rejection, and validation isolation.
