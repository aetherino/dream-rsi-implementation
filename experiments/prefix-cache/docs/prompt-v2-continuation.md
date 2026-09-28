# Prompt v2 and the resumed search

The user requested prompt revisions during the diagnostic study. The v1
coordinator was interrupted after all 33 requests in its current wave had saved
their responses. Those results were evaluated locally and applied to a settled
checkpoint without resubmitting any model requests. The old run is preserved as
`runs/controller-transfer-20260927`, status `superseded_by_prompt_v2`.

At the transition, the diagnostic study had used **219 of 420 new calls** and
an estimated **$0.709812555 of its $3 allowance**. The continuation retains all
paid attempts, training histories, best policies, fixed controllers, and per-run
call/dollar limits. It permits at most **201 additional calls** within the same
study budget. The $0.9424 spent on the preceding completed experiment remains
separate; imported continuation costs are not counted twice.

## Prompt changes

- Explain that expression-size/depth bounds are maximums, not targets; require
  exactly the allowed JSON keys, and request a short, explicitly hypothetical
  rationale instead of long unsupported explanations.
- Give a concrete leaf-eviction example. Evicting C from resident A → B → C
  preserves cached A and B. Depth is not itself a per-eviction cost multiplier.
- Explain score direction, common additive terms that cannot change rankings,
  and that policy cost includes scoring every eligible entry.
- Supply authoritative measured aggregate saved tokens, denominator, and score.
- Always include the best-ever training policy, even after it leaves the recent
  rollout window. Add a run-wide record of attempted expressions, outcome scores,
  repeat counts, and recent errors. Formatting-equivalent expressions are grouped
  by their parsed syntax; algebraic equivalence is not claimed.
- Bound the formula memory to 18 KB. If it must be compacted, report the number
  of omitted formulas explicitly. The parent and best-ever policy remain present.
  Older auxiliary observations can be removed to stay within the API input guard.
- Give controller proposals an explicit descending-priority example, the
  distinct-node constraint, and separate live/replay horizons (three versus six
  rounds for this study). Require the actual online limit rather than guessing it.

The legacy prompt text remains available through `prompt_version: "v1"`.
Newly prepared configurations default to v2. Existing checkpoint configurations
without a prompt version retain v1 behavior unless explicitly migrated. Each
new coordinator request records its prompt version.

## Interpretation and recovery

The new output directory is `runs/prompt-v2-continuation-20260927`. It records the
source checkpoint hash, every run's first v2 call number, code snapshots, and all
historical spending. Validation shards 13–18 were still unobserved at the change
and remain absent from prompts. The evaluator, objective, model, thinking mode,
controller programs, and workload stay unchanged. Both policy and controller
prompt functions are revised; this diagnostic continuation keeps controllers
frozen, so its remaining API requests propose cache policies only.

This is **a search continuation, not a controlled prompt A/B experiment**.
Its histories mix v1 and v2 proposals, with different call boundaries per run.
Final arm differences cannot isolate the causal effect of the prompt change.
Reports display this limitation explicitly.

The migration refuses outstanding requests, observed validation, or changes to
non-prompt simulator/scoring modules. It does not reset budgets or send requests
that already have results. Resume the new coordinator with:

```sh
.venv/bin/python -m dream_rsi.diagnostics \
  --output runs/prompt-v2-continuation-20260927 --resume
```

The explicit migration command, for an already settled checkpoint, is:

```sh
.venv/bin/python -m dream_rsi.diagnostics \
  --source runs/controller-transfer-20260927/checkpoint.json \
  --revise-prompts --output runs/prompt-v2-continuation-20260927
```

Do not execute this again into another directory to continue the same budget;
use `--resume` with the existing continuation instead.
