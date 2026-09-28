# Follow-up: longer fixed searches and prospective controller tests

The 240-call experiment finished at an estimated $0.9424. Fixed-controller runs
reduced held-out extra recomputation by 22.75% on average; adaptive runs reduced
it by 13.39%. Four controller revisions passed historical replay, but that is not
evidence they improved subsequent discovery. Two fixed runs were still improving
on their last allowed call.

The next study is frozen in `configs/dream-diagnostics.json`:

- Continue **all three** fixed runs from 40 to **100 total calls**, retaining their
  actual training histories and best policies. This permits 180 additional calls.
- Extract **every accepted revision**, together with the incumbent it replaced.
  Do not pick transitions by whether their later results were favorable.
- For each of the four transitions, run **two independent replicates per arm**
  with a **15-call allowance**. Both predecessor and revised controllers stay
  frozen. All 16 discovery runs start at LRU with empty histories; they share no
  generated policies across arms. This permits 240 additional calls.
- Keep MiMo-V2.6-Pro, thinking disabled, standard realtime pricing, training
  shards 1–3, simulator, prompts, capacities, three online rounds, and two selected
  nodes per round unchanged. Up to four API requests execute concurrently.
- Use fresh validation shards **13–18** from each dataset at both capacities.
  Validation never enters generation. The test split remains untouched.
- Limit **new spending to $3**: $0.60 for each fixed continuation and $0.075 for
  each controller-transfer run. The previous $0.9424 is not part of this new
  allowance. A budget ceiling may stop a run early; report actual calls/spend.

This is **420 new calls maximum across 19 runs**. A pair matches call and dollar
ceilings, not random model responses. Two replicates per transition are a
diagnostic rather than a significance test. Controllers can legitimately choose
less exploration; early stopping and actual calls remain visible in the report.

The original experiment is never overwritten. The new state records its source
checkpoint hash, imported calls/cost, frozen source copies and trace hashes. Each
continuation's saved 40-call policy is also evaluated on the new validation suite
after search finishes, so its final 100-call policy is compared on the same data.
Historical validation results are removed from the continuation state.

For controller transfer, the report compares final training-selected policies on
fresh held-out data and records actual model calls. A positive revised-minus-
predecessor difference favors the revision. Historical replay gains and all
before/after controller expressions are saved in the frozen plan; they must not
be confused with the new live-search outcomes.

```sh
.venv/bin/python -m dream_rsi.diagnostics \
  --source runs/comparison-long-realtime-20260927/checkpoint.json \
  --config configs/dream-diagnostics.json \
  --output runs/controller-transfer-20260927

# Resume after interruption; completed requests are not resubmitted.
.venv/bin/python -m dream_rsi.diagnostics \
  --output runs/controller-transfer-20260927 --resume
```

The coordinator persists each request and continues until all runs finish or a
local/account error needs attention. Unknown interrupted requests consume their
reserved attempt rather than being retried. `--prepare-only` freezes the plan
and prepares the first wave without sending any requests.

The [prompt audit](prompt-audit-20260927.md) identifies repetition, output-schema
errors, simulator misconceptions, and missing controller-horizon context. Those
are candidates for a separate prompt comparison; changing them here would make
it harder to attribute differences to search length or the controller revision.
