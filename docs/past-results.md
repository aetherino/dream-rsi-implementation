# Archived experiments and reproduction

The tracked [historical archive](../experiment-results/historical/) preserves selected evidence from three completed September 27, 2026 studies. The ignored `runs/` directories retain local execution state; they are not required to read these results. This archive contains aggregate reports, frozen plans/configurations, training-selected policy expressions, initial/final controllers and revision records, per-scenario validation and baseline metrics, usage totals, and source-code/trace hashes. Manifests record SHA-256 of both original files and the sanitized public copies. Source hashes establish provenance; they do not imply current source is identical to the historical implementation.

| Study | Calls | Estimated API cost | Validation finding |
| --- | ---: | ---: | --- |
| Corrected objective, `comparison-total-v2-20260927` | 48 | $0.150068475 | Fixed mean 3.79965%; adaptive mean 11.19747% extra-compute reduction versus LRU; zero accepted controller revisions |
| Longer comparison, `comparison-long-realtime-20260927` | 240 | $0.94238487 | Fixed mean 22.74961%; adaptive mean 13.38561%; four accepted historical replay revisions |
| Diagnostic continuation, `prompt-v2-continuation-20260927` | 410 new | $2.37055338 new | 19/19 runs completed; three 100-call fixed policies mean 26.72187% on fresh validation |

Dollar amounts are token-rate estimates, not provider invoices. The diagnostic study imported three 40-call fixed histories: its 410 **new** calls include their 180 continuation calls and 230 prospective controller-test calls. Imported costs are excluded from its new-spend figure. Ceilings were 420 new calls and $3; realized requests differ between controller arms.

The final fixed-policy mean is an arithmetic mean of three suite-level scores. Its denominator is LRU's avoidable extra computation above the unlimited-cache reference, rather than all prefill computation. Recalculating from archived per-scenario metrics gives a mean **4.01158% reduction in total computed prompt tokens**, and token-weighted suite hit ratios of **80.76428% LRU → 81.53594% policy**, averaged across those three trials. These are CPU trace-replay metrics, not measured GPU latency gains.

On the same fresh validation suite, extending fixed searches from 40 to 100 calls changed their scores by −1.22, +3.31 and +0.03 percentage points. For prospective controller tests, the mean revised-minus-predecessor score differences were +15.82, −4.06, −2.75 and −6.17 percentage points for transitions 1–4. Only the first transition improved in both replicates. Two replicates per transition are diagnostic evidence, not a significance test or a general claim of self-improvement.

The continuation mixes v1 and v2 prompt histories. Prompt revision happened after 219 new requests and $0.709812555 estimated new spending; per-run first-v2 call numbers and checkpoint hashes are in the archived plan. It is not a controlled prompt A/B experiment and cannot identify a causal prompt effect. Training shards 1–3 selected policies; validation shards 7–12 served the 240-call study, and fresh validation shards 13–18 served the diagnostic study. These validation observations are now public. They are not an unseen hidden test; the test split was not used for these findings.

## Environment and data

From the repository root, use Python 3.11+ and `uv`:

```sh
uv sync --locked
uv run --locked python scripts/setup_datasets.py
uv run --locked python scripts/setup_datasets.py --verify-only
uv run --locked python scripts/tokenize_datasets.py
uv run --locked python scripts/tokenize_datasets.py --verify-only
uv run --locked python -m unittest discover -s tests -v
```

Setup downloads ShareGPT and MASH-QA into ignored `data/raw/`; tokenization uses the pinned Qwen2.5 tokenizer locally and writes replay shards under `data/tokenized/qwen2.5/`. No model weights or paid API requests are needed for those steps. See [dataset preparation](dataset-preparation.md) for source attribution, checksums and preprocessing. Compare regenerated trace hashes with the archived configuration before claiming an exact workload reproduction.

For paid searches, set `XIAOMI_API` in your environment or a local ignored `.env`. Keep credentials private. The historical backend was regular `mimo`, model `mimo-v2.6-pro`, thinking disabled; saved configs record historical rates and limits. Provider availability, pricing and sampled responses may change. API generation is nondeterministic, so rerunning reproduces the protocol, not necessarily the same policies or scores.

## Recreate the protocols

These commands use fresh output directories. `mock` checks orchestration without paid generation; replace it with `mimo` only when intentionally starting a paid study.

```sh
uv run --locked python -m dream_rsi.compare \
  --config configs/dream-comparison-total.json --backend mock
uv run --locked python -m dream_rsi.batch_compare \
  --config configs/dream-comparison-long.json --backend mock \
  --output runs/reproduction-long --watch
```

The checked-in total-objective config and long config provide the respective protocols. The archived per-run configs freeze historical values and paths normalized relative to the repository. Current defaults may differ, including prompt version; consult the archived plans to match historical settings.

The diagnostic protocol requires the **local full 240-call checkpoint**, since it continues complete histories and extracts accepted controller transitions:

```sh
uv run --locked python -m dream_rsi.diagnostics \
  --source runs/reproduction-long/checkpoint.json \
  --config configs/dream-diagnostics.json \
  --output runs/reproduction-diagnostics --prepare-only
```

Omit `--prepare-only` to execute the configured backend after inspecting the frozen plan. See [controller-transfer protocol](controller-transfer-study.md) and [prompt migration](prompt-v2-continuation.md) for the historical prompt boundary. An exact historical resume requires the original ignored checkpoint; the public metrics archive intentionally is not a resumable checkpoint. Migration requires a settled, not-yet-validated checkpoint; do not migrate completed archived results or reset budgets. Resume an interrupted local study using its existing output directory with `--resume`.
