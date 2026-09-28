# Historical result archive

See [past results and reproduction](../../docs/past-results.md) for interpretation and rerun instructions. Each study contains `comparison.json`, `report.md`, `plan.json`, per-run evidence, and `manifest.json`.

Copies omit prompts, raw responses, rationale text, detailed usage records and per-request details. Repository-root paths are relative; remaining home-directory paths are redacted. Source hashes in manifests refer to original ignored run files; archive hashes refer to the transformed copies. Trace content, token IDs, credentials, provider job IDs and giant execution checkpoints are not redistributed.

A validation score is `(LRU extra computed tokens − policy extra computed tokens) / LRU extra computed tokens`, summed across scenarios. Unlimited-cache computation defines the unavoidable reference. Timing measurements depend on local CPU and instrumentation. Usage totals are estimated token-rate accounting, not invoices.

The archive preserves evidence rather than runnable coordinator state. Full source snapshots and ignored checkpoints remain local. Use the frozen configs and trace/source hashes to assess reproduction; do not treat later source changes or stochastic API responses as an exact historical rerun.
