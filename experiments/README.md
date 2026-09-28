# Experiment conventions

Create `experiments/<task-name>/` for each independent research task. Keep its README, dependency manifest/lockfile, evaluator, search adaptation, configurations, setup scripts, tests and protocol documentation together. Use separate named runs when comparing settings within one task.

Each README must state the hypothesis, scoring units, baseline, split boundaries, run commands, model and budget settings, and known adaptations to the paper. Keep generated datasets in ignored `data/` and live checkpoints in ignored `runs/`. Store reviewed public results in `experiment-results/<run-name>/`, including selected programs, configurations, usage, validation, provenance and limitations. Never commit credentials or raw licensed datasets. Preserve historical source snapshots and hashes when code changes.

Add each task to the repository README. Dependencies and commands are scoped to that task's directory. Extract shared code only when another experiment establishes a concrete common interface; avoid implicit cross-experiment imports.

## Prefix-cache migration

The original root project now lives in `prefix-cache/`. Python module names and relative commands are unchanged after `cd experiments/prefix-cache`. Existing local datasets and runs moved with it. The local `.venv` and `.env` symlinks reuse the original ignored root environment and credential file; fresh clones create their own experiment-local environment and supply `XIAOMI_API` locally.

Historical checkpoints retain their original bytes and absolute paths for provenance. Do not blindly resume them after relocation. The latest run is completed; for new searches regenerate configuration paths from the new experiment directory. Public historical archive paths are interpreted relative to this experiment, unless their manifest identifies an original historical path.
