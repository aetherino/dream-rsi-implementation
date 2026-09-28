# Dream-RSI experiments

A repository for experiments with Dream-RSI program search and learned exploration controllers. Each task owns its implementation, dependencies, setup, tests, local data and published results.

| Experiment | Task | Latest result |
| --- | --- | --- |
| [Prefix cache](experiments/prefix-cache/README.md) | Search executable cache eviction policies using MiMo and CPU trace replay | Dream: 2.61% less extra recomputation than LRU on development validation; fixed: 0%; one $9.41 comparison |

## Repository layout

```text
experiments/
  prefix-cache/
    cache_sim/          # CPU evaluator
    dream_rsi/          # task-adapted search and controller implementation
    configs/ scripts/   # experiment configuration and data preparation
    tests/ docs/        # verification and protocol documentation
    experiment-results/ # published archives, including the completed $10 run
    pyproject.toml      # experiment dependencies
    uv.lock
    data/ runs/         # local only; ignored by Git
```

The current search implementation is task-adapted and stays inside the prefix-cache experiment. A generic cross-task API has not yet been extracted; future tasks should not import cache-specific scoring or prompts accidentally.

## Work on an experiment

```sh
cd experiments/prefix-cache
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```

All commands in an experiment's documentation run from that experiment directory. Its README explains dataset setup and paid search configuration. No search is started by repository setup.

See [experiment conventions](experiments/README.md) before adding another task, and the [completed run archive](experiments/prefix-cache/experiment-results/paper-section3-20260928/README.md) for the latest evidence.
