# `evaluate_sweep` performance notes

`mlflow.genai.evaluate_sweep` runs `mlflow.genai.evaluate` once per
`(config, repeat)` cell over a grid of `len(predict_fns) x n_repeats` cells.
This directory benchmarks how its wall-clock and cost scale.

## Cost model

For a sweep with `C` configs, `R` repeats, `N` rows, and `S` scorers:

- **LLM/judge calls** ≈ `C x R x N x (1 predict + S scorer)` calls.
  Every axis multiplies. `n_repeats` is the axis you add for confidence
  intervals, so a CI costs `R x` the calls of a single eval.
- **Wall-clock** ≈ `sum over cells of (predict + score time)`. Cells run
  **sequentially** (see below), so wall-clock is the sum of per-cell times, not
  the max. Within a cell, rows are parallelized by the existing harness
  (hundreds of threads), so a single cell already saturates the LLM rate-limit
  budget.

`predict_once=True` drops the predict term on repeats `2..R`: predictions run
once, and later repeats only re-run scorers. This trades away the model's own
generation variance (the CI then reflects scorer/judge variance only) for
roughly a `predict_fraction`-proportional speedup.

## Why cells run sequentially

Cross-cell concurrency is intentionally not implemented. MLflow's fluent
active-run stack and autologging are process-global, so concurrent nested runs
in threads would corrupt each other's run/trace association. Because the harness
already parallelizes rows within a single cell up to hundreds of threads, one
cell saturates the LLM budget on its own — running cells sequentially with a
shared, sequential rate-limit budget is near-optimal and avoids the 429 storms
that uncoordinated per-cell rate limiters would cause. Subprocess isolation per
cell is possible future work if profiling ever shows a gain.

## Representative numbers

Fake predict function with fixed 50ms/row latency, trivial in-process scorer,
10-row dataset, sqlite backend, on a developer laptop:

```
== Scaling in n_repeats (1 config) ==
 n_repeats     wall_s   per_cell_s
         1       1.20         1.20
         2       1.03         0.52
         3       1.41         0.47
         5       2.44         0.49

== Scaling in number of configs (n_repeats=3) ==
 n_configs     wall_s   per_cell_s
         1       1.31         0.44
         2       2.96         0.49
         4       5.59         0.47

== predict_once speedup (1 config, n_repeats=5) ==
          mode     wall_s
 predict+score       2.36
  predict_once       1.29

speedup: 1.83x
```

Per-cell time is roughly constant, so total wall-clock grows linearly in
`n_repeats x n_configs`, matching the cost model. The `predict_once` speedup
approaches `1 / (1 - predict_fraction)` — here predictions are about half of
per-cell time, giving ~1.8x.

## Running

```bash
uv run python dev/benchmarks/genai_evaluate_sweep/benchmark_sweep.py
uv run python dev/benchmarks/genai_evaluate_sweep/benchmark_sweep.py \
    --n-rows 20 --predict-latency-ms 100
```
