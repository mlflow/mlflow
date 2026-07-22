"""Benchmark the wall-clock and cost scaling of ``mlflow.genai.evaluate_sweep``.

Uses a fake predict function with a fixed, configurable latency and a trivial
in-process scorer, so the numbers isolate the sweep's orchestration overhead and
scaling behavior from real LLM/judge latency and cost.

It measures three things:

1. Wall-clock scaling in ``n_repeats`` (single config).
2. Wall-clock scaling in the number of configs (fixed repeats).
3. The ``predict_once`` speedup — reusing repeat-0 predictions and only
   re-scoring on later repeats.

Run with::

    uv run python dev/benchmarks/genai_evaluate_sweep/benchmark_sweep.py

Optional flags let you change the dataset size and per-prediction latency::

    uv run python dev/benchmarks/genai_evaluate_sweep/benchmark_sweep.py \
        --n-rows 20 --predict-latency-ms 100
"""

import argparse
import tempfile
import time

import mlflow
from mlflow.genai.scorers import scorer


@scorer
def trivial(outputs) -> float:
    return float(len(str(outputs)))


def make_predict_fn(latency_ms: float):
    def predict_fn(question: str) -> str:
        if latency_ms:
            time.sleep(latency_ms / 1000.0)
        return f"answer to {question}"

    return predict_fn


def _time_sweep(**kwargs) -> float:
    start = time.monotonic()
    mlflow.genai.evaluate_sweep(**kwargs)
    return time.monotonic() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=10)
    parser.add_argument("--predict-latency-ms", type=float, default=50.0)
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///" + tempfile.mkdtemp() + "/mlflow.db")
    mlflow.set_experiment("evaluate_sweep_benchmark")

    data = [{"inputs": {"question": f"q{i}"}} for i in range(args.n_rows)]
    predict_fn = make_predict_fn(args.predict_latency_ms)

    print(f"dataset: {args.n_rows} rows, predict latency: {args.predict_latency_ms:.0f}ms/row\n")

    print("== Scaling in n_repeats (1 config) ==")
    print(f"{'n_repeats':>10} {'wall_s':>10} {'per_cell_s':>12}")
    for n_repeats in (1, 2, 3, 5):
        elapsed = _time_sweep(
            data=data,
            scorers=[trivial],
            predict_fns={"m": predict_fn},
            n_repeats=n_repeats,
        )
        print(f"{n_repeats:>10} {elapsed:>10.2f} {elapsed / n_repeats:>12.2f}")

    print("\n== Scaling in number of configs (n_repeats=3) ==")
    print(f"{'n_configs':>10} {'wall_s':>10} {'per_cell_s':>12}")
    for n_configs in (1, 2, 4):
        fns = {f"m{i}": make_predict_fn(args.predict_latency_ms) for i in range(n_configs)}
        elapsed = _time_sweep(
            data=data,
            scorers=[trivial],
            predict_fns=fns,
            n_repeats=3,
        )
        print(f"{n_configs:>10} {elapsed:>10.2f} {elapsed / (n_configs * 3):>12.2f}")

    print("\n== predict_once speedup (1 config, n_repeats=5) ==")
    print(f"{'mode':>14} {'wall_s':>10}")
    full = _time_sweep(
        data=data,
        scorers=[trivial],
        predict_fns={"m": predict_fn},
        n_repeats=5,
        predict_once=False,
    )
    once = _time_sweep(
        data=data,
        scorers=[trivial],
        predict_fns={"m": predict_fn},
        n_repeats=5,
        predict_once=True,
    )
    print(f"{'predict+score':>14} {full:>10.2f}")
    print(f"{'predict_once':>14} {once:>10.2f}")
    if once:
        print(f"\nspeedup: {full / once:.2f}x")


if __name__ == "__main__":
    main()
