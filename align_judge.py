#!/usr/bin/env python3
"""
Integration test script for judge alignment optimizers (GEPA and SIMBA).

Usage:
    # SIMBA with 50 traces
    python /tmp/align_judge.py --optimizer simba --num-traces 50

    # GEPA with 50 traces
    python /tmp/align_judge.py --optimizer gepa --num-traces 50

    # Dry run (don't register scorer)
    python /tmp/align_judge.py --optimizer simba --num-traces 50 --dry-run

    # Custom experiment
    python /tmp/align_judge.py --optimizer simba --experiment-id 123456789 --num-traces 100
"""

import argparse
import mlflow
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.optimizers import GEPAAlignmentOptimizer, SIMBAAlignmentOptimizer


def main():
    parser = argparse.ArgumentParser(description="Run judge alignment optimization")
    parser.add_argument(
        "--optimizer",
        choices=["gepa", "simba"],
        default="simba",
        help="Optimizer to use (default: simba)",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="databricks://df2",
        help="MLflow tracking URI (default: databricks://df2)",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        required=True,
        help="MLflow experiment ID to fetch traces from",
    )
    parser.add_argument(
        "--num-traces",
        type=int,
        default=50,
        help="Number of traces to use for optimization (default: 50)",
    )
    parser.add_argument(
        "--judge-name",
        type=str,
        default="query_addressed_directly",
        help="Name of the judge/feedback to align (default: query_addressed_directly)",
    )
    parser.add_argument(
        "--optimizer-model",
        type=str,
        default="databricks:/mlflow-gepa-test",
        help="Model for optimizer to use (default: databricks:/mlflow-gepa-test)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't register the optimized scorer",
    )
    # GEPA-specific
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="GEPA: max metric calls (default: 4x num-traces)",
    )
    # SIMBA-specific
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="SIMBA: batch size (default: auto)",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=4,
        help="SIMBA: max demos to include (default: 4)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="SIMBA: max optimization steps (default: 8)",
    )

    args = parser.parse_args()

    # Set up MLflow
    mlflow.set_tracking_uri(args.tracking_uri)
    print(f"Using tracking URI: {args.tracking_uri}")

    if args.experiment_id:
        mlflow.set_experiment(experiment_id=args.experiment_id)

    # Fetch traces in chronological order
    print(f"Fetching up to {args.num_traces} traces (chronological order)...")
    traces_df = mlflow.search_traces(
        experiment_ids=[args.experiment_id],
        max_results=args.num_traces,
        order_by=["timestamp_ms ASC"],
    )
    print(f"Found {len(traces_df)} traces")

    # Convert DataFrame rows to Trace objects
    from mlflow.entities import Trace
    traces = [Trace.from_json(row["trace"]) for _, row in traces_df.iterrows()]

    if len(traces) == 0:
        print("No traces found. Exiting.")
        return

    # Create the base judge
    base_instructions = (
        "Check whether the response in {{outputs}} directly addresses "
        "the query in {{inputs}}. Answer yes/no."
    )
    judge = make_judge(
        name=args.judge_name,
        instructions=base_instructions,
    )

    print(f"\nOriginal judge instructions:\n{judge.instructions}\n")

    # Create optimizer
    if args.optimizer == "gepa":
        optimizer_kwargs = {"model": args.optimizer_model}
        if args.max_calls:
            optimizer_kwargs["max_metric_calls"] = args.max_calls

        optimizer = GEPAAlignmentOptimizer(**optimizer_kwargs)
        print(f"Using GEPA optimizer with model: {args.optimizer_model}")
    else:
        optimizer_kwargs = {"model": args.optimizer_model}
        if args.batch_size:
            optimizer_kwargs["batch_size"] = args.batch_size

        simba_kwargs = {}
        if args.max_demos:
            simba_kwargs["max_demos"] = args.max_demos
        if args.max_steps:
            simba_kwargs["max_steps"] = args.max_steps
        if simba_kwargs:
            optimizer_kwargs["simba_kwargs"] = simba_kwargs

        optimizer = SIMBAAlignmentOptimizer(**optimizer_kwargs)
        print(f"Using SIMBA optimizer with model: {args.optimizer_model}")

    # Run optimization
    print(f"\nRunning alignment optimization with {len(traces)} traces...")
    optimized_judge = optimizer.align(judge, traces)

    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nOptimized judge instructions:\n{optimized_judge.instructions}\n")

    # Check if instructions changed
    if optimized_judge.instructions == judge.instructions:
        print("NOTE: Instructions unchanged (SIMBA adds demos, not instruction changes)")
    else:
        print("Instructions were modified by optimization")

    # Check template variables
    has_inputs = "{{inputs}}" in optimized_judge.instructions
    has_outputs = "{{outputs}}" in optimized_judge.instructions
    print(f"Template variables preserved: inputs={has_inputs}, outputs={has_outputs}")

    # Register if not dry run
    if not args.dry_run:
        scorer_name = f"{args.judge_name}_{args.optimizer}_optimized"
        print(f"\nRegistering scorer as: {scorer_name}")
        # Note: Registration would happen here
        # mlflow.genai.register_scorer(optimized_judge, name=scorer_name)
        print("(Registration not implemented in this script)")
    else:
        print("\nDry run - not registering scorer")


if __name__ == "__main__":
    main()
