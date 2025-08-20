#!/usr/bin/env python
"""
Performance test script for MLflow server.
Logs batches of 500 metrics on 32 parallel workers and 500 traces.
"""

import concurrent.futures
import random
import time

import mlflow
from mlflow import MlflowClient

# Configuration
MLFLOW_TRACKING_URI = "http://localhost:5004"
NUM_WORKERS = 64
METRICS_PER_BATCH = 500
NUM_TRACES = 2000

# Set tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()


def foo(x):
    """Simple function to trace."""
    return x * 2 + random.random()


def log_metrics_batch(worker_id, experiment_id):
    """Log a batch of metrics for a single worker."""
    run_name = f"worker_{worker_id}_run_{int(time.time())}"

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        # Log metrics in batches
        metrics = []
        for i in range(METRICS_PER_BATCH):
            metric_name = f"metric_{worker_id}_{i}"
            metric_value = random.random() * 100
            timestamp = int(time.time() * 1000)
            metrics.append(mlflow.entities.Metric(metric_name, metric_value, timestamp, 0))

        # Log all metrics at once
        client.log_batch(run.info.run_id, metrics=metrics)

        print(f"Worker {worker_id} logged {METRICS_PER_BATCH} metrics to run {run.info.run_id}")
        return run.info.run_id


def log_traces_batch(worker_id, experiment_id, traces_per_worker):
    """Log traces of the foo function for a single worker."""
    print(f"Worker {worker_id} starting to log {traces_per_worker} traces...")

    for i in range(traces_per_worker):
        trace_id = worker_id * traces_per_worker + i
        with mlflow.start_span(name=f"trace_{trace_id}_worker_{worker_id}") as span:
            # Call the traced function
            result = foo(trace_id)
            span.set_attribute("input", trace_id)
            span.set_attribute("result", result)
            span.set_attribute("worker_id", worker_id)

    print(f"Worker {worker_id} completed logging {traces_per_worker} traces")
    return traces_per_worker


def main():
    """Main performance test function."""
    print(f"Starting MLflow performance test against {MLFLOW_TRACKING_URI}")
    print(f"Workers: {NUM_WORKERS}, Metrics per batch: {METRICS_PER_BATCH}, Traces: {NUM_TRACES}")

    # Create or get experiment
    experiment_name = f"perf_test_{int(time.time())}"
    experiment = client.create_experiment(experiment_name)
    print(f"\nCreated experiment: {experiment_name} (ID: {experiment})")

    # Start timing
    start_time = time.time()

    # Log metrics in parallel
    print(f"\nStarting {NUM_WORKERS} parallel workers to log metrics...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for worker_id in range(NUM_WORKERS):
            future = executor.submit(log_metrics_batch, worker_id, experiment)
            futures.append(future)

        # Wait for all workers to complete
        run_ids = []
        for future in concurrent.futures.as_completed(futures):
            try:
                run_id = future.result()
                run_ids.append(run_id)
            except Exception as e:
                print(f"Worker failed with error: {e}")

    metrics_time = time.time() - start_time
    print(f"\nMetrics logging completed in {metrics_time:.2f} seconds")
    print(f"Total metrics logged: {NUM_WORKERS * METRICS_PER_BATCH}")
    print(f"Metrics per second: {(NUM_WORKERS * METRICS_PER_BATCH) / metrics_time:.2f}")

    # Log traces in parallel
    trace_start_time = time.time()
    print(f"\nStarting {NUM_WORKERS} parallel workers to log traces...")

    # Calculate traces per worker
    traces_per_worker = NUM_TRACES // NUM_WORKERS
    remaining_traces = NUM_TRACES % NUM_WORKERS

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for worker_id in range(NUM_WORKERS):
            # Distribute remaining traces to first few workers
            worker_traces = traces_per_worker + (1 if worker_id < remaining_traces else 0)
            if worker_traces > 0:
                future = executor.submit(log_traces_batch, worker_id, experiment, worker_traces)
                futures.append(future)

        # Wait for all workers to complete
        total_traces = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                traces_logged = future.result()
                total_traces += traces_logged
            except Exception as e:
                print(f"Trace worker failed with error: {e}")

    trace_time = time.time() - trace_start_time
    print(f"\nTrace logging completed in {trace_time:.2f} seconds")
    print(f"Total traces logged: {total_traces}")
    print(f"Traces per second: {total_traces / trace_time:.2f}")

    # Total summary
    total_time = time.time() - start_time
    print(f"\n{'=' * 50}")
    print("Performance Test Summary:")
    print(f"{'=' * 50}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total metrics: {NUM_WORKERS * METRICS_PER_BATCH}")
    print(f"Total traces: {total_traces}")
    print(f"Total runs created: {len(run_ids)}")
    print(f"Experiment ID: {experiment}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
