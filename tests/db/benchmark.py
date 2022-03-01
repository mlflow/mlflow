import os
import time

import click
import mlflow
from mlflow.tracking._tracking_service.utils import _TRACKING_URI_ENV_VAR

import psycopg2


def generate_runs(num_runs):
    print(f"Creating {num_runs} runs...")
    for _ in range(num_runs):
        with mlflow.start_run():
            mlflow.log_params({f"p_{idx}": idx for idx in range(5)})
            mlflow.log_metrics({f"m_{idx}": idx for idx in range(5)})


def run_query(query):
    with psycopg2.connect(os.getenv(_TRACKING_URI_ENV_VAR)) as conn:
        cursor = conn.cursor()
        before = time.time()
        cursor.execute(query)
        return time.time() - before


@click.command()
@click.option("--query", type=click.STRING, help="Query to run")
@click.option("--num-runs", type=click.INT, default=100, help="Number of runs to create")
def main(query, num_runs):
    generate_runs(num_runs)
    exec_time = run_query(query)
    print("Number of runs:", num_runs)
    print("Query:")
    max_line_length = max(map(len, query.split("\n")))
    print("=" * max_line_length)
    print(query)
    print("=" * max_line_length)
    print("Execution time in seconds:", exec_time)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
