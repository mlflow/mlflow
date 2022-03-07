import os
import time

import click
import pandas as pd
import psycopg2
import tabulate

import mlflow
from mlflow.tracking._tracking_service.utils import _TRACKING_URI_ENV_VAR
from mlflow.utils.validation import MAX_PARAMS_TAGS_PER_BATCH

tabulate.PRESERVE_WHITESPACE = True


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def print_dataframe(df):
    print(df.to_markdown() + "\n")


def benchmark():
    calls = [
        (
            mlflow.search_runs,
            {"experiment_ids": ["1"]},
        ),
        (
            mlflow.search_runs,
            {
                "experiment_ids": ["1"],
                "filter_string": "metrics.p_0 > 0.5",
            },
        ),
        (
            mlflow.search_runs,
            {
                "experiment_ids": ["1"],
                "filter_string": "metrics.m_0 > 0.5",
            },
        ),
    ]

    results = []
    for func, kwargs in calls:
        before = time.time()
        func(**kwargs)
        elapsed_time_in_sec = time.time() - before
        results.append(
            (
                func.__name__,
                "\n".join(f"{k} = {v!r}" for k, v in kwargs.items()),
                elapsed_time_in_sec,
            )
        )

    print_dataframe(
        pd.DataFrame(
            results,
            columns=[
                "api",
                "arguments",
                "elapsed_time_in_seconds",
            ],
        )
    )


def generate_data(num_experiments, num_runs, num_params, num_metrics):
    for idx_exp in range(num_experiments):
        mlflow.set_experiment(f"experiment_{idx_exp}")
        for _ in range(num_runs):
            with mlflow.start_run():
                for indices in chunks(range(num_params), MAX_PARAMS_TAGS_PER_BATCH):
                    mlflow.log_params({f"p_{idx}": idx for idx in indices})
                for indices in chunks(range(num_metrics), MAX_PARAMS_TAGS_PER_BATCH):
                    mlflow.log_metrics({f"m_{idx}": idx for idx in indices})
        if (idx_exp + 1) % int(num_experiments / 10) == 0:
            show_tables()
            join_params_and_metrics_on_runs()
            benchmark()
            print("=" * 80 + "\n")


def show_tables():
    # https://stackoverflow.com/a/2611745/6943581
    query = """
WITH tbl1 AS
  (SELECT table_schema,
          table_name
   FROM information_schema.tables
   WHERE table_name not like 'pg_%'
     AND table_schema in ('public')),
  tbl2 AS
    (SELECT table_name,
       (xpath('/row/c/text()', query_to_xml(format('select count(*) as c from %I.%I', table_schema, TABLE_NAME), FALSE, TRUE, '')))[1]::text::int AS num_rows,
       pg_table_size(quote_ident(table_name)) AS size
    FROM tbl1)

SELECT *
FROM tbl2
WHERE num_rows > 0
ORDER BY num_rows DESC;
"""
    with psycopg2.connect(os.getenv(_TRACKING_URI_ENV_VAR)) as conn:
        print_dataframe(pd.read_sql(query, conn))


def join_params_and_metrics_on_runs():
    query = """
EXPLAIN ANALYZE

SELECT *
FROM runs
JOIN params on params.run_uuid = runs.run_uuid
JOIN metrics on metrics.run_uuid = params.run_uuid
WHERE runs.experiment_id = 0
"""
    with psycopg2.connect(os.getenv(_TRACKING_URI_ENV_VAR)) as conn:
        print_dataframe(pd.read_sql(query, conn))


@click.command()
@click.option(
    "--num-experiments", type=click.INT, default=5, help="Number of experiments to create"
)
@click.option(
    "--num-runs", type=click.INT, default=5, help="Number of runs to create per experiment"
)
@click.option(
    "--num-params", type=click.INT, default=5, help="Number of paramters to create per run"
)
@click.option(
    "--num-metrics", type=click.INT, default=5, help="Number of metrics to create per run"
)
def main(num_experiments, num_runs, num_params, num_metrics):
    generate_data(num_experiments, num_runs, num_params, num_metrics)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
