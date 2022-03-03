import os
import time
import uuid

import click
import psycopg2
import pandas as pd
import tabulate

import mlflow
from mlflow.tracking._tracking_service.utils import _TRACKING_URI_ENV_VAR


tabulate.PRESERVE_WHITESPACE = True


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def generate_runs(num_experiments, num_runs, num_params, num_metrics):
    for i in range(num_experiments):
        if i != 0:
            mlflow.set_experiment(f"experiment_{i}")
        for i in range(num_runs):
            print(i)
            with mlflow.start_run() as run:
                for indices in chunks(range(num_params), 100):
                    mlflow.log_params({f"p_{uuid.uuid4().hex}": idx for idx in indices})
                for indices in chunks(range(num_metrics), 100):
                    mlflow.log_metrics({f"m_{idx}": idx for idx in indices})


def run_query(query, show_result=True):
    with psycopg2.connect(os.getenv(_TRACKING_URI_ENV_VAR)) as conn:
        cursor = conn.cursor()
        before = time.time()
        cursor.execute(query)
        exec_time = time.time() - before
        max_line_length = max(map(len, query.split("\n")))
        print("\nQuery:")
        print("=" * max_line_length)
        print(query)
        print("=" * max_line_length)
        print("\nResult:")
        if show_result:
            rows = cursor.fetchall()
            columns = [i[0] for i in cursor.description]
            print(pd.DataFrame(rows, columns=columns).head(100).to_markdown(index=False))
        print("\nTook:", exec_time, "seconds")
        return exec_time


SHOW_UNINDEXED_FOREIGN_KEYS = """
SELECT c.conrelid::regclass AS "table",
    /* list of key column names in order */
    string_agg(a.attname, ',' ORDER BY x.n) AS columns,
    pg_catalog.pg_size_pretty(
        pg_catalog.pg_relation_size(c.conrelid)
    ) AS size,
    c.conname AS constraint,
    c.confrelid::regclass AS referenced_table
FROM pg_catalog.pg_constraint c
/* enumerated key column numbers per foreign key */
CROSS JOIN LATERAL
    unnest(c.conkey) WITH ORDINALITY AS x(attnum, n)
/* name for each key column */
JOIN pg_catalog.pg_attribute a
    ON a.attnum = x.attnum
        AND a.attrelid = c.conrelid
WHERE NOT EXISTS
    /* is there a matching index for the constraint? */
    (SELECT 1 FROM pg_catalog.pg_index i
        WHERE i.indrelid = c.conrelid
        /* the first index columns must be the same as the
            key columns, but order doesn't matter */
        AND (i.indkey::smallint[])[0:cardinality(c.conkey)-1]
            OPERATOR(pg_catalog.@>) c.conkey)
AND c.contype = 'f'
GROUP BY c.conrelid, c.conname, c.confrelid
ORDER BY pg_catalog.pg_relation_size(c.conrelid) DESC;
"""


@click.command()
@click.option("--query", type=click.STRING, help="Query to run")
@click.option(
    "--num-experiments", type=click.INT, default=3, help="Number of experiments to create"
)
@click.option("--num-runs", type=click.INT, default=100, help="Number of runs to create")
@click.option(
    "--num-params", type=click.INT, default=5, help="Number of paramters to create per run"
)
@click.option(
    "--num-metrics", type=click.INT, default=5, help="Number of metrics to create per run"
)
def main(query, num_experiments, num_runs, num_params, num_metrics):
    generate_runs(num_experiments, num_runs, num_params, num_metrics)
    show_result = True
    run_query(query, show_result)
    run_query(SHOW_UNINDEXED_FOREIGN_KEYS)
    print("Number of experiments:", num_experiments)
    print("Number of runs:", num_runs)
    print("Number of parameters:", num_params)
    print("Number of metrics:", num_metrics)
    client = mlflow.tracking.MlflowClient()
    run_id = client.list_run_infos(experiment_id="0")[0].run_id
    before = time.time()
    client.get_metric_history(run_id, "m_0")
    print(time.time() - before)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
