"""
Generate metrics table for Monitoring documentation.
"""

from operator import itemgetter
import os
from pathlib import Path

from flask import Flask
from pytablewriter import RstSimpleTableWriter

from mlflow.server import PROMETHEUS_EXPORTER_ENV_VAR
from mlflow.server.prometheus_exporter import activate_prometheus_exporter


def get_metrics():
    prometheus_metrics_path = os.environ[PROMETHEUS_EXPORTER_ENV_VAR]
    if not os.path.exists(prometheus_metrics_path):
        os.makedirs(prometheus_metrics_path)

    dummy_app = Flask(__name__)
    metrics = activate_prometheus_exporter(dummy_app)

    return list(metrics.registry.collect())


def create_table_writer(headers, values):
    writer = RstSimpleTableWriter()
    writer.headers = headers
    writer.value_matrix = values
    return writer


def inline_code(text):
    return f"``{text}``"


def build_values(metrics):
    values = []
    for metric in metrics:
        values.append((inline_code(metric.name), inline_code(metric.type), metric.documentation))
    return sorted(set(values), key=itemgetter(0))


def write_table(metrics, headers, filename):
    with open(filename, "w") as f:
        values = build_values(metrics)
        writer = create_table_writer(headers, values)
        f.write(writer.dumps())


def main():
    metrics = get_metrics()
    if len(metrics) == 0:
        raise ValueError("No metrics could be exported to documentation.")

    headers = ["Name", "Type", "Description"]
    generated_docs_dir = Path("generated/monitoring")

    generated_docs_dir.mkdir(parents=True, exist_ok=True)
    write_table(metrics, headers, generated_docs_dir.joinpath("metrics.rst"))


if __name__ == "__main__":
    main()
