"""
Generate metrics and statistics tables for Monitoring documentation.
"""

from operator import itemgetter
import os
from pathlib import Path

from flask import Flask
from pytablewriter import RstSimpleTableWriter

from mlflow.server import PROMETHEUS_EXPORTER_ENV_VAR
from mlflow.server.prometheus_exporter import activate_prometheus_exporter
from mlflow.server.statistics_collector import MlflowStatisticsCollector


def get_metrics():
    prometheus_metrics_path = os.environ[PROMETHEUS_EXPORTER_ENV_VAR]
    if not os.path.exists(prometheus_metrics_path):
        os.makedirs(prometheus_metrics_path)

    dummy_app = Flask(__name__)
    metrics = activate_prometheus_exporter(dummy_app)

    return list(metrics.registry.collect())


def get_statistics():
    collector = MlflowStatisticsCollector(update_interval_seconds=0)
    collector.register_metrics()

    metrics = []
    for metric in collector.get_registered_metrics():
        for descriptor in metric.describe():
            metrics.append(descriptor)
    return metrics


def create_writer(headers, values):
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
    with filename.open("w") as f:
        values = build_values(metrics)
        writer = create_writer(headers, values)
        f.write(writer.dumps())


def main():
    metrics = get_metrics()
    statistics = get_statistics()
    headers = ["Name", "Type", "Description"]
    docs_build = Path("generated/monitoring")

    docs_build.mkdir(parents=True, exist_ok=True)

    write_table(metrics, headers, docs_build.joinpath("metrics.rst"))
    write_table(statistics, headers, docs_build.joinpath("statistics.rst"))


if __name__ == "__main__":
    main()
