from prometheus_flask_exporter.multiprocess import GunicornInternalPrometheusMetrics
from flask import request

from mlflow.version import VERSION


def activate_prometheus_exporter(app):
    def mlflow_version(req: request):  # pylint: disable=unused-argument
        return VERSION

    metrics = GunicornInternalPrometheusMetrics(
        app,
        export_defaults=True,
        defaults_prefix="mlflow",
        excluded_paths=["/health"],
        group_by=mlflow_version,
    )

    return metrics
