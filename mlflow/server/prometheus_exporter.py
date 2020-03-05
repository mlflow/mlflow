from prometheus_flask_exporter.multiprocess import GunicornInternalPrometheusMetrics
from flask import request


def activate_prometheus_exporter(app):
    metrics = GunicornInternalPrometheusMetrics(app, export_defaults=False)

    endpoint = app.view_functions
    histogram = metrics.histogram('mlflow_requests_by_status_and_path',
                                  'Request latencies and count by status and path',
                                  labels={'status': lambda r: r.status_code,
                                          'path': lambda: change_path_for_metric(request.path)})
    for func_name, func in endpoint.items():
        if func_name in ["_search_runs", "_log_metric", "_log_param", "_set_tag", "_create_run"]:
            app.view_functions[func_name] = histogram(func)

    return app


def change_path_for_metric(path):
    """
    Replace the '/' in the metric path by '_' so grafana can correctly use it.
    :param path: path of the metric (example: runs/search)
    :return: path with '_' instead of '/'
    """
    if 'mlflow/' in path:
        path = path.split('mlflow/')[-1]
    return path.replace('/', '_')
