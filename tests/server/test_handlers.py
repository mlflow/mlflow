from mlflow.server.handlers import get_endpoints, _create_experiment


def test_get_endpoints():
    endpoints = get_endpoints()
    create_experiment_endpoint = [e for e in endpoints if e[1] == _create_experiment]
    assert len(create_experiment_endpoint) == 2
