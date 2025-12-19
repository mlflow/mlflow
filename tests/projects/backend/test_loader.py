from mlflow.projects.backend import loader


def test_plugin_backend():
    backend = loader.load_backend("dummy-backend")
    assert backend is not None


def test_plugin_does_not_exist():
    backend = loader.load_backend("my_plugin")
    assert backend is None
