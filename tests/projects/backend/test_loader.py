import importlib.metadata
from unittest import mock

from mlflow.projects.backend import loader


def test_plugin_backend():
    with mock.patch(
        "mlflow.utils.plugins._get_entry_points",
        return_value=[mock.MagicMock(spec=importlib.metadata.EntryPoint)],
    ) as mock_get_single:
        loader.load_backend("my_plugin")
        mock_get_single.assert_called_once()


def test_plugin_does_not_exist():
    backend = loader.load_backend("my_plugin")
    assert backend is None
