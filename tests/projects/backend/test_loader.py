import entrypoints
from unittest import mock

from mlflow.projects.backend import loader


def test_plugin_backend():
    with mock.patch("entrypoints.get_single") as mock_get_single:
        mock_entry_point = mock.MagicMock(spec=entrypoints.EntryPoint)
        mock_get_single.return_value = mock_entry_point
        loader.load_backend("my_plugin")
        # Check calls to entrypoints
        mock_get_single.assert_called_with(loader.ENTRYPOINT_GROUP_NAME, "my_plugin")
        # Check backend has been built
        mock_entry_point.load.assert_called_once()


def test_plugin_does_not_exist():
    def raise_entrypoint_exception(group, name):
        raise entrypoints.NoSuchEntryPoint(group, name)

    with mock.patch("entrypoints.get_single") as mock_get_single:
        mock_get_single.side_effect = raise_entrypoint_exception
        backend = loader.load_backend("my_plugin")
        assert backend is None
