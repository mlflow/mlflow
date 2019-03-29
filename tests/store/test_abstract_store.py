import mock
import pytest

from mlflow.store.abstract_store import AbstractStore


@pytest.fixture
def mock_abstract_methods():
    with mock.patch.object(AbstractStore, "list_experiments"), \
        mock.patch.object(AbstractStore, "create_experiment"), \
        mock.patch.object(AbstractStore, "get_experiment"), \
        mock.patch.object(AbstractStore, "delete_experiment"), \
        mock.patch.object(AbstractStore, "restore_experiment"), \
        mock.patch.object(AbstractStore, "rename_experiment"), \
        mock.patch.object(AbstractStore, "get_run"), \
        mock.patch.object(AbstractStore, "update_run_info"), \
        mock.patch.object(AbstractStore, "create_run"), \
        mock.patch.object(AbstractStore, "delete_run"), \
        mock.patch.object(AbstractStore, "restore_run"), \
        mock.patch.object(AbstractStore, "get_metric_history"), \
        mock.patch.object(AbstractStore, "search_runs"), \
        mock.patch.object(AbstractStore, "list_run_infos"), \
            mock.patch.object(AbstractStore, "log_batch"):
        yield


def test_log_metric(mock_abstract_methods):
    run_id = mock.Mock()
    metric = mock.Mock()

    store = AbstractStore()
    store.log_metric(run_id, metric)
    store.log_batch.assert_called_once_with(
        run_id, metrics=[metric], params=[], tags=[]
    )


def test_log_param(mock_abstract_methods):
    run_id = mock.Mock()
    param = mock.Mock()

    store = AbstractStore()
    store.log_param(run_id, param)
    store.log_batch.assert_called_once_with(
        run_id, metrics=[], params=[param], tags=[]
    )


def test_set_tag(mock_abstract_methods):
    run_id = mock.Mock()
    tag = mock.Mock()

    store = AbstractStore()
    store.set_tag(run_id, tag)
    store.log_batch.assert_called_once_with(
        run_id, metrics=[], params=[], tags=[tag]
    )
