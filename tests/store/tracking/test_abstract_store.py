from unittest import mock

from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.store.tracking.abstract_store import AbstractStore


class AbstractStoreTestImpl(AbstractStore):
    def create_experiment(self, name, artifact_location, tags):
        raise NotImplementedError()

    def get_experiment(self, experiment_id):
        raise NotImplementedError()

    def delete_experiment(self, experiment_id):
        raise NotImplementedError()

    def restore_experiment(self, experiment_id):
        raise NotImplementedError()

    def rename_experiment(self, experiment_id, new_name):
        raise NotImplementedError()

    def get_run(self, run_id):
        raise NotImplementedError()

    def update_run_info(self, run_id, run_status, end_time, run_name):
        raise NotImplementedError()

    def create_run(self, experiment_id, user_id, start_time, tags, run_name):
        raise NotImplementedError()

    def delete_run(self, run_id):
        raise NotImplementedError()

    def restore_run(self, run_id):
        raise NotImplementedError()

    def get_metric_history(self, run_id, metric_key):
        raise NotImplementedError()

    def _search_runs(
        self, experiment_ids, filter_string, run_view_type, max_results, order_by, page_token
    ):
        raise NotImplementedError()

    def log_batch(self, run_id, metrics, params, tags):
        raise NotImplementedError()

    def record_logged_model(self, run_id, mlflow_model):
        raise NotImplementedError()


def test_log_metric():
    run_id = mock.Mock()
    metric = mock.Mock()

    with mock.patch.object(AbstractStoreTestImpl, "log_batch"):
        store = AbstractStoreTestImpl()
        store.log_metric(run_id, metric)
        store.log_batch.assert_called_once_with(run_id, metrics=[metric], params=[], tags=[])


def test_log_param():
    run_id = mock.Mock()
    param = mock.Mock()

    with mock.patch.object(AbstractStoreTestImpl, "log_batch"):
        store = AbstractStoreTestImpl()
        store.log_param(run_id, param)
        store.log_batch.assert_called_once_with(run_id, metrics=[], params=[param], tags=[])


def test_set_tag():
    run_id = mock.Mock()
    tag = mock.Mock()

    with mock.patch.object(AbstractStoreTestImpl, "log_batch"):
        store = AbstractStoreTestImpl()
        store.set_tag(run_id, tag)
        store.log_batch.assert_called_once_with(run_id, metrics=[], params=[], tags=[tag])


def test_search_runs():
    experiment_id = mock.Mock()
    view_type = mock.Mock()
    runs = [mock.Mock(), mock.Mock()]
    token = "adfoiweroh12334kj129318934u"

    with mock.patch.object(AbstractStoreTestImpl, "_search_runs", return_value=(runs, token)):
        store = AbstractStoreTestImpl()
        result = store.search_runs([experiment_id], None, view_type)
        assert list(result) == runs
        assert result.token == token
        store._search_runs.assert_called_once_with(
            [experiment_id], None, view_type, SEARCH_MAX_RESULTS_DEFAULT, None, None
        )
