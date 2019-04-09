import mock

from mlflow.store.abstract_store import AbstractStore
from mlflow.entities import ViewType


class AbstractStoreTestImpl(AbstractStore):

    def list_experiments(self, view_type=ViewType.ACTIVE_ONLY):
        raise NotImplementedError()

    def create_experiment(self, name, artifact_location):
        raise NotImplementedError()

    def get_experiment(self, experiment_id):
        raise NotImplementedError()

    def delete_experiment(self, experiment_id):
        raise NotImplementedError()

    def restore_experiment(self, experiment_id):
        raise NotImplementedError()

    def rename_experiment(self, experiment_id, new_name):
        raise NotImplementedError()

    def get_run(self, run_uuid):
        raise NotImplementedError()

    def update_run_info(self, run_uuid, run_status, end_time):
        raise NotImplementedError()

    def create_run(self, experiment_id, user_id, run_name, source_type, source_name,
                   entry_point_name, start_time, source_version, tags, parent_run_id):
        raise NotImplementedError()

    def delete_run(self, run_id):
        raise NotImplementedError()

    def restore_run(self, run_id):
        raise NotImplementedError()

    def get_metric_history(self, run_uuid, metric_key):
        raise NotImplementedError()

    def search_runs(self, experiment_ids, search_filter, run_view_type):
        raise NotImplementedError()

    def log_batch(self, run_id, metrics, params, tags):
        raise NotImplementedError()


def test_get_experiment_by_name():
    experiments = [mock.Mock(), mock.Mock(), mock.Mock()]
    # Configure name after mock creation as name is a reserved argument to Mock()
    experiments[1].configure_mock(name="my experiment")
    with mock.patch.object(AbstractStoreTestImpl, "list_experiments", return_value=experiments):
        store = AbstractStoreTestImpl()
        assert store.get_experiment_by_name("my experiment") == experiments[1]
        store.list_experiments.assert_called_once_with(ViewType.ALL)


def test_get_experiment_by_name_missing():
    with mock.patch.object(AbstractStoreTestImpl, "list_experiments", return_value=[]):
        store = AbstractStoreTestImpl()
        assert store.get_experiment_by_name("my experiment") is None
        store.list_experiments.assert_called_once_with(ViewType.ALL)


def test_log_metric():
    run_id = mock.Mock()
    metric = mock.Mock()

    with mock.patch.object(AbstractStoreTestImpl, "log_batch"):
        store = AbstractStoreTestImpl()
        store.log_metric(run_id, metric)
        store.log_batch.assert_called_once_with(
            run_id, metrics=[metric], params=[], tags=[]
        )


def test_log_param():
    run_id = mock.Mock()
    param = mock.Mock()

    with mock.patch.object(AbstractStoreTestImpl, "log_batch"):
        store = AbstractStoreTestImpl()
        store.log_param(run_id, param)
        store.log_batch.assert_called_once_with(
            run_id, metrics=[], params=[param], tags=[]
        )


def test_set_tag():
    run_id = mock.Mock()
    tag = mock.Mock()

    with mock.patch.object(AbstractStoreTestImpl, "log_batch"):
        store = AbstractStoreTestImpl()
        store.set_tag(run_id, tag)
        store.log_batch.assert_called_once_with(
            run_id, metrics=[], params=[], tags=[tag]
        )


def test_list_run_infos():
    experiment_id = mock.Mock()
    view_type = mock.Mock()
    run_infos = [mock.Mock(), mock.Mock()]
    runs = [mock.Mock(info=info) for info in run_infos]

    with mock.patch.object(AbstractStoreTestImpl, "search_runs", return_value=runs):
        store = AbstractStoreTestImpl()
        assert store.list_run_infos(experiment_id, view_type) == run_infos
        store.search_runs.assert_called_once_with([experiment_id], None, view_type)
