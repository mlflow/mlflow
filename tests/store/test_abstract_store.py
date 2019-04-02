import mock

from mlflow.store.abstract_store import AbstractStore


class ConcreteStore(AbstractStore):

    def list_experiments(self, *args, **kwargs):
        pass

    def create_experiment(self, *args, **kwargs):
        pass

    def get_experiment(self, *args, **kwargs):
        pass

    def delete_experiment(self, *args, **kwargs):
        pass

    def restore_experiment(self, *args, **kwargs):
        pass

    def rename_experiment(self, *args, **kwargs):
        pass

    def get_run(self, *args, **kwargs):
        pass

    def update_run_info(self, *args, **kwargs):
        pass

    def create_run(self, *args, **kwargs):
        pass

    def delete_run(self, *args, **kwargs):
        pass

    def restore_run(self, *args, **kwargs):
        pass

    def get_metric_history(self, *args, **kwargs):
        pass

    def search_runs(self, *args, **kwargs):
        pass

    def log_batch(self, *args, **kwargs):
        pass


def test_log_metric():
    run_id = mock.Mock()
    metric = mock.Mock()

    with mock.patch.object(ConcreteStore, "log_batch"):
        store = ConcreteStore()
        store.log_metric(run_id, metric)
        store.log_batch.assert_called_once_with(
            run_id, metrics=[metric], params=[], tags=[]
        )


def test_log_param():
    run_id = mock.Mock()
    param = mock.Mock()

    with mock.patch.object(ConcreteStore, "log_batch"):
        store = ConcreteStore()
        store.log_param(run_id, param)
        store.log_batch.assert_called_once_with(
            run_id, metrics=[], params=[param], tags=[]
        )


def test_set_tag():
    run_id = mock.Mock()
    tag = mock.Mock()

    with mock.patch.object(ConcreteStore, "log_batch"):
        store = ConcreteStore()
        store.set_tag(run_id, tag)
        store.log_batch.assert_called_once_with(
            run_id, metrics=[], params=[], tags=[tag]
        )
