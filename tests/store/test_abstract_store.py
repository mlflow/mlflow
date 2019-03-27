import mock

from mlflow.store.abstract_store import AbstractStore


def test_log_metric():

    run_id = mock.Mock()
    metric = mock.Mock()

    with mock.patch.object(AbstractStore, 'log_batch'):
        store = AbstractStore()
        store.log_metric(run_id, metric)
        store.log_batch.assert_called_once_with(run_id, metrics=[metric], params=[], tags=[])


def test_log_param():

    run_id = mock.Mock()
    param = mock.Mock()

    with mock.patch.object(AbstractStore, 'log_batch'):
        store = AbstractStore()
        store.log_param(run_id, param)
        store.log_batch.assert_called_once_with(run_id, metrics=[], params=[param], tags=[])


def test_set_tag():

    run_id = mock.Mock()
    tag = mock.Mock()

    with mock.patch.object(AbstractStore, 'log_batch'):
        store = AbstractStore()
        store.set_tag(run_id, tag)
        store.log_batch.assert_called_once_with(run_id, metrics=[], params=[], tags=[tag])
