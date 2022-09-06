import json
import pytest

from mlflow.entities import RunTag
from mlflow.models import Model
from mlflow.utils.mlflow_tags import MLFLOW_LOGGED_MODELS


class AbstractStoreTest:
    def create_test_run(self):
        raise Exception("this should be overridden")

    def get_store(self):
        raise Exception("this should be overridden")

    def test_record_logged_model(self):
        store = self.get_store()
        run_id = self.create_test_run().info.run_id
        m = Model(artifact_path="model/path", run_id=run_id, flavors={"tf": "flavor body"})
        store.record_logged_model(run_id, m)
        self._verify_logged(
            store,
            run_id=run_id,
            params=[],
            metrics=[],
            tags=[RunTag(MLFLOW_LOGGED_MODELS, json.dumps([m.to_dict()]))],
        )
        m2 = Model(
            artifact_path="some/other/path", run_id=run_id, flavors={"R": {"property": "value"}}
        )
        store.record_logged_model(run_id, m2)
        self._verify_logged(
            store,
            run_id,
            params=[],
            metrics=[],
            tags=[RunTag(MLFLOW_LOGGED_MODELS, json.dumps([m.to_dict(), m2.to_dict()]))],
        )
        m3 = Model(
            artifact_path="some/other/path2", run_id=run_id, flavors={"R2": {"property": "value"}}
        )
        store.record_logged_model(run_id, m3)
        self._verify_logged(
            store,
            run_id,
            params=[],
            metrics=[],
            tags=[
                RunTag(MLFLOW_LOGGED_MODELS, json.dumps([m.to_dict(), m2.to_dict(), m3.to_dict()]))
            ],
        )
        with pytest.raises(
            TypeError,
            match="Argument 'mlflow_model' should be mlflow.models.Model, got '<class 'dict'>'",
        ):
            store.record_logged_model(run_id, m.to_dict())

    @staticmethod
    def _verify_logged(store, run_id, metrics, params, tags):
        run = store.get_run(run_id)
        all_metrics = sum([store.get_metric_history(run_id, key) for key in run.data.metrics], [])
        assert len(all_metrics) == len(metrics)
        logged_metrics = [(m.key, m.value, m.timestamp, m.step) for m in all_metrics]
        assert set(logged_metrics) == {(m.key, m.value, m.timestamp, m.step) for m in metrics}
        logged_tags = set(run.data.tags.items())
        assert {(tag.key, tag.value) for tag in tags} <= logged_tags
        assert len(run.data.params) == len(params)
        assert set(run.data.params.items()) == {(param.key, param.value) for param in params}
