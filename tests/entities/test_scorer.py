from mlflow.entities.scorer import ScorerVersion
from mlflow.protos.service_pb2 import Scorer as ProtoScorer


def test_from_proto_converts_experiment_id_to_string():
    scorer_version = ScorerVersion.from_proto(
        ProtoScorer(
            experiment_id=123,
            scorer_name="accuracy_scorer",
            scorer_version=1,
            serialized_scorer="serialized_scorer",
            creation_time=1640995200000,
        )
    )

    assert scorer_version.experiment_id == "123"
