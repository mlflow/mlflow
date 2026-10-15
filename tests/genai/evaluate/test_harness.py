from mlflow.entities import Feedback
from mlflow.genai import scorer
from mlflow.genai.evaluation.entities import EvalItem
from mlflow.genai.evaluation.harness import _compute_eval_scores
from mlflow.genai.scorers.base import SCORER_BACKEND_DATABRICKS
from mlflow.genai.scorers.scorer_utils import get_scorer_definition_digest
from mlflow.tracing.constant import AssessmentMetadataKey


def test_compute_eval_scores_adds_registered_scorer_metadata():
    @scorer(name="quality_judge")
    def quality_judge(outputs):
        return Feedback(value=True, metadata={"existing": "value"})

    quality_judge._set_registration_metadata(
        backend=SCORER_BACKEND_DATABRICKS,
        experiment_id="123",
        sampling_config=None,
        scorer_version=4,
        canonical_resource_name="experiments/123/scorers/cXVhbGl0eV9qdWRnZQ/versions/4",
        canonical_resource_name_type="databricks_scorer_version",
    )
    eval_item = EvalItem(request_id="request", inputs={}, outputs="output", expectations={})

    result = _compute_eval_scores(eval_item=eval_item, scorers=[quality_judge])

    assert result.assessments[0].metadata == {
        "existing": "value",
        AssessmentMetadataKey.SCORER_DIGEST: get_scorer_definition_digest(quality_judge),
        AssessmentMetadataKey.SCORER_NAME: "quality_judge",
        AssessmentMetadataKey.SCORER_VERSION: "4",
        AssessmentMetadataKey.SCORER_RESOURCE_NAME: (
            "experiments/123/scorers/cXVhbGl0eV9qdWRnZQ/versions/4"
        ),
        AssessmentMetadataKey.SCORER_RESOURCE_NAME_TYPE: "databricks_scorer_version",
    }


def test_compute_eval_scores_adds_registered_scorer_metadata_to_errors():
    @scorer(name="broken_judge")
    def broken_judge(outputs):
        raise RuntimeError("scoring failed")

    broken_judge._set_registration_metadata(
        backend=SCORER_BACKEND_DATABRICKS,
        experiment_id="123",
        sampling_config=None,
        scorer_version=2,
        canonical_resource_name="experiments/123/scorers/YnJva2VuX2p1ZGdl/versions/2",
        canonical_resource_name_type="databricks_scorer_version",
    )
    eval_item = EvalItem(request_id="request", inputs={}, outputs="output", expectations={})

    result = _compute_eval_scores(eval_item=eval_item, scorers=[broken_judge])

    assert result.assessments[0].metadata == {
        AssessmentMetadataKey.SCORER_DIGEST: get_scorer_definition_digest(broken_judge),
        AssessmentMetadataKey.SCORER_NAME: "broken_judge",
        AssessmentMetadataKey.SCORER_VERSION: "2",
        AssessmentMetadataKey.SCORER_RESOURCE_NAME: (
            "experiments/123/scorers/YnJva2VuX2p1ZGdl/versions/2"
        ),
        AssessmentMetadataKey.SCORER_RESOURCE_NAME_TYPE: "databricks_scorer_version",
    }
