from mlflow.entities import Feedback
from mlflow.genai import scorer
from mlflow.genai.evaluation.entities import EvalItem
from mlflow.genai.evaluation.harness import _compute_eval_scores
from mlflow.tracing.constant import AssessmentMetadataKey


def test_compute_eval_scores_adds_registered_scorer_metadata():
    @scorer(name="quality_judge")
    def quality_judge(outputs):
        return Feedback(value=True, metadata={"existing": "value"})

    quality_judge._scorer_version = 4
    eval_item = EvalItem(request_id="request", inputs={}, outputs="output", expectations={})

    result = _compute_eval_scores(eval_item=eval_item, scorers=[quality_judge])

    assert result.assessments[0].metadata == {
        "existing": "value",
        AssessmentMetadataKey.SCORER_NAME: "quality_judge",
        AssessmentMetadataKey.SCORER_VERSION: "4",
    }


def test_compute_eval_scores_adds_registered_scorer_metadata_to_errors():
    @scorer(name="broken_judge")
    def broken_judge(outputs):
        raise RuntimeError("scoring failed")

    broken_judge._scorer_version = 2
    eval_item = EvalItem(request_id="request", inputs={}, outputs="output", expectations={})

    result = _compute_eval_scores(eval_item=eval_item, scorers=[broken_judge])

    assert result.assessments[0].metadata == {
        AssessmentMetadataKey.SCORER_NAME: "broken_judge",
        AssessmentMetadataKey.SCORER_VERSION: "2",
    }
