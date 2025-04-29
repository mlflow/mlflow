from mlflow.entities import Metric
from mlflow.evaluation.assessment import AssessmentEntity, AssessmentSource
from mlflow.evaluation.evaluation import EvaluationEntity
from mlflow.evaluation.evaluation_tag import EvaluationTag
from mlflow.evaluation.utils import evaluations_to_dataframes


def test_evaluations_to_dataframes_basic():
    # Setup an evaluation with minimal data
    evaluation = EvaluationEntity(
        evaluation_id="eval1",
        run_id="run1",
        inputs_id="inputs1",
        inputs={"feature1": 1.0, "feature2": 2.0},
    )

    evaluations_df, metrics_df, assessments_df, tags_df = evaluations_to_dataframes([evaluation])

    # Check the evaluations DataFrame
    assert len(evaluations_df) == 1
    assert evaluations_df["evaluation_id"].iloc[0] == "eval1"
    assert evaluations_df["run_id"].iloc[0] == "run1"
    assert evaluations_df["inputs_id"].iloc[0] == "inputs1"
    assert evaluations_df["inputs"].iloc[0] == {"feature1": 1.0, "feature2": 2.0}

    # Check that the other DataFrames are empty
    assert metrics_df.empty
    assert assessments_df.empty
    assert tags_df.empty


def test_evaluations_to_dataframes_full_data():
    # Setup an evaluation with full data
    source = AssessmentSource(source_type="HUMAN", source_id="user_1")
    assessment = AssessmentEntity(
        evaluation_id="eval1",
        name="accuracy",
        source=source,
        timestamp=123456789,
        numeric_value=0.95,
        rationale="Good performance",
    )
    metric = Metric(key="metric1", value=0.9, timestamp=1234567890, step=0)
    tag = EvaluationTag(key="tag1", value="value1")

    evaluation = EvaluationEntity(
        evaluation_id="eval1",
        run_id="run1",
        inputs_id="inputs1",
        inputs={"feature1": 1.0, "feature2": 2.0},
        outputs={"output1": 0.5},
        request_id="request1",
        targets={"target1": 0.6},
        error_code="E001",
        error_message="An error occurred",
        assessments=[assessment],
        metrics=[metric],
        tags=[tag],
    )

    evaluations_df, metrics_df, assessments_df, tags_df = evaluations_to_dataframes([evaluation])

    # Check the evaluations DataFrame
    assert len(evaluations_df) == 1
    assert evaluations_df["evaluation_id"].iloc[0] == "eval1"
    assert evaluations_df["run_id"].iloc[0] == "run1"
    assert evaluations_df["inputs_id"].iloc[0] == "inputs1"
    assert evaluations_df["inputs"].iloc[0] == {"feature1": 1.0, "feature2": 2.0}
    assert evaluations_df["outputs"].iloc[0] == {"output1": 0.5}
    assert evaluations_df["request_id"].iloc[0] == "request1"
    assert evaluations_df["targets"].iloc[0] == {"target1": 0.6}
    assert evaluations_df["error_code"].iloc[0] == "E001"
    assert evaluations_df["error_message"].iloc[0] == "An error occurred"

    # Check the metrics DataFrame
    assert len(metrics_df) == 1
    assert metrics_df["evaluation_id"].iloc[0] == "eval1"
    assert metrics_df["key"].iloc[0] == "metric1"
    assert metrics_df["value"].iloc[0] == 0.9
    assert metrics_df["timestamp"].iloc[0] == 1234567890

    # Check the assessments DataFrame
    assert len(assessments_df) == 1
    assert assessments_df["evaluation_id"].iloc[0] == "eval1"
    assert assessments_df["name"].iloc[0] == "accuracy"
    assert assessments_df["source"].iloc[0] == source.to_dictionary()
    assert assessments_df["boolean_value"].iloc[0] is None
    assert assessments_df["numeric_value"].iloc[0] == 0.95
    assert assessments_df["string_value"].iloc[0] is None
    assert assessments_df["rationale"].iloc[0] == "Good performance"
    assert assessments_df["error_code"].iloc[0] is None
    assert assessments_df["error_message"].iloc[0] is None

    # Check the tags DataFrame
    assert len(tags_df) == 1
    assert tags_df["evaluation_id"].iloc[0] == "eval1"
    assert tags_df["key"].iloc[0] == "tag1"
    assert tags_df["value"].iloc[0] == "value1"


def test_evaluations_to_dataframes_empty():
    # Empty evaluations list
    evaluations_df, metrics_df, assessments_df, tags_df = evaluations_to_dataframes([])

    # Verify that the DataFrames are empty
    assert evaluations_df.empty
    assert metrics_df.empty
    assert assessments_df.empty
    assert tags_df.empty

    # Verify the column names of the empty DataFrames
    expected_evaluation_columns = [
        "evaluation_id",
        "run_id",
        "inputs_id",
        "inputs",
        "outputs",
        "request_id",
        "targets",
        "error_code",
        "error_message",
    ]
    expected_metrics_columns = [
        "evaluation_id",
        "key",
        "value",
        "timestamp",
        "model_id",
        "dataset_name",
        "dataset_digest",
        "run_id",
    ]
    expected_assessments_columns = [
        "evaluation_id",
        "name",
        "source",
        "timestamp",
        "boolean_value",
        "numeric_value",
        "string_value",
        "rationale",
        "metadata",
        "error_code",
        "error_message",
        "span_id",
    ]
    expected_tags_columns = ["evaluation_id", "key", "value"]

    assert list(evaluations_df.columns) == expected_evaluation_columns
    assert list(metrics_df.columns) == expected_metrics_columns
    assert list(assessments_df.columns) == expected_assessments_columns
    assert list(tags_df.columns) == expected_tags_columns


def test_evaluations_to_dataframes_basic():
    # Setup an evaluation with minimal data
    evaluation = EvaluationEntity(
        evaluation_id="eval1",
        run_id="run1",
        inputs_id="inputs1",
        inputs={"feature1": 1.0, "feature2": 2.0},
    )

    evaluations_df, metrics_df, assessments_df, tags_df = evaluations_to_dataframes([evaluation])

    # Check the evaluations DataFrame
    assert len(evaluations_df) == 1
    assert evaluations_df["evaluation_id"].iloc[0] == "eval1"
    assert evaluations_df["run_id"].iloc[0] == "run1"
    assert evaluations_df["inputs_id"].iloc[0] == "inputs1"
    assert evaluations_df["inputs"].iloc[0] == {"feature1": 1.0, "feature2": 2.0}

    # Check that the other


def test_evaluations_to_dataframes_different_assessments():
    # Different types of assessments in evaluations
    source = AssessmentSource(source_type="HUMAN", source_id="user_1")
    assessment_1 = AssessmentEntity(
        evaluation_id="eval1",
        name="accuracy",
        source=source,
        timestamp=123456789,
        numeric_value=0.95,
        rationale="Good performance",
    )
    assessment_2 = AssessmentEntity(
        evaluation_id="eval1",
        name="precision",
        source=source,
        timestamp=123456789,
        numeric_value=0.85,
        rationale="Reasonable performance",
    )

    evaluation = EvaluationEntity(
        evaluation_id="eval1",
        run_id="run1",
        inputs_id="inputs1",
        inputs={"feature1": 1.0, "feature2": 2.0},
        assessments=[assessment_1, assessment_2],
    )

    evaluations_df, metrics_df, assessments_df, tags_df = evaluations_to_dataframes([evaluation])

    # Check the assessments DataFrame
    assert len(assessments_df) == 2
    assert assessments_df["evaluation_id"].iloc[0] == "eval1"
    assert assessments_df["name"].iloc[0] == "accuracy"
    assert assessments_df["numeric_value"].iloc[0] == 0.95

    assert assessments_df["evaluation_id"].iloc[1] == "eval1"
    assert assessments_df["name"].iloc[1] == "precision"
    assert assessments_df["numeric_value"].iloc[1] == 0.85


def test_evaluations_to_dataframes_different_metrics():
    # Different types of metrics in evaluations
    metric_1 = Metric(key="metric1", value=0.9, timestamp=1234567890, step=0)
    metric_2 = Metric(key="metric2", value=0.8, timestamp=1234567891, step=0)

    evaluation = EvaluationEntity(
        evaluation_id="eval1",
        run_id="run1",
        inputs_id="inputs1",
        inputs={"feature1": 1.0, "feature2": 2.0},
        metrics=[metric_1, metric_2],
    )

    evaluations_df, metrics_df, assessments_df, tags_df = evaluations_to_dataframes([evaluation])

    # Check the metrics DataFrame
    assert len(metrics_df) == 2
    assert metrics_df["evaluation_id"].iloc[0] == "eval1"
    assert metrics_df["key"].iloc[0] == "metric1"
    assert metrics_df["value"].iloc[0] == 0.9
    assert metrics_df["timestamp"].iloc[0] == 1234567890

    assert metrics_df["evaluation_id"].iloc[1] == "eval1"
    assert metrics_df["key"].iloc[1] == "metric2"
    assert metrics_df["value"].iloc[1] == 0.8
    assert metrics_df["timestamp"].iloc[1] == 1234567891


def test_evaluations_to_dataframes_different_tags():
    # Different tags in evaluations
    tag1 = EvaluationTag(key="tag1", value="value1")
    tag2 = EvaluationTag(key="tag2", value="value2")

    evaluation = EvaluationEntity(
        evaluation_id="eval1",
        run_id="run1",
        inputs_id="inputs1",
        inputs={"feature1": 1.0, "feature2": 2.0},
        tags=[tag1, tag2],
    )

    evaluations_df, metrics_df, assessments_df, tags_df = evaluations_to_dataframes([evaluation])

    # Check the tags DataFrame
    assert len(tags_df) == 2
    assert tags_df["evaluation_id"].iloc[0] == "eval1"
    assert tags_df["key"].iloc[0] == "tag1"
    assert tags_df["value"].iloc[0] == "value1"

    assert tags_df["evaluation_id"].iloc[1] == "eval1"
    assert tags_df["key"].iloc[1] == "tag2"
    assert tags_df["value"].iloc[1] == "value2"
