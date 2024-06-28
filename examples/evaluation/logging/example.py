import time

import mlflow
from mlflow.entities.metric import Metric
from mlflow.evaluation import (
    Assessment,
    AssessmentSource,
    AssessmentSourceType,
    Evaluation,
    get_evaluation,
    log_assessments,
    log_evaluations,
)
from mlflow.evaluation.utils import (
    read_assessments_dataframe,
    read_evaluations_dataframe,
    read_metrics_dataframe,
)

evaluation1 = Evaluation(
    inputs={"query": "example query", "context": "some context"},
    outputs={"response": "example answer for query"},
    inputs_id="query-id-1",
    request_id="fake_tracing_request_id",
    targets={"correct_response": "example correct response"},
    assessments=[
        Assessment(
            name="overall_judgement",
            value="correct",
            source=AssessmentSource(
                source_id="gpt-4",
                source_type=AssessmentSourceType.AI_JUDGE,
                metadata={"judge_prompt": "fake judge prompt"},
            ),
            rationale="As an AI judge, I think the response is correct because...",
        ),
        Assessment(
            name="readability_score",
            value=7,
            source=AssessmentSource(
                source_id="gpt-4",
                source_type=AssessmentSourceType.AI_JUDGE,
                metadata={"judge_prompt": "fake judge prompt"},
            ),
        ),
        Assessment(
            name="is_relevant",
            value=True,
            source=AssessmentSource(
                source_id="gpt-4",
                source_type=AssessmentSourceType.AI_JUDGE,
                metadata={"judge_prompt": "fake judge prompt"},
            ),
        ),
    ],
    metrics=[Metric(key="num_output_tokens", value=25, timestamp=int(time.time() * 1000), step=0)],
)

evaluation2 = Evaluation(
    inputs={"query": "example query 2", "context": "some context"},
    outputs={"response": "example answer for query 2"},
    inputs_id="query-id-2",
    request_id="fake_tracing_request_id",
    # No targets
    assessments=[
        Assessment(
            name="overall_judgement",
            value="hallucinating",
            source=AssessmentSource(
                source_id="gpt-4",
                source_type=AssessmentSourceType.AI_JUDGE,
                metadata={"judge_prompt": "fake judge prompt"},
            ),
        ),
        Assessment(
            name="readability_score",
            value=5,
            source=AssessmentSource(
                source_id="gpt-4",
                source_type=AssessmentSourceType.AI_JUDGE,
                metadata={"judge_prompt": "fake judge prompt"},
            ),
        ),
        Assessment(
            name="is_relevant",
            value=False,
            source=AssessmentSource(
                source_id="gpt-4",
                source_type=AssessmentSourceType.AI_JUDGE,
                metadata={"judge_prompt": "fake judge prompt"},
            ),
        ),
    ],
    metrics=[Metric(key="num_output_tokens", value=24, timestamp=int(time.time() * 1000), step=0)],
)

with mlflow.start_run():
    print("\n")
    print("Started run with ID:", mlflow.active_run().info.run_id)
    print("\n")
    print("\n")

    logged_evaluations = log_evaluations(
        run_id=mlflow.active_run().info.run_id, evaluations=[evaluation1, evaluation2]
    )
    print("\n")
    print("Logged evaluations:\n\n")
    print(logged_evaluations)
    print("\n\n")
    print("Logged run metrics:\n\n")
    print(mlflow.get_run(mlflow.active_run().info.run_id).data.metrics)

    log_assessments(
        run_id=mlflow.active_run().info.run_id,
        evaluation_id=logged_evaluations[0].evaluation_id,
        assessments=[
            Assessment(
                name="overall_judgement",
                value="hallucinating",
                source=AssessmentSource(
                    source_id="coreyuserid",
                    source_type=AssessmentSourceType.HUMAN,
                    metadata={"judge_prompt": "fake judge prompt"},
                ),
                rationale="The model hallucinated a response.",
            ),
            Assessment(
                name="readability_score",
                value=6,
                source=AssessmentSource(
                    source_id="coreyuserid",
                    source_type=AssessmentSourceType.HUMAN,
                    metadata={"judge_prompt": "fake judge prompt"},
                ),
            ),
            Assessment(
                name="is_relevant",
                value=True,
                source=AssessmentSource(
                    source_id="coreyuserid",
                    source_type=AssessmentSourceType.HUMAN,
                    metadata={"judge_prompt": "fake judge prompt"},
                ),
            ),
        ],
    )

    print("\n\n")
    print(f"Updated evaluation with ID {logged_evaluations[0].evaluation_id}:")
    print("\n\n")
    print(
        get_evaluation(
            run_id=mlflow.active_run().info.run_id,
            evaluation_id=logged_evaluations[0].evaluation_id,
        )
    )

    print("\n\n")
    print("Updated run metrics:\n\n")
    print(mlflow.get_run(mlflow.active_run().info.run_id).data.metrics)

    print("\n\n")
    print("Artifacts for evaluation, assessments, and evaluation metrics storage:\n\n")
    evaluations_path = mlflow.MlflowClient().download_artifacts(
        mlflow.active_run().info.run_id, "_evaluations.json"
    )
    assessments_path = mlflow.MlflowClient().download_artifacts(
        mlflow.active_run().info.run_id, "_assessments.json"
    )
    metrics_path = mlflow.MlflowClient().download_artifacts(
        mlflow.active_run().info.run_id, "_metrics.json"
    )
    print([evaluations_path, assessments_path, metrics_path])

    print("\n\n")
    print("Evaluations dataframe")
    evaluations_dataframe = read_evaluations_dataframe(evaluations_path)
    print(evaluations_dataframe.to_string())

    print("\n\n")
    print("Assessments dataframe")
    assessments_dataframe = read_assessments_dataframe(assessments_path)
    print(assessments_dataframe.to_string())

    print("\n\n")
    print("Metrics dataframe")
    metrics_dataframe = read_metrics_dataframe(metrics_path)
    print(metrics_dataframe.to_string())
