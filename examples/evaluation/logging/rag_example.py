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
    inputs={
        "query": "Can you provide an example of extracting nested data from JSON using Databricks SQL?",
    },
    outputs={
        "retrieved_context": [
            {
                "doc_id": 1,
                "content": "Databricks SQL is a distributed SQL query engine that allows you to query and join data ...",
            }
        ],
        "response": "Sure, here's an example of extracting nested data from JSON using Databricks SQL: ...",
    },
    inputs_id="query-id-1",
    request_id="fake_tracing_request_id_1",
    targets={
        "correct_response": "Here is an example of extracting nested data from JSON using Databricks SQL: ..."
    },
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
            name="code_readability_score",
            value=7,
            source=AssessmentSource(
                source_id="gpt-4",
                source_type=AssessmentSourceType.AI_JUDGE,
                metadata={"judge_prompt": "fake judge prompt"},
            ),
        ),
        Assessment(
            name="context_is_relevant",
            value=True,
            source=AssessmentSource(
                source_id="gpt-4",
                source_type=AssessmentSourceType.AI_JUDGE,
                metadata={"judge_prompt": "fake judge prompt"},
            ),
        ),
    ],
    metrics=[Metric(key="num_output_tokens", value=50, timestamp=int(time.time() * 1000), step=0)],
)

evaluation2 = Evaluation(
    inputs={"query": "How many metastores can I have per region?"},
    outputs={
        # Failed to retrieve any relevant documents
        "retrieved_context": [],
        "response": "On AWS Glue, you can have one metastore per region.",
    },
    inputs_id="query-id-2",
    request_id="fake_tracing_request_id_2",
    # No targets
    targets={
        "correct_response": "You can have up to 100 metastores per region in Databricks Unity Catalog."
    },
    assessments=[
        Assessment(
            name="overall_judgement",
            value="bad_context",
            source=AssessmentSource(
                source_id="gpt-4",
                source_type=AssessmentSourceType.AI_JUDGE,
                metadata={"judge_prompt": "fake judge prompt"},
            ),
        ),
        Assessment(
            name="context_is_relevant",
            value=False,
            source=AssessmentSource(
                source_id="gpt-4",
                source_type=AssessmentSourceType.AI_JUDGE,
                metadata={"judge_prompt": "fake judge prompt"},
            ),
        ),
    ],
    metrics=[Metric(key="num_output_tokens", value=25, timestamp=int(time.time() * 1000), step=0)],
)

with mlflow.start_run():
    print("\n")
    print("Started run with ID:", mlflow.active_run().info.run_id)
    print("\n")
    print("\n")

    # Log two evaluations with assessments (feedback from an LLM judge). The assessment from the
    # LLM judge on the first evaluatioon indicates that the response is correct, readable, and
    # comes from relevant context. The assessment from the LLM judge on the second evaluation
    # indicates that the response is bad because the context is irrelevant (because it is missing).
    logged_evaluations = log_evaluations(
        run_id=mlflow.active_run().info.run_id, evaluations=[evaluation1, evaluation2]
    )
    print("\n")
    print("Logged evaluations:\n\n")
    print(logged_evaluations)
    print("\n\n")
    print("Logged run metrics:\n\n")
    print(mlflow.get_run(mlflow.active_run().info.run_id).data.metrics)

    # Add an assessment from a human on the first evaluation, which identifies some problems
    # with the response: the code is hard to read / there's a more elegant approach.
    # (This will normally be done through the MLflow UI)
    log_assessments(
        run_id=mlflow.active_run().info.run_id,
        evaluation_id=logged_evaluations[0].evaluation_id,
        assessments=[
            Assessment(
                name="overall_judgement",
                value="low_quality_code",
                source=AssessmentSource(
                    source_id="coreyuserid",
                    source_type=AssessmentSourceType.HUMAN,
                    metadata={"judge_prompt": "fake judge prompt"},
                ),
                rationale="The model hallucinated a response.",
            ),
            Assessment(
                name="code_readability_score",
                value=3,
                source=AssessmentSource(
                    source_id="coreyuserid",
                    source_type=AssessmentSourceType.HUMAN,
                    metadata={"judge_prompt": "fake judge prompt"},
                ),
            ),
            Assessment(
                name="context_is_relevant",
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

    # Print the evaluation info as a dataframe.
    # This will be cleaned up substantially and folded into a clean search API in the near future.
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
