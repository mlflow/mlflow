import mlflow
import pandas as pd

from mlflow.genai.optimize.types import PromptOptimizerOutput
from mlflow.genai.optimize.util import prompt_optimization_autolog


class StubPrompt:
    def __init__(self, uri="prompts:/test/1"):
        self.uri = uri


def test_final_eval_score_zero_should_be_logged(tmp_path):
    # Minimal training data
    df = pd.DataFrame([
        {"inputs": {"q": "hello"}, "outputs": "world"}
    ])

    # Fake optimizer output with final_eval_score = 0.0
    output = PromptOptimizerOutput(
        optimized_prompts={"p": "template"},
        initial_eval_score=1.0,
        final_eval_score=0.0,
    )

    with prompt_optimization_autolog(
        optimizer_name="test_opt",
        num_prompts=1,
        num_training_samples=1,
        train_data_df=df,
    ) as results:
        results["optimizer_output"] = output

    # Verify metric logged
    client = mlflow.tracking.MlflowClient()

    runs = client.search_runs(
        experiment_ids=["0"],
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    assert runs, "No MLflow run found"
    run = runs[0]

    data = client.get_run(run.info.run_id).data
    assert "final_eval_score" in data.metrics
    assert float(data.metrics["final_eval_score"]) == 0.0
