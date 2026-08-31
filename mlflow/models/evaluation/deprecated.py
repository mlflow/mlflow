import functools
import warnings

from mlflow.models.evaluation import evaluate as model_evaluate


@functools.wraps(model_evaluate)
def evaluate(*args, **kwargs):
    model_type = kwargs.get("model_type")
    if model_type in {
        "question-answering",
        "text-summarization",
        "text",
        "retriever",
        "databricks-agent",
    }:
        from mlflow.agent.hint import maybe_warn_agent

        maybe_warn_agent(
            "legacy-genai-evaluate",
            "The deprecated mlflow.evaluate API is being used for a GenAI model; "
            "mlflow.genai.evaluate is the current API.",
        )
    warnings.warn(
        "The `mlflow.evaluate` API has been deprecated as of MLflow 3.0.0. "
        "Please use these new alternatives:\n\n"
        " - For traditional ML or deep learning models: Use `mlflow.models.evaluate`, "
        "which maintains full compatibility with the original `mlflow.evaluate` API.\n\n"
        " - For LLMs or GenAI applications: Use the new `mlflow.genai.evaluate` API, "
        "which offers enhanced features specifically designed for evaluating "
        "LLMs and GenAI applications.\n",
        FutureWarning,
    )
    return model_evaluate(*args, **kwargs)
