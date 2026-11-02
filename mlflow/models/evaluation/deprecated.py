import functools
import warnings

from mlflow.models.evaluation import evaluate as model_evaluate


@functools.wraps(model_evaluate)
def evaluate(*args, **kwargs):
    model_type = kwargs.get("model_type")
    warning_message = (
        "The `mlflow.evaluate` API has been deprecated as of MLflow 3.0.0. "
        "Please use these new alternatives:\n\n"
        " - For traditional ML or deep learning models: Use `mlflow.models.evaluate`, "
        "which maintains full compatibility with the original `mlflow.evaluate` API.\n\n"
        " - For LLMs or GenAI applications: Use the new `mlflow.genai.evaluate` API, "
        "which offers enhanced features specifically designed for evaluating "
        "LLMs and GenAI applications.\n"
    )
    if model_type in {
        "question-answering",
        "text-summarization",
        "text",
        "retriever",
        "databricks-agent",
    }:
        from mlflow.agent.hint import maybe_append_agent_hint

        warning_message = maybe_append_agent_hint(
            "legacy-genai-evaluate",
            warning_message,
        )
    warnings.warn(warning_message, FutureWarning)
    return model_evaluate(*args, **kwargs)
