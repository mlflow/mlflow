_MIGRATION_GUIDE = (
    "Use the new GenAI evaluation functionality instead. See "
    "https://mlflow.org/docs/latest/genai/eval-monitor/legacy-llm-evaluation/ "
    "for the migration guide."
)


def _get_latest_metric_version():
    return "v1"


def _get_default_model():
    return "openai:/gpt-4"
