from mlflow.utils.logging_utils import eprint

_BOLD_ORANGE = "\033[1;38;5;208m"
_LIGHT_BLUE = "\033[94m"
_RESET = "\033[0m"


def show_existing_experiment_upsell():
    doc_url = "https://docs.databricks.com/aws/en/mlflow3/genai/tracing/migrate-traces-to-uc"
    eprint(
        f"{_BOLD_ORANGE}If you are using MLflow Tracing, you can migrate your traces "
        f"to Unity Catalog for unlimited storage, fine-grained access controls, "
        f"and queryability from notebooks, SQL, and dashboards. "
        f"{_LIGHT_BLUE}Learn more: {doc_url}{_RESET}"
    )


def show_new_experiment_upsell():
    doc_url = "https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog"
    eprint(
        f"{_BOLD_ORANGE}If you are using MLflow Tracing, consider storing your "
        f"traces in Unity Catalog for unlimited storage (no 100,000 trace limit), "
        f"fine-grained access controls, and queryability from notebooks, SQL, "
        f"and dashboards. {_LIGHT_BLUE}Learn more: {doc_url}{_RESET}"
    )
