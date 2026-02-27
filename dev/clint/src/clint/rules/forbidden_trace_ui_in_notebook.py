from clint.rules.base import Rule


class ForbiddenTraceUIInNotebook(Rule):
    def _message(self) -> str:
        return (
            "Found the MLflow Trace UI iframe in the notebook. "
            "The trace UI in cell outputs will not render correctly in previews or the website. "
            "Please run `mlflow.tracing.disable_notebook_display()` and rerun the cell "
            "to remove the iframe."
        )
