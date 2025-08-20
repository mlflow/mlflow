from clint.rules.base import Rule


class MlflowClassName(Rule):
    def _message(self) -> str:
        return "Should use `Mlflow` in class name, not `MLflow` or `MLFlow`."
