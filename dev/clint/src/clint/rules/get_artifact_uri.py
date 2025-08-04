from clint.rules.base import Rule


class GetArtifactUri(Rule):
    def _message(self) -> str:
        return (
            "`mlflow.get_artifact_uri` should not be used in examples. "
            "Use the return value of `log_model` instead."
        )
