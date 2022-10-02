import mlflow
from mlflow.utils.autologging_utils import get_autologging_config


def get_best_model(self):
    if self.state.best_model_checkpoint:
        latest_model = self.model
        self._load_best_model()
        best_model = self.model
        self.model = latest_model
        return best_model
    else:
        return self.model


def patched_train(original, self, *args, **kwargs):
    """
    A patched implementation of `transformers.Trainer.train` which enables logging the
    following parameters, metrics and artifacts:

    - Training epochs
    - Optimizer parameters
    - Model checkpoints
    - Trained model
    """

    task = get_autologging_config(mlflow.transformers.FLAVOR_NAME, "task", None)
    log_models = get_autologging_config(mlflow.transformers.FLAVOR_NAME, "log_models", True)

    result = original(self, *args, **kwargs)

    if log_models:
        if task is None:
            raise ("Task name is required for logging transformers model", 400)

        mlflow.transformers.log_model(get_best_model(self), artifact_path="model_artifact", tokenizer=self.tokenizer, task=task)

    return result
