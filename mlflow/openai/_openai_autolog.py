import logging

import mlflow
from mlflow.entities import RunTag
from mlflow.tracking.context import registry as context_registry
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.autologging_utils import disable_autologging, get_autologging_config
from mlflow.utils.autologging_utils.safety import _resolve_extra_tags

_logger = logging.getLogger(__name__)


def _patched_call(original, self, *args, **kwargs):
    run_id = getattr(self, "run_id", None)
    active_run = mlflow.active_run()
    if run_id is None:
        # only log the tags once
        extra_tags = get_autologging_config(
            mlflow.openai.FLAVOR_NAME, "extra_tags", None
        )
        # include run context tags
        resolved_tags = context_registry.resolve_tags(extra_tags)
        tags = _resolve_extra_tags(mlflow.openai.FLAVOR_NAME, resolved_tags)
        if active_run:
            run_id = active_run.info.run_id
            mlflow.MlflowClient().log_batch(
                run_id=run_id,
                tags=[RunTag(key, str(value)) for key, value in tags.items()],
            )
        else:
            run = mlflow.MlflowClient().create_run(
                experiment_id=_get_experiment_id(),
                tags=tags,
            )
            run_id = run.info.run_id

    with disable_autologging():
        result = original(self, *args, **kwargs)

    log_models = get_autologging_config(mlflow.openai.FLAVOR_NAME, "log_models", False)
    log_input_examples = get_autologging_config(
        mlflow.openai.FLAVOR_NAME, "log_input_examples", False
    )
    log_model_signatures = get_autologging_config(
        mlflow.openai.FLAVOR_NAME, "log_model_signatures", False
    )
    input_example = None
    if log_models and not hasattr(self, "model_logged"):
        if log_input_examples:
            # input_example = deepcopy(_get_input_data_from_function(func_name, self, args, kwargs))
            if not log_model_signatures:
                _logger.info(
                    "Signature is automatically generated for logged model if "
                    "input_example is provided. To disable log_model_signatures, "
                    "please also disable log_input_examples."
                )

        registered_model_name = get_autologging_config(
            mlflow.openai.FLAVOR_NAME, "registered_model_name", None
        )
        try:
            task = mlflow.openai._get_task_name_from_class(self.__class__)
            with disable_autologging():
                mlflow.openai.log_model(
                    kwargs.get("model", None),
                    task,
                    "model",
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                    run_id=run_id,
                )
        except Exception as e:
            _logger.warning(f"Failed to log model due to error {e}.")
        self.model_logged = True

    if not hasattr(self, "run_id"):
        self.run_id = run_id

    # Terminate the run if it is not managed by the user
    if active_run is None or active_run.info.run_id != run_id:
        mlflow.MlflowClient().set_terminated(run_id)

    return result
