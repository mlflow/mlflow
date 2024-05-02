import inspect
import json
import logging
import uuid
from copy import deepcopy

import mlflow
from mlflow import MlflowException
from mlflow.entities import RunTag
from mlflow.tracking.context import registry as context_registry
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.autologging_utils import disable_autologging, get_autologging_config
from mlflow.utils.autologging_utils.safety import _resolve_extra_tags

_logger = logging.getLogger(__name__)


def _get_input_from_model(model, kwargs):
    from openai.resources.chat.completions import Completions as ChatCompletions
    from openai.resources.completions import Completions
    from openai.resources.embeddings import Embeddings

    model_class_param_name_mapping = {
        ChatCompletions: "messages",
        Completions: "prompt",
        Embeddings: "input",
    }
    if param_name := model_class_param_name_mapping.get(model.__class__):
        # openai tasks accept only keyword arguments
        if param := kwargs.get(param_name):
            return param
        input_example_exc = MlflowException(
            "Inference function signature changes, please contact MLflow team to "
            "fix OpenAI autologging.",
        )
    else:
        input_example_exc = MlflowException(
            f"Unsupported OpenAI task. Only support {list(model_class_param_name_mapping.keys())}."
        )
    _logger.warning(
        f"Failed to gather input example of model {model.__class__.__name__} "
        f"due to error: {input_example_exc}."
    )


def _convert_data_to_dict(data, key):
    if isinstance(data, dict):
        return {f"{key}-{k}": v for k, v in data.items()}
    if isinstance(data, list):
        return {key: data}
    if isinstance(data, str):
        return {key: [data]}
    raise MlflowException("Unsupported data type.")


def _combine_input_and_output(input, output):
    """
    Combine input and output into a single dictionary
    """
    result = {}
    if input:
        result.update(_convert_data_to_dict(input, "input"))
    if output:
        output = [output.model_dump(mode="json")]
        result.update(_convert_data_to_dict(output, "output"))
    return result


def patched_call(original, self, *args, **kwargs):
    run_id = getattr(self, "run_id", None)
    active_run = mlflow.active_run()
    mlflow_client = mlflow.MlflowClient()
    if run_id is None:
        # only log the tags once
        extra_tags = get_autologging_config(mlflow.openai.FLAVOR_NAME, "extra_tags", None)
        # include run context tags
        resolved_tags = context_registry.resolve_tags(extra_tags)
        tags = _resolve_extra_tags(mlflow.openai.FLAVOR_NAME, resolved_tags)
        if active_run:
            run_id = active_run.info.run_id
            mlflow_client.log_batch(
                run_id=run_id,
                tags=[RunTag(key, str(value)) for key, value in tags.items()],
            )
        else:
            run = mlflow_client.create_run(
                experiment_id=_get_experiment_id(),
                tags=tags,
            )
            run_id = run.info.run_id

    with disable_autologging():
        result = original(self, *args, **kwargs)

    class _OpenAIJsonEncoder(json.JSONEncoder):
        def default(self, o):
            from openai._types import NotGiven

            if isinstance(o, NotGiven):
                return str(o)

            return super().default(o)

    # Use session_id-inference_id as artifact directory where mlflow
    # callback logs artifacts into, to avoid overriding artifacts
    session_id = getattr(self, "session_id", uuid.uuid4().hex)
    inference_id = getattr(self, "inference_id", 0)

    # log input and output as artifacts
    call_args = inspect.getcallargs(original, self, *args, **kwargs)
    call_args.pop("self")
    mlflow_client.log_text(
        run_id=run_id,
        text=json.dumps(call_args, indent=2, cls=_OpenAIJsonEncoder),
        artifact_file=f"artifacts-{session_id}-{inference_id}/input.json",
    )
    mlflow_client.log_text(
        run_id=run_id,
        text=result.to_json(),
        artifact_file=f"artifacts-{session_id}-{inference_id}/output.json",
    )

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
            input_example = deepcopy(_get_input_from_model(self, kwargs))
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
            _logger.warning(f"Failed to log model due to error: {e}.")
        self.model_logged = True

    # Even if the model is not logged, we keep a single run per model
    if not hasattr(self, "run_id"):
        self.run_id = run_id
    if not hasattr(self, "session_id"):
        self.session_id = session_id
    self.inference_id = inference_id + 1

    # Terminate the run if it is not managed by the user
    if active_run is None or active_run.info.run_id != run_id:
        mlflow_client.set_terminated(run_id)

    return result
