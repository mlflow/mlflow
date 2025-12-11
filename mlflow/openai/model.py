import importlib.metadata
import itertools
import logging
import os
import warnings
from functools import partial
from string import Formatter
from typing import Any

import yaml
from packaging.version import Version

import mlflow
from mlflow import pyfunc
from mlflow.entities.model_registry.prompt import Prompt
from mlflow.environment_variables import MLFLOW_OPENAI_SECRET_SCOPE
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME, _update_active_model_id_based_on_mlflow_model
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.openai.constant import FLAVOR_NAME
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types import ColSpec, Schema, TensorSpec
from mlflow.utils.databricks_utils import (
    check_databricks_secret_scope_access,
    is_in_databricks_runtime,
)
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.openai_utils import (
    _OAITokenHolder,
    _OpenAIApiConfig,
    _OpenAIEnvVar,
    _validate_model_params,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

MODEL_FILENAME = "model.yaml"
_PYFUNC_SUPPORTED_TASKS = ("chat.completions", "embeddings", "completions")

_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
        that, at minimum, contains these requirements.
    """
    return list(map(_get_pinned_requirement, ["openai", "tiktoken", "tenacity"]))


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


def _get_obj_to_task_mapping():
    from openai import resources as r

    mapping = {
        r.Audio: "audio",
        r.chat.Completions: "chat.completions",
        r.Completions: "completions",
        r.Images.edit: "images.edit",
        r.Embeddings: "embeddings",
        r.Files: "files",
        r.Images: "images",
        r.FineTuning: "fine_tuning",
        r.Moderations: "moderations",
        r.Models: "models",
        r.chat.AsyncCompletions: "chat.completions",
        r.AsyncCompletions: "completions",
        r.AsyncEmbeddings: "embeddings",
    }

    try:
        from openai.resources.beta.chat import completions as c

        mapping.update(
            {
                c.AsyncCompletions: "chat.completions",
                c.Completions: "chat.completions",
            }
        )
    except ImportError:
        pass
    return mapping


def _get_model_name(model):
    import openai

    if isinstance(model, str):
        return model

    if Version(_get_openai_package_version()).major < 1 and isinstance(model, openai.Model):
        return model.id

    raise mlflow.MlflowException(
        f"Unsupported model type: {type(model)}", error_code=INVALID_PARAMETER_VALUE
    )


def _get_task_name(task):
    mapping = _get_obj_to_task_mapping()
    if isinstance(task, str):
        if task not in mapping.values():
            raise mlflow.MlflowException(
                f"Unsupported task: {task}", error_code=INVALID_PARAMETER_VALUE
            )
        return task
    else:
        task_name = (
            mapping.get(task)
            or mapping.get(task.__class__)
            or mapping.get(getattr(task, "__func__"))  # if task is a method
        )
        if task_name is None:
            raise mlflow.MlflowException(
                f"Unsupported task object: {task}", error_code=INVALID_PARAMETER_VALUE
            )
        return task_name


def _get_api_config() -> _OpenAIApiConfig:
    """Gets the parameters and configuration of the OpenAI API connected to."""
    import openai

    api_type = os.getenv(_OpenAIEnvVar.OPENAI_API_TYPE.value, openai.api_type)
    api_version = os.getenv(_OpenAIEnvVar.OPENAI_API_VERSION.value, openai.api_version)
    api_base = os.getenv(_OpenAIEnvVar.OPENAI_API_BASE.value) or os.getenv(
        _OpenAIEnvVar.OPENAI_BASE_URL.value
    )
    deployment_id = os.getenv(_OpenAIEnvVar.OPENAI_DEPLOYMENT_NAME.value, None)
    organization = os.getenv(_OpenAIEnvVar.OPENAI_ORGANIZATION.value, None)
    if api_type in ("azure", "azure_ad", "azuread"):
        batch_size = 16
        max_tokens_per_minute = 60_000
    else:
        # The maximum batch size is 2048:
        # https://github.com/openai/openai-python/blob/b82a3f7e4c462a8a10fa445193301a3cefef9a4a/openai/embeddings_utils.py#L43
        # We use a smaller batch size to be safe.
        batch_size = 1024
        max_tokens_per_minute = 90_000
    return _OpenAIApiConfig(
        api_type=api_type,
        batch_size=batch_size,
        max_requests_per_minute=3_500,
        max_tokens_per_minute=max_tokens_per_minute,
        api_base=api_base,
        api_version=api_version,
        deployment_id=deployment_id,
        organization=organization,
    )


def _get_openai_package_version():
    return importlib.metadata.version("openai")


def _log_secrets_yaml(local_model_dir, scope):
    with open(os.path.join(local_model_dir, "openai.yaml"), "w") as f:
        yaml.safe_dump({e.value: f"{scope}:{e.secret_key}" for e in _OpenAIEnvVar}, f)


def _parse_format_fields(content: str | list[Any] | dict[str, Any] | Any) -> set[str]:
    """Parse format fields from content recursively."""
    if isinstance(content, str):
        return {fn for _, fn, _, _ in Formatter().parse(content) if fn is not None}
    elif isinstance(content, list):
        # Handle multimodal content (list of objects)
        fields = set()
        for item in content:
            if isinstance(item, dict):
                for value in item.values():
                    fields.update(_parse_format_fields(value))  # Recursive call
            elif isinstance(item, str):
                fields.update(_parse_format_fields(item))
        return fields
    elif isinstance(content, dict):
        # Handle dict content (recursively)
        fields = set()
        for value in content.values():
            fields.update(_parse_format_fields(value))  # Recursive call
        return fields
    else:
        # For other types (e.g., None), return empty set
        return set()


def _get_input_schema(task, content):
    if content:
        formatter = _ContentFormatter(task, content)
        variables = formatter.variables
        if len(variables) == 1:
            return Schema([ColSpec(type="string")])
        elif len(variables) > 1:
            return Schema([ColSpec(name=v, type="string") for v in variables])
        else:
            return Schema([ColSpec(type="string")])
    else:
        return Schema([ColSpec(type="string")])


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    model,
    task,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    **kwargs,
):
    """
    Save an OpenAI model to a path on the local file system.

    Args:
        model: The OpenAI model name.
        task: The task the model is performing, e.g., ``openai.chat.completions`` or
            ``'chat.completions'``.
        path: Local path where the model is to be saved.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
            describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
            The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
            from datasets with valid model input (e.g. the training dataset with target
            column omitted) and valid model output (e.g. model predictions generated on
            the training dataset), for example:

            .. code-block:: python

                from mlflow.models import infer_signature

                train = df.drop_column("target_label")
                predictions = ...  # compute model predictions
                signature = infer_signature(train, predictions)
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
        kwargs: Keyword arguments specific to the OpenAI task, such as the ``messages`` (see
            :ref:`mlflow.openai.messages` for more details on this parameter)
            or ``top_p`` value to use for chat completion.

    .. code-block:: python

        import mlflow
        import openai

        # Chat
        mlflow.openai.save_model(
            model="gpt-4o-mini",
            task=openai.chat.completions,
            messages=[{"role": "user", "content": "Tell me a joke."}],
            path="model",
        )

        # Completions
        mlflow.openai.save_model(
            model="text-davinci-002",
            task=openai.completions,
            prompt="{text}. The general sentiment of the text is",
            path="model",
        )

        # Embeddings
        mlflow.openai.save_model(
            model="text-embedding-ada-002",
            task=openai.embeddings,
            path="model",
        )
    """
    if Version(_get_openai_package_version()).major < 1:
        raise MlflowException("Only openai>=1.0 is supported.")

    import numpy as np

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)
    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)
    task = _get_task_name(task)

    if mlflow_model is None:
        mlflow_model = Model()

    if signature is not None:
        if signature.params:
            _validate_model_params(
                task, kwargs, {p.name: p.default for p in signature.params.params}
            )
    elif task == "chat.completions":
        messages = kwargs.get("messages", [])
        if messages and not (
            all(isinstance(m, dict) for m in messages) and all(map(_is_valid_message, messages))
        ):
            raise mlflow.MlflowException.invalid_parameter_value(
                "If `messages` is provided, it must be a list of dictionaries with keys "
                "'role' and 'content'."
            )

        signature = ModelSignature(
            inputs=_get_input_schema(task, messages),
            outputs=Schema([ColSpec(type="string", name=None)]),
        )
    elif task == "completions":
        prompt = kwargs.get("prompt")
        signature = ModelSignature(
            inputs=_get_input_schema(task, prompt),
            outputs=Schema([ColSpec(type="string", name=None)]),
        )
    elif task == "embeddings":
        signature = ModelSignature(
            inputs=Schema([ColSpec(type="string", name=None)]),
            outputs=Schema([TensorSpec(type=np.dtype("float64"), shape=(-1,))]),
        )

    saved_example = _save_example(mlflow_model, input_example, path)
    if signature is None and saved_example is not None:
        wrapped_model = _OpenAIWrapper(model)
        signature = _infer_signature_from_input_example(saved_example, wrapped_model)

    if signature is not None:
        mlflow_model.signature = signature

    if metadata is not None:
        mlflow_model.metadata = metadata
    model_data_path = os.path.join(path, MODEL_FILENAME)
    model_dict = {
        "model": _get_model_name(model),
        "task": task,
        **kwargs,
    }
    with open(model_data_path, "w") as f:
        yaml.safe_dump(model_dict, f)

    if task in _PYFUNC_SUPPORTED_TASKS:
        pyfunc.add_to_model(
            mlflow_model,
            loader_module="mlflow.openai",
            data=MODEL_FILENAME,
            conda_env=_CONDA_ENV_FILE_NAME,
            python_env=_PYTHON_ENV_FILE_NAME,
            code=code_dir_subpath,
        )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        openai_version=_get_openai_package_version(),
        data=MODEL_FILENAME,
        code=code_dir_subpath,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if is_in_databricks_runtime():
        if scope := MLFLOW_OPENAI_SECRET_SCOPE.get():
            url = "https://docs.databricks.com/en/machine-learning/model-serving/store-env-variable-model-serving.html"
            warnings.warn(
                "Specifying secrets for model serving with `MLFLOW_OPENAI_SECRET_SCOPE` is "
                f"deprecated. Use secrets-based environment variables ({url}) instead.",
                FutureWarning,
            )
            check_databricks_secret_scope_access(scope)
            _log_secrets_yaml(path, scope)

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path, FLAVOR_NAME, fallback=default_reqs
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # Save `requirements.txt`
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    model,
    task,
    artifact_path: str | None = None,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    prompts: list[str | Prompt] | None = None,
    name: str | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    model_type: str | None = None,
    step: int = 0,
    model_id: str | None = None,
    **kwargs,
):
    """
    Log an OpenAI model as an MLflow artifact for the current run.

    Args:
        model: The OpenAI model name or reference instance, e.g.,
            ``openai.Model.retrieve("gpt-4o-mini")``.
        task: The task the model is performing, e.g., ``openai.chat.completions`` or
            ``'chat.completions'``.
        artifact_path: Deprecated. Use `name` instead.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        registered_model_name: If given, create a model version under
            ``registered_model_name``, also creating a registered model if one
            with the given name does not exist.
        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
            describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
            The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
            from datasets with valid model input (e.g. the training dataset with target
            column omitted) and valid model output (e.g. model predictions generated on
            the training dataset), for example:

            .. code-block:: python

                from mlflow.models import infer_signature

                train = df.drop_column("target_label")
                predictions = ...  # compute model predictions
                signature = infer_signature(train, predictions)

        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version to finish
            being created and is in ``READY`` status. By default, the function
            waits for five minutes. Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
        prompts: {{ prompts }}
        name: {{ name }}
        params: {{ params }}
        tags: {{ tags }}
        model_type: {{ model_type }}
        step: {{ step }}
        model_id: {{ model_id }}
        kwargs: Keyword arguments specific to the OpenAI task, such as the ``messages`` (see
            :ref:`mlflow.openai.messages` for more details on this parameter)
            or ``top_p`` value to use for chat completion.

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.

    .. code-block:: python
        :caption: Example

        import mlflow
        import openai
        import pandas as pd

        # Chat
        with mlflow.start_run():
            info = mlflow.openai.log_model(
                model="gpt-4o-mini",
                task=openai.chat.completions,
                messages=[{"role": "user", "content": "Tell me a joke about {animal}."}],
                name="model",
            )
            model = mlflow.pyfunc.load_model(info.model_uri)
            df = pd.DataFrame({"animal": ["cats", "dogs"]})
            print(model.predict(df))

        # Embeddings
        with mlflow.start_run():
            info = mlflow.openai.log_model(
                model="text-embedding-ada-002",
                task=openai.embeddings,
                name="embeddings",
            )
            model = mlflow.pyfunc.load_model(info.model_uri)
            print(model.predict(["hello", "world"]))
    """
    return Model.log(
        artifact_path=artifact_path,
        name=name,
        flavor=mlflow.openai,
        registered_model_name=registered_model_name,
        model=model,
        task=task,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        prompts=prompts,
        params=params,
        tags=tags,
        model_type=model_type,
        step=step,
        model_id=model_id,
        **kwargs,
    )


def _load_model(path):
    model_file_path = os.path.dirname(path)
    if os.path.exists(model_file_path):
        mlflow_model = Model.load(model_file_path)
        _update_active_model_id_based_on_mlflow_model(mlflow_model)
    with open(path) as f:
        return yaml.safe_load(f)


def _is_valid_message(d):
    return isinstance(d, dict) and "content" in d and "role" in d


class _ContentFormatter:
    def __init__(self, task, template=None):
        if task == "completions":
            template = template or "{prompt}"
            if not isinstance(template, str):
                raise mlflow.MlflowException.invalid_parameter_value(
                    f"Template for task {task} expects type `str`, but got {type(template)}."
                )

            self.template = template
            self.format_fn = self.format_prompt
            self.variables = sorted(_parse_format_fields(self.template))
        elif task == "chat.completions":
            if not template:
                template = [{"role": "user", "content": "{content}"}]
            if not all(map(_is_valid_message, template)):
                raise mlflow.MlflowException.invalid_parameter_value(
                    f"Template for task {task} expects type `dict` with keys 'content' "
                    f"and 'role', but got {type(template)}."
                )

            self.template = template.copy()
            self.format_fn = self.format_chat
            self.variables = sorted(
                set(
                    itertools.chain.from_iterable(
                        _parse_format_fields(message.get("content"))
                        | _parse_format_fields(message.get("role"))
                        for message in self.template
                    )
                )
            )
            if not self.variables:
                self.template.append({"role": "user", "content": "{content}"})
                self.variables.append("content")
        else:
            raise mlflow.MlflowException.invalid_parameter_value(
                f"Task type ``{task}`` is not supported for formatting."
            )

    def format(self, **params):
        if missing_params := set(self.variables) - set(params):
            raise mlflow.MlflowException.invalid_parameter_value(
                f"Expected parameters {self.variables} to be provided, "
                f"only got {list(params)}, {list(missing_params)} are missing."
            )
        return self.format_fn(**params)

    def format_prompt(self, **params):
        return self.template.format(**{v: params[v] for v in self.variables})

    def format_chat(self, **params):
        format_args = {v: params[v] for v in self.variables}

        def format_value(
            value: str | list[Any] | dict[str, Any] | Any,
        ) -> str | list[Any] | dict[str, Any] | Any:
            if isinstance(value, str):
                return value.format(**format_args)
            elif isinstance(value, list):
                return [format_value(item) for item in value]
            elif isinstance(value, dict):
                return {key: format_value(val) for key, val in value.items()}
            else:
                return value

        formatted_messages = []

        for message in self.template:
            role = message.get("role")
            content = message.get("content")

            # Format role and content recursively
            formatted_role = format_value(role)
            formatted_content = format_value(content)

            formatted_messages.append(
                {
                    "role": formatted_role,
                    "content": formatted_content,
                }
            )

        return formatted_messages


def _first_string_column(pdf):
    iter_str_cols = (c for c, v in pdf.iloc[0].items() if isinstance(v, str))
    col = next(iter_str_cols, None)
    if col is None:
        raise mlflow.MlflowException.invalid_parameter_value(
            f"Could not find a string column in the input data: {pdf.dtypes.to_dict()}"
        )
    return col


class _OpenAIWrapper:
    def __init__(self, model):
        task = model.pop("task")
        if task not in _PYFUNC_SUPPORTED_TASKS:
            raise mlflow.MlflowException.invalid_parameter_value(
                f"Unsupported task: {task}. Supported tasks: {_PYFUNC_SUPPORTED_TASKS}."
            )
        self.model = model
        self.task = task
        self.api_config = _get_api_config()
        self.api_token = _OAITokenHolder(self.api_config.api_type)

        if self.task != "embeddings":
            self._setup_completions()

    def get_raw_model(self):
        """
        Returns the underlying model.
        """
        return self.model

    def _setup_completions(self):
        if self.task == "chat.completions":
            self.template = self.model.get("messages", [])
        else:
            self.template = self.model.get("prompt")
        self.formatter = _ContentFormatter(self.task, self.template)

    def format_completions(self, params_list):
        return [self.formatter.format(**params) for params in params_list]

    def get_params_list(self, data):
        if len(self.formatter.variables) == 1:
            variable = self.formatter.variables[0]
            if variable in data.columns:
                return data[[variable]].to_dict(orient="records")
            else:
                first_string_column = _first_string_column(data)
                return [{variable: s} for s in data[first_string_column]]
        else:
            return data[self.formatter.variables].to_dict(orient="records")

    def get_client(self, max_retries: int, timeout: float):
        # with_option method should not be used before v1.3.8: https://github.com/openai/openai-python/issues/865
        if self.api_config.api_type in ("azure", "azure_ad", "azuread"):
            from openai import AzureOpenAI

            return AzureOpenAI(
                api_key=self.api_token.token,
                azure_endpoint=self.api_config.api_base,
                api_version=self.api_config.api_version,
                azure_deployment=self.api_config.deployment_id,
                max_retries=max_retries,
                timeout=timeout,
            )
        else:
            from openai import OpenAI

            return OpenAI(
                api_key=self.api_token.token,
                base_url=self.api_config.api_base,
                max_retries=max_retries,
                timeout=timeout,
            )

    def _predict_chat(self, data, params):
        from mlflow.openai.api_request_parallel_processor import process_api_requests

        _validate_model_params(self.task, self.model, params)
        max_retries = params.pop("max_retries", self.api_config.max_retries)
        timeout = params.pop("timeout", self.api_config.timeout)

        messages_list = self.format_completions(self.get_params_list(data))
        client = self.get_client(max_retries=max_retries, timeout=timeout)

        requests = [
            partial(
                client.chat.completions.create,
                messages=messages,
                model=self.model["model"],
                **params,
            )
            for messages in messages_list
        ]

        results = process_api_requests(request_tasks=requests)

        return [r.choices[0].message.content for r in results]

    def _predict_completions(self, data, params):
        from mlflow.openai.api_request_parallel_processor import process_api_requests

        _validate_model_params(self.task, self.model, params)
        prompts_list = self.format_completions(self.get_params_list(data))
        max_retries = params.pop("max_retries", self.api_config.max_retries)
        timeout = params.pop("timeout", self.api_config.timeout)
        batch_size = params.pop("batch_size", self.api_config.batch_size)
        _logger.debug(f"Requests are being batched by {batch_size} samples.")

        client = self.get_client(max_retries=max_retries, timeout=timeout)

        requests = [
            partial(
                client.completions.create,
                prompt=prompts_list[i : i + batch_size],
                model=self.model["model"],
                **params,
            )
            for i in range(0, len(prompts_list), batch_size)
        ]

        results = process_api_requests(request_tasks=requests)

        return [row.text for batch in results for row in batch.choices]

    def _predict_embeddings(self, data, params):
        from mlflow.openai.api_request_parallel_processor import process_api_requests

        _validate_model_params(self.task, self.model, params)
        max_retries = params.pop("max_retries", self.api_config.max_retries)
        timeout = params.pop("timeout", self.api_config.timeout)
        batch_size = params.pop("batch_size", self.api_config.batch_size)
        _logger.debug(f"Requests are being batched by {batch_size} samples.")

        first_string_column = _first_string_column(data)
        texts = data[first_string_column].tolist()

        client = self.get_client(max_retries=max_retries, timeout=timeout)

        requests = [
            partial(
                client.embeddings.create,
                input=texts[i : i + batch_size],
                model=self.model["model"],
                **params,
            )
            for i in range(0, len(texts), batch_size)
        ]

        results = process_api_requests(request_tasks=requests)

        return [row.embedding for batch in results for row in batch.data]

    def predict(self, data, params: dict[str, Any] | None = None):
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions.
        """
        self.api_token.refresh()
        if self.task == "chat.completions":
            return self._predict_chat(data, params or {})
        elif self.task == "completions":
            return self._predict_completions(data, params or {})
        elif self.task == "embeddings":
            return self._predict_embeddings(data, params or {})


def _load_pyfunc(path):
    """Loads PyFunc implementation. Called by ``pyfunc.load_model``.

    Args:
        path: Local filesystem path to the MLflow Model with the ``openai`` flavor.
    """
    return _OpenAIWrapper(_load_model(path))


def load_model(model_uri, dst_path=None):
    """
    Load an OpenAI model from a local file or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
            artifact-locations>`_.
        dst_path: The local filesystem path to which to download the model artifact.
            This directory must already exist. If unspecified, a local output
            path will be created.

    Returns:
        A dictionary representing the OpenAI model.
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(local_model_path, FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    model_data_path = os.path.join(local_model_path, flavor_conf.get("data", MODEL_FILENAME))
    return _load_model(model_data_path)
