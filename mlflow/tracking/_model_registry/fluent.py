import json
import logging
import threading
import uuid
import warnings
from typing import Any

from pydantic import BaseModel

import mlflow
from mlflow.entities.logged_model import LoggedModel
from mlflow.entities.model_registry import ModelVersion, Prompt, PromptVersion, RegisteredModel
from mlflow.entities.model_registry.prompt_version import PromptModelConfig
from mlflow.entities.run import Run
from mlflow.environment_variables import MLFLOW_PRINT_MODEL_URLS_ON_CREATION
from mlflow.exceptions import MlflowException
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.prompt.registry_utils import require_prompt_registry
from mlflow.protos.databricks_pb2 import (
    ALREADY_EXISTS,
    NOT_FOUND,
    RESOURCE_ALREADY_EXISTS,
    ErrorCode,
)
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.store.artifact.utils.models import _parse_model_id_if_present
from mlflow.store.model_registry import (
    SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
)
from mlflow.telemetry.events import LoadPromptEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.fluent import get_active_trace_id, get_current_active_span
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils.prompt import update_linked_prompts_tag
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.fluent import _get_latest_active_run, get_active_model_id
from mlflow.utils import get_results_from_paginated_fn, mlflow_tags
from mlflow.utils.databricks_utils import (
    _construct_databricks_uc_registered_model_url,
    get_workspace_id,
    get_workspace_url,
    stage_model_for_databricks_model_serving,
)
from mlflow.utils.env_pack import (
    EnvPackConfig,
    EnvPackType,
    _validate_env_pack,
    pack_env_for_databricks_model_serving,
)
from mlflow.utils.logging_utils import eprint
from mlflow.utils.uri import is_databricks_unity_catalog_uri

_logger = logging.getLogger(__name__)


PROMPT_API_MIGRATION_MSG = (
    "The `mlflow.{func_name}` API is moved to the `mlflow.genai` namespace. Please use "
    "`mlflow.genai.{func_name}` instead. The original API will be removed in the "
    "future release."
)


def register_model(
    model_uri,
    name,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    *,
    tags: dict[str, Any] | None = None,
    env_pack: EnvPackType | EnvPackConfig | None = None,
) -> ModelVersion:
    """Create a new model version in model registry for the model files specified by ``model_uri``.

    Note that this method assumes the model registry backend URI is the same as that of the
    tracking backend.

    Args:
        model_uri: URI referring to the MLmodel directory. Use a ``runs:/`` URI if you want to
            record the run ID with the model in model registry (recommended), or pass the
            local filesystem path of the model if registering a locally-persisted MLflow
            model that was previously saved using ``save_model``.
            ``models:/`` URIs are currently not supported.
        name: Name of the registered model under which to create a new model version. If a
            registered model with the given name does not exist, it will be created
            automatically.
        await_registration_for: Number of seconds to wait for the model version to finish
            being created and is in ``READY`` status. By default, the function
            waits for five minutes. Specify 0 or None to skip waiting.
        tags: A dictionary of key-value pairs that are converted into
            :py:class:`mlflow.entities.model_registry.ModelVersionTag` objects.
        env_pack: Either a string or an EnvPackConfig. If specified,
            the model dependencies are optionally first installed into the current Python
            environment, and then the complete environment will be packaged and included
            in the registered model artifacts. If the string shortcut "databricks_model_serving" is
            used, then model dependencies will be installed in the current environment. This is
            useful when deploying the model to a serving environment like Databricks Model Serving.

            .. Note:: Experimental: This parameter may change or be removed in a future
                                    release without warning.

    Returns:
        Single :py:class:`mlflow.entities.model_registry.ModelVersion` object created by
        backend.

    .. code-block:: python
        :test:
        :caption: Example

        import mlflow.sklearn
        from mlflow.models import infer_signature
        from sklearn.datasets import make_regression
        from sklearn.ensemble import RandomForestRegressor

        mlflow.set_tracking_uri("sqlite:////tmp/mlruns.db")
        params = {"n_estimators": 3, "random_state": 42}
        X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
        # Log MLflow entities
        with mlflow.start_run() as run:
            rfr = RandomForestRegressor(**params).fit(X, y)
            signature = infer_signature(X, rfr.predict(X))
            mlflow.log_params(params)
            mlflow.sklearn.log_model(rfr, name="sklearn-model", signature=signature)
        model_uri = f"runs:/{run.info.run_id}/sklearn-model"
        mv = mlflow.register_model(model_uri, "RandomForestRegressionModel")
        print(f"Name: {mv.name}")
        print(f"Version: {mv.version}")

    .. code-block:: text
        :caption: Output

        Name: RandomForestRegressionModel
        Version: 1
    """
    return _register_model(
        model_uri=model_uri,
        name=name,
        await_registration_for=await_registration_for,
        tags=tags,
        env_pack=env_pack,
    )


def _register_model(
    model_uri,
    name,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    *,
    tags: dict[str, Any] | None = None,
    local_model_path=None,
    env_pack: EnvPackType | EnvPackConfig | None = None,
) -> ModelVersion:
    client = MlflowClient()
    try:
        create_model_response = client.create_registered_model(name)
        eprint(f"Successfully registered model '{create_model_response.name}'.")
    except MlflowException as e:
        if e.error_code in (
            ErrorCode.Name(RESOURCE_ALREADY_EXISTS),
            ErrorCode.Name(ALREADY_EXISTS),
        ):
            eprint(
                f"Registered model {name!r} already exists. Creating a new version of this model..."
            )
        else:
            raise e

    run_id = None
    model_id = None
    source = model_uri
    if RunsArtifactRepository.is_runs_uri(model_uri):
        # If the uri is of the form runs:/...
        (run_id, artifact_path) = RunsArtifactRepository.parse_runs_uri(model_uri)
        runs_artifact_repo = RunsArtifactRepository(model_uri)
        # List artifacts in `<run_artifact_root>/<artifact_path>` to see if the run has artifacts.
        # If so use the run's artifact location as source.
        artifacts = runs_artifact_repo._list_run_artifacts()
        if MLMODEL_FILE_NAME in (art.path for art in artifacts):
            source = RunsArtifactRepository.get_underlying_uri(model_uri)
        # Otherwise check if there's a logged model with
        # name artifact_path and source_run_id run_id
        else:
            run = client.get_run(run_id)
            logged_models = _get_logged_models_from_run(run, artifact_path)
            if not logged_models:
                raise MlflowException(
                    f"Unable to find a logged_model with artifact_path {artifact_path} "
                    f"under run {run_id}",
                    error_code=ErrorCode.Name(NOT_FOUND),
                )
            if len(logged_models) > 1:
                if run.outputs is None:
                    raise MlflowException.invalid_parameter_value(
                        f"Multiple logged models found for run {run_id}. Cannot determine "
                        "which model to register. Please use `models:/<model_id>` instead."
                    )
                # If there are multiple such logged models, get the one logged at the largest step
                model_id_to_step = {m_o.model_id: m_o.step for m_o in run.outputs.model_outputs}
                model_id = max(logged_models, key=lambda lm: model_id_to_step[lm.model_id]).model_id
            else:
                model_id = logged_models[0].model_id
            source = f"models:/{model_id}"
            _logger.warning(
                f"Run with id {run_id} has no artifacts at artifact path {artifact_path!r}, "
                f"registering model based on {source} instead"
            )

    # Otherwise if the uri is of the form models:/..., try to get the model_id from the uri directly
    model_id = _parse_model_id_if_present(model_uri) if not model_id else model_id

    # Passing in the string value is a shortcut for passing in the EnvPackConfig
    # Validate early; `_validate_env_pack` will raise on invalid inputs.
    validated_env_pack = _validate_env_pack(env_pack)

    # If env_pack is supported and indicates Databricks Model Serving, pack env and
    # log the resulting artifacts.
    if validated_env_pack:
        eprint(
            "Packing environment for Databricks Model Serving with install_dependencies "
            f"{validated_env_pack.install_dependencies}..."
        )
        with pack_env_for_databricks_model_serving(
            model_uri,
            enforce_pip_requirements=validated_env_pack.install_dependencies,
        ) as artifacts_path_with_env:
            client.log_model_artifacts(model_id, artifacts_path_with_env)

    create_version_response = client._create_model_version(
        name=name,
        source=source,
        run_id=run_id,
        tags=tags,
        await_creation_for=await_registration_for,
        local_model_path=local_model_path,
        model_id=model_id,
    )
    created_message = (
        f"Created version '{create_version_response.version}' of model "
        f"'{create_version_response.name}'"
    )
    # Print a link to the UC model version page if the model is in UC.
    registry_uri = mlflow.get_registry_uri()
    if (
        MLFLOW_PRINT_MODEL_URLS_ON_CREATION.get()
        and is_databricks_unity_catalog_uri(registry_uri)
        and (url := get_workspace_url())
    ):
        uc_model_url = _construct_databricks_uc_registered_model_url(
            url,
            create_version_response.name,
            create_version_response.version,
            get_workspace_id(),
        )
        created_message = "🔗 " + created_message + f": {uc_model_url}"
    else:
        created_message += "."
    eprint(created_message)

    if model_id:
        new_value = [
            {
                "name": create_version_response.name,
                "version": create_version_response.version,
            }
        ]
        model = client.get_logged_model(model_id)
        if existing_value := model.tags.get(mlflow_tags.MLFLOW_MODEL_VERSIONS):
            new_value = json.loads(existing_value) + new_value

        client.set_logged_model_tags(
            model_id,
            {mlflow_tags.MLFLOW_MODEL_VERSIONS: json.dumps(new_value)},
        )

    if validated_env_pack:
        eprint(
            f"Staging model {create_version_response.name} "
            f"version {create_version_response.version} "
            "for Databricks Model Serving..."
        )
        try:
            stage_model_for_databricks_model_serving(
                model_name=create_version_response.name,
                model_version=create_version_response.version,
            )
        except Exception as e:
            eprint(
                f"Failed to stage model for Databricks Model Serving: {e!s}. "
                "The model was registered successfully and is available for serving, but may take "
                "longer to deploy."
            )

    return create_version_response


def _get_logged_models_from_run(source_run: Run, model_name: str) -> list[LoggedModel]:
    """Get all logged models from the source rnu that have the specified model name.

    Args:
        source_run: Source run from which to retrieve logged models.
        model_name: Name of the model to retrieve.
    """
    client = MlflowClient()
    logged_models = []
    page_token = None

    while True:
        logged_models_page = client.search_logged_models(
            experiment_ids=[source_run.info.experiment_id],
            # TODO: Filter by 'source_run_id' once Databricks backend supports it
            filter_string=f"name = '{model_name}'",
            page_token=page_token,
        )
        logged_models.extend(
            m for m in logged_models_page if m.source_run_id == source_run.info.run_id
        )
        if not logged_models_page.token:
            break
        page_token = logged_models_page.token

    return logged_models


def search_registered_models(
    max_results: int | None = None,
    filter_string: str | None = None,
    order_by: list[str] | None = None,
) -> list[RegisteredModel]:
    """Search for registered models that satisfy the filter criteria.

    Args:
        max_results: If passed, specifies the maximum number of models desired. If not
            passed, all models will be returned.
        filter_string: Filter query string (e.g., "name = 'a_model_name' and tag.key = 'value1'"),
            defaults to searching for all registered models. The following identifiers, comparators,
            and logical operators are supported.

            Identifiers
              - "name": registered model name.
              - "tags.<tag_key>": registered model tag. If "tag_key" contains spaces, it must be
                wrapped with backticks (e.g., "tags.`extra key`").

            Comparators
              - "=": Equal to.
              - "!=": Not equal to.
              - "LIKE": Case-sensitive pattern match.
              - "ILIKE": Case-insensitive pattern match.

            Logical operators
              - "AND": Combines two sub-queries and returns True if both of them are True.

        order_by: List of column names with ASC|DESC annotation, to be used for ordering
            matching search results.

    Returns:
        A list of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
        that satisfy the search expressions.

    .. code-block:: python
        :test:
        :caption: Example

        import mlflow
        from sklearn.linear_model import LogisticRegression

        with mlflow.start_run():
            mlflow.sklearn.log_model(
                LogisticRegression(),
                name="Cordoba",
                registered_model_name="CordobaWeatherForecastModel",
            )
            mlflow.sklearn.log_model(
                LogisticRegression(),
                name="Boston",
                registered_model_name="BostonWeatherForecastModel",
            )

        # Get search results filtered by the registered model name
        filter_string = "name = 'CordobaWeatherForecastModel'"
        results = mlflow.search_registered_models(filter_string=filter_string)
        print("-" * 80)
        for res in results:
            for mv in res.latest_versions:
                print(f"name={mv.name}; run_id={mv.run_id}; version={mv.version}")

        # Get search results filtered by the registered model name that matches
        # prefix pattern
        filter_string = "name LIKE 'Boston%'"
        results = mlflow.search_registered_models(filter_string=filter_string)
        print("-" * 80)
        for res in results:
            for mv in res.latest_versions:
                print(f"name={mv.name}; run_id={mv.run_id}; version={mv.version}")

        # Get all registered models and order them by ascending order of the names
        results = mlflow.search_registered_models(order_by=["name ASC"])
        print("-" * 80)
        for res in results:
            for mv in res.latest_versions:
                print(f"name={mv.name}; run_id={mv.run_id}; version={mv.version}")

    .. code-block:: text
        :caption: Output

        --------------------------------------------------------------------------------
        name=CordobaWeatherForecastModel; run_id=248c66a666744b4887bdeb2f9cf7f1c6; version=1
        --------------------------------------------------------------------------------
        name=BostonWeatherForecastModel; run_id=248c66a666744b4887bdeb2f9cf7f1c6; version=1
        --------------------------------------------------------------------------------
        name=BostonWeatherForecastModel; run_id=248c66a666744b4887bdeb2f9cf7f1c6; version=1
        name=CordobaWeatherForecastModel; run_id=248c66a666744b4887bdeb2f9cf7f1c6; version=1
    """

    def pagination_wrapper_func(number_to_get, next_page_token):
        return MlflowClient().search_registered_models(
            max_results=number_to_get,
            filter_string=filter_string,
            order_by=order_by,
            page_token=next_page_token,
        )

    return get_results_from_paginated_fn(
        pagination_wrapper_func,
        SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
        max_results,
    )


def search_model_versions(
    max_results: int | None = None,
    filter_string: str | None = None,
    order_by: list[str] | None = None,
) -> list[ModelVersion]:
    """Search for model versions that satisfy the filter criteria.

    .. warning:

        The model version search results may not have aliases populated for performance reasons.

    Args:
        max_results: If passed, specifies the maximum number of models desired. If not
            passed, all models will be returned.
        filter_string: Filter query string
            (e.g., ``"name = 'a_model_name' and tag.key = 'value1'"``),
            defaults to searching for all model versions. The following identifiers, comparators,
            and logical operators are supported.

            Identifiers
              - ``name``: model name.
              - ``source_path``: model version source path.
              - ``run_id``: The id of the mlflow run that generates the model version.
              - ``tags.<tag_key>``: model version tag. If ``tag_key`` contains spaces, it must be
                wrapped with backticks (e.g., ``"tags.`extra key`"``).

            Comparators
              - ``=``: Equal to.
              - ``!=``: Not equal to.
              - ``LIKE``: Case-sensitive pattern match.
              - ``ILIKE``: Case-insensitive pattern match.
              - ``IN``: In a value list. Only ``run_id`` identifier supports ``IN`` comparator.

            Logical operators
              - ``AND``: Combines two sub-queries and returns True if both of them are True.

        order_by: List of column names with ASC|DESC annotation, to be used for ordering
            matching search results.

    Returns:
        A list of :py:class:`mlflow.entities.model_registry.ModelVersion` objects
            that satisfy the search expressions.

    .. code-block:: python
        :test:
        :caption: Example

        import mlflow
        from sklearn.linear_model import LogisticRegression

        for _ in range(2):
            with mlflow.start_run():
                mlflow.sklearn.log_model(
                    LogisticRegression(),
                    name="Cordoba",
                    registered_model_name="CordobaWeatherForecastModel",
                )

        # Get all versions of the model filtered by name
        filter_string = "name = 'CordobaWeatherForecastModel'"
        results = mlflow.search_model_versions(filter_string=filter_string)
        print("-" * 80)
        for res in results:
            print(f"name={res.name}; run_id={res.run_id}; version={res.version}")

        # Get the version of the model filtered by run_id
        filter_string = "run_id = 'ae9a606a12834c04a8ef1006d0cff779'"
        results = mlflow.search_model_versions(filter_string=filter_string)
        print("-" * 80)
        for res in results:
            print(f"name={res.name}; run_id={res.run_id}; version={res.version}")

    .. code-block:: text
        :caption: Output

        --------------------------------------------------------------------------------
        name=CordobaWeatherForecastModel; run_id=ae9a606a12834c04a8ef1006d0cff779; version=2
        name=CordobaWeatherForecastModel; run_id=d8f028b5fedf4faf8e458f7693dfa7ce; version=1
        --------------------------------------------------------------------------------
        name=CordobaWeatherForecastModel; run_id=ae9a606a12834c04a8ef1006d0cff779; version=2
    """

    def pagination_wrapper_func(number_to_get, next_page_token):
        return MlflowClient().search_model_versions(
            max_results=number_to_get,
            filter_string=filter_string,
            order_by=order_by,
            page_token=next_page_token,
        )

    return get_results_from_paginated_fn(
        paginated_fn=pagination_wrapper_func,
        max_results_per_page=SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
        max_results=max_results,
    )


def set_model_version_tag(
    name: str,
    version: str | None = None,
    key: str | None = None,
    value: Any = None,
) -> None:
    """
    Set a tag for the model version.

    Args:
        name: Registered model name.
        version: Registered model version.
        key: Tag key to log. key is required.
        value: Tag value to log. value is required.
    """
    return MlflowClient().set_model_version_tag(
        name=name,
        version=version,
        key=key,
        value=value,
    )


@require_prompt_registry
def register_prompt(
    name: str,
    template: str | list[dict[str, Any]],
    commit_message: str | None = None,
    tags: dict[str, str] | None = None,
    response_format: type[BaseModel] | dict[str, Any] | None = None,
    model_config: "PromptModelConfig | dict[str, Any] | None" = None,
) -> PromptVersion:
    """
    Register a new :py:class:`Prompt <mlflow.entities.Prompt>` in the MLflow Prompt Registry.

    A :py:class:`Prompt <mlflow.entities.Prompt>` is a pair of name and
    template content at minimum. With MLflow Prompt Registry, you can create, manage, and
    version control prompts with the MLflow's robust model tracking framework.

    If there is no registered prompt with the given name, a new prompt will be created.
    Otherwise, a new version of the existing prompt will be created.


    Args:
        name: The name of the prompt.
        template: The template content of the prompt. Can be either:
            - A string containing text with variables enclosed in double curly braces,
              e.g. {{variable}}, which will be replaced with actual values by the `format` method.
            - A list of dictionaries representing chat messages, where each message has
              'role' and 'content' keys (e.g., [{"role": "user", "content": "Hello {{name}}"}])

            .. note::

                If you want to use the prompt with a framework that uses single curly braces
                e.g. LangChain, you can use the `to_single_brace_format` method to convert the
                loaded prompt to a format that uses single curly braces.

                .. code-block:: python

                    prompt = client.load_prompt("my_prompt")
                    langchain_format = prompt.to_single_brace_format()

        commit_message: A message describing the changes made to the prompt, similar to a
            Git commit message. Optional.
        tags: A dictionary of tags associated with the **prompt version**.
            This is useful for storing version-specific information, such as the author of
            the changes. Optional.
        response_format: Optional Pydantic class or dictionary defining the expected response
            structure. This can be used to specify the schema for structured outputs from LLM calls.
        model_config: Optional PromptModelConfig instance or dictionary containing model-specific
            configuration. Using PromptModelConfig provides validation and type safety.

    Returns:
        A :py:class:`Prompt <mlflow.entities.Prompt>` object that was created.

    Example:

    .. code-block:: python

        import mlflow
        from pydantic import BaseModel

        # Register a text prompt
        mlflow.register_prompt(
            name="greeting_prompt",
            template="Respond to the user's message as a {{style}} AI.",
            response_format={"type": "string", "description": "A friendly response"},
        )

        # Register a chat prompt with multiple messages
        mlflow.register_prompt(
            name="assistant_prompt",
            template=[
                {"role": "system", "content": "You are a helpful {{style}} assistant."},
                {"role": "user", "content": "{{question}}"},
            ],
            response_format={"type": "object", "properties": {"answer": {"type": "string"}}},
        )

        # Load the prompt from the registry
        prompt = mlflow.load_prompt("greeting_prompt")

        # Use the prompt in your application
        import openai

        openai_client = openai.OpenAI()
        openai_client.chat.completion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt.format(style="friendly")},
                {"role": "user", "content": "Hello, how are you?"},
            ],
        )

        # Update the prompt with a new version
        prompt = mlflow.register_prompt(
            name="greeting_prompt",
            template="Respond to the user's message as a {{style}} AI. {{greeting}}",
            commit_message="Add a greeting to the prompt.",
            tags={"author": "Bob"},
        )
    """
    warnings.warn(
        PROMPT_API_MIGRATION_MSG.format(func_name="register_prompt"),
        category=FutureWarning,
        stacklevel=3,
    )

    return MlflowClient().register_prompt(
        name=name,
        template=template,
        commit_message=commit_message,
        tags=tags,
        response_format=response_format,
        model_config=model_config,
    )


@require_prompt_registry
def search_prompts(
    filter_string: str | None = None,
    max_results: int | None = None,
) -> list[Prompt]:
    """
    Search for prompts in the MLflow Prompt Registry.

    This call returns prompt metadata for prompts that have been marked
    as prompts (i.e. tagged with `mlflow.prompt.is_prompt=true`). We can
    further restrict results via a standard registry filter expression.

    Args:
        filter_string (Optional[str]):
            An additional registry-search expression to apply (e.g.
            `"name LIKE 'my_prompt%'"`).  For Unity Catalog registries, must include
            catalog and schema: "catalog = 'catalog_name' AND schema = 'schema_name'".
        max_results (Optional[int]):
            The maximum number of prompts to return.

    Returns:
        A list of :py:class:`Prompt <mlflow.entities.Prompt>` objects representing prompt metadata:

        - name: The prompt name
        - description: The prompt description
        - tags: Prompt-level tags
        - creation_timestamp: When the prompt was created

        To get the actual prompt template content,
        use :py:func:`mlflow.genai.load_prompt()` API with a specific version:

        .. code-block:: python
            import mlflow

            # Search for prompts
            prompts = mlflow.genai.search_prompts(filter_string="name LIKE 'greeting%'")

            # Get prompts by experiment
            prompts = mlflow.genai.search_prompts(filter_string='experiment_id = "1"')

            # Get specific version content
            for prompt in prompts:
                prompt_version = mlflow.genai.load_prompt(prompt.name, version="1")
                print(f"Template: {prompt_version.template}")
    """
    warnings.warn(
        PROMPT_API_MIGRATION_MSG.format(func_name="search_prompts"),
        category=FutureWarning,
        stacklevel=3,
    )

    def pagination_wrapper_func(number_to_get, next_page_token):
        return MlflowClient().search_prompts(
            filter_string=filter_string, max_results=number_to_get, page_token=next_page_token
        )

    return get_results_from_paginated_fn(
        pagination_wrapper_func,
        SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
        max_results,
    )


@require_prompt_registry
@record_usage_event(LoadPromptEvent)
def load_prompt(
    name_or_uri: str,
    version: str | int | None = None,
    allow_missing: bool = False,
    link_to_model: bool = True,
    model_id: str | None = None,
    cache_ttl_seconds: float | None = None,
) -> PromptVersion:
    """
    Load a :py:class:`Prompt <mlflow.entities.Prompt>` from the MLflow Prompt Registry.

    The prompt can be specified by name and version, or by URI.

    Args:
        name_or_uri: The name of the prompt, or the URI in the format "prompts:/name/version".
        version: The version of the prompt (required when using name, not allowed when using URI).
        allow_missing: If True, return None instead of raising Exception if the specified prompt
            is not found.
        link_to_model: If True, the prompt will be linked to the model with the ID specified
                       by `model_id`, or the active model ID if `model_id` is None and
                       there is an active model.
        model_id: The ID of the model to which to link the prompt, if `link_to_model` is True.
        cache_ttl_seconds: Time-to-live in seconds for the cached prompt. If not specified,
            uses the value from `MLFLOW_ALIAS_PROMPT_CACHE_TTL_SECONDS` environment variable for
            alias-based prompts (default 60), and the value from
            `MLFLOW_VERSION_PROMPT_CACHE_TTL_SECONDS` environment variable for version-based prompts
            (default None, no TTL). Set to 0 to bypass the cache and always fetch from the server.

    Example:

    .. code-block:: python

        import mlflow

        # Load a specific version of the prompt
        prompt = mlflow.load_prompt("my_prompt", version=1)

        # Load a specific version of the prompt by URI
        prompt = mlflow.load_prompt("prompts:/my_prompt/1")

        # Load a prompt version with an alias "production"
        prompt = mlflow.load_prompt("prompts:/my_prompt@production")

        # Load with custom cache TTL (5 minutes)
        prompt = mlflow.load_prompt("my_prompt", version=1, cache_ttl_seconds=300)

        # Bypass cache entirely
        prompt = mlflow.load_prompt("my_prompt", version=1, cache_ttl_seconds=0)

    """
    warnings.warn(
        PROMPT_API_MIGRATION_MSG.format(func_name="load_prompt"),
        category=FutureWarning,
        stacklevel=3,
    )

    client = MlflowClient()

    # Load prompt with caching (handled by client)
    prompt = client.load_prompt(
        name_or_uri=name_or_uri,
        version=version,
        allow_missing=allow_missing,
        cache_ttl_seconds=cache_ttl_seconds,
    )
    if prompt is None:
        return

    # If there is an active MLflow run, associate the prompt with the run.
    # Note that we do this synchronously because it's unlikely that run linking occurs
    # in a latency sensitive environment, since runs aren't typically used in real-time /
    # production scenarios
    # NB: We shouldn't use `active_run()` here because it only returns the active run
    # from the current thread. It doesn't work in multi-threaded scenarios such as
    # MLflow GenAI evaluation.
    if run := _get_latest_active_run():
        client.link_prompt_version_to_run(run.info.run_id, prompt)

    if link_to_model:
        model_id = model_id or get_active_model_id()
        if model_id is not None:
            # Run linking in background thread to avoid blocking prompt loading. Prompt linking
            # is not critical for the user's workflow (if the prompt fails to link, the user's
            # workflow is minorly affected), so we handle it asynchronously and gracefully
            # handle any failures without impacting the core prompt loading functionality.

            def _link_prompt_async():
                try:
                    client.link_prompt_version_to_model(
                        name=prompt.name,
                        version=prompt.version,
                        model_id=model_id,
                    )
                except Exception:
                    # NB: We should still load the prompt even if linking fails, since the prompt
                    # is critical to the caller's application logic
                    _logger.warning(
                        f"Failed to link prompt '{prompt.name}' version '{prompt.version}'"
                        f" to model '{model_id}'.",
                        exc_info=True,
                    )

            # Start linking in background - don't wait for completion
            link_thread = threading.Thread(
                target=_link_prompt_async, name=f"link_prompt_thread-{uuid.uuid4().hex[:8]}"
            )
            link_thread.start()

    if trace_id := get_active_trace_id():
        InMemoryTraceManager.get_instance().register_prompt(
            trace_id=trace_id,
            prompt=prompt,
        )

    # Set prompt version information as span attributes if there's an active span
    if span := get_current_active_span():
        current_value = span.attributes.get(SpanAttributeKey.LINKED_PROMPTS)
        updated_value = update_linked_prompts_tag(current_value, [prompt])
        span.set_attribute(SpanAttributeKey.LINKED_PROMPTS, updated_value)

    return prompt


@require_prompt_registry
def set_prompt_alias(name: str, alias: str, version: int) -> None:
    """
    Set an alias for a :py:class:`Prompt <mlflow.entities.Prompt>` in the MLflow Prompt Registry.

    Args:
        name: The name of the prompt.
        alias: The alias to set for the prompt.
        version: The version of the prompt.

    Example:

    .. code-block:: python

        import mlflow

        # Set an alias for the prompt
        mlflow.set_prompt_alias(name="my_prompt", version=1, alias="production")

        # Load the prompt by alias (use "@" to specify the alias)
        prompt = mlflow.load_prompt("prompts:/my_prompt@production")

        # Switch the alias to a new version of the prompt
        mlflow.set_prompt_alias(name="my_prompt", version=2, alias="production")

        # Delete the alias
        mlflow.delete_prompt_alias(name="my_prompt", alias="production")
    """
    warnings.warn(
        PROMPT_API_MIGRATION_MSG.format(func_name="set_prompt_alias"),
        category=FutureWarning,
        stacklevel=3,
    )

    MlflowClient().set_prompt_alias(name=name, version=version, alias=alias)


@require_prompt_registry
def delete_prompt_alias(name: str, alias: str) -> None:
    """
    Delete an alias for a :py:class:`Prompt <mlflow.entities.Prompt>` in the MLflow Prompt Registry.

    Args:
        name: The name of the prompt.
        alias: The alias to delete for the prompt.
    """
    warnings.warn(
        PROMPT_API_MIGRATION_MSG.format(func_name="delete_prompt_alias"),
        category=FutureWarning,
        stacklevel=3,
    )

    MlflowClient().delete_prompt_alias(name=name, alias=alias)
