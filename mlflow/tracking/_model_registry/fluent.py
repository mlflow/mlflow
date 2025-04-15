from typing import Any, Optional

from mlflow.entities.model_registry import ModelVersion, Prompt, RegisteredModel
from mlflow.exceptions import MlflowException
from mlflow.prompt.registry_utils import require_prompt_registry
from mlflow.protos.databricks_pb2 import ALREADY_EXISTS, RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.store.model_registry import (
    SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
)
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.fluent import active_run
from mlflow.utils import get_results_from_paginated_fn
from mlflow.utils.annotations import experimental
from mlflow.utils.logging_utils import eprint


def register_model(
    model_uri,
    name,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    *,
    tags: Optional[dict[str, Any]] = None,
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
            mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model", signature=signature)
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
        model_uri=model_uri, name=name, await_registration_for=await_registration_for, tags=tags
    )


def _register_model(
    model_uri,
    name,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    *,
    tags: Optional[dict[str, Any]] = None,
    local_model_path=None,
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
    source = model_uri
    if RunsArtifactRepository.is_runs_uri(model_uri):
        source = RunsArtifactRepository.get_underlying_uri(model_uri)
        (run_id, _) = RunsArtifactRepository.parse_runs_uri(model_uri)

    create_version_response = client._create_model_version(
        name=name,
        source=source,
        run_id=run_id,
        tags=tags,
        await_creation_for=await_registration_for,
        local_model_path=local_model_path,
    )
    eprint(
        f"Created version '{create_version_response.version}' of model "
        f"'{create_version_response.name}'."
    )
    return create_version_response


def search_registered_models(
    max_results: Optional[int] = None,
    filter_string: Optional[str] = None,
    order_by: Optional[list[str]] = None,
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
                "Cordoba",
                registered_model_name="CordobaWeatherForecastModel",
            )
            mlflow.sklearn.log_model(
                LogisticRegression(),
                "Boston",
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
    max_results: Optional[int] = None,
    filter_string: Optional[str] = None,
    order_by: Optional[list[str]] = None,
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
                    "Cordoba",
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


@experimental
@require_prompt_registry
def register_prompt(
    name: str,
    template: str,
    commit_message: Optional[str] = None,
    version_metadata: Optional[dict[str, str]] = None,
    tags: Optional[dict[str, str]] = None,
) -> Prompt:
    """
    Register a new :py:class:`Prompt <mlflow.entities.Prompt>` in the MLflow Prompt Registry.

    A :py:class:`Prompt <mlflow.entities.Prompt>` is a pair of name and
    template text at minimum. With MLflow Prompt Registry, you can create, manage, and
    version control prompts with the MLflow's robust model tracking framework.

    If there is no registered prompt with the given name, a new prompt will be created.
    Otherwise, a new version of the existing prompt will be created.


    Args:
        name: The name of the prompt.
        template: The template text of the prompt. It can contain variables enclosed in
            double curly braces, e.g. {variable}, which will be replaced with actual values
            by the `format` method.

            .. note::

                If you want to use the prompt with a framework that uses single curly braces
                e.g. LangChain, you can use the `to_single_brace_format` method to convert the
                loaded prompt to a format that uses single curly braces.

                .. code-block:: python

                    prompt = client.load_prompt("my_prompt")
                    langchain_format = prompt.to_single_brace_format()

        commit_message: A message describing the changes made to the prompt, similar to a
            Git commit message. Optional.
        version_metadata: A dictionary of metadata associated with the **prompt version**.
            This is useful for storing version-specific information, such as the author of
            the changes. Optional.
        tags: A dictionary of tags associated with the entire prompt. This is different from
            the `version_metadata` as it is not tied to a specific version of the prompt,
            but to the prompt as a whole. For example, you can use tags to add an application
            name for which the prompt is created. Since the application uses the prompt in
            multiple versions, it makes sense to use tags instead of version-specific metadata.
            Optional.

    Returns:
        A :py:class:`Prompt <mlflow.entities.Prompt>` object that was created.

    Example:

    .. code-block:: python

        import mlflow

        # Register a new prompt
        mlflow.register_prompt(
            name="my_prompt",
            template="Respond to the user's message as a {{style}} AI.",
            version_metadata={"author": "Alice"},
        )

        # Load the prompt from the registry
        prompt = mlflow.load_prompt("my_prompt")

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
            name="my_prompt",
            template="Respond to the user's message as a {{style}} AI. {{greeting}}",
            commit_message="Add a greeting to the prompt.",
            version_metadata={"author": "Bob"},
        )
    """
    return MlflowClient().register_prompt(
        name=name,
        template=template,
        commit_message=commit_message,
        tags=tags,
        version_metadata=version_metadata,
    )


@experimental
@require_prompt_registry
def load_prompt(name_or_uri: str, version: Optional[int] = None) -> Prompt:
    """
    Load a :py:class:`Prompt <mlflow.entities.Prompt>` from the MLflow Prompt Registry.

    The prompt can be specified by name and version, or by URI.

    Args:
        name_or_uri: The name of the prompt, or the URI in the format "prompts:/name/version".
        version: The version of the prompt. If not specified, the latest version will be loaded.

    Example:

    .. code-block:: python

        import mlflow

        # Load the latest version of the prompt
        prompt = mlflow.load_prompt("my_prompt")

        # Load a specific version of the prompt
        prompt = mlflow.load_prompt("my_prompt", version=1)

        # Load a specific version of the prompt by URI
        prompt = mlflow.load_prompt(uri="prompts:/my_prompt/1")

        # Load a prompt version with an alias "production"
        prompt = mlflow.load_prompt("prompts:/my_prompt@production")

    """
    client = MlflowClient()
    prompt = client.load_prompt(name_or_uri=name_or_uri, version=version)

    # If there is an active MLflow run, associate the prompt with the run
    if run := active_run():
        client.log_prompt(run.info.run_id, f"prompts:/{prompt.name}/{prompt.version}")

    return prompt


@experimental
@require_prompt_registry
def delete_prompt(name: str, version: int) -> Prompt:
    """
    Delete a :py:class:`Prompt <mlflow.entities.Prompt>` from the MLflow Prompt Registry.

    Args:
        name: The name of the prompt.
        version: The version of the prompt to delete.
    """
    return MlflowClient().delete_prompt(name=name, version=version)


@experimental
@require_prompt_registry
def set_prompt_alias(name: str, alias: str, version: int) -> Prompt:
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

    return MlflowClient().set_prompt_alias(name=name, version=version, alias=alias)


@experimental
@require_prompt_registry
def delete_prompt_alias(name: str, alias: str) -> Prompt:
    """
    Delete an alias for a :py:class:`Prompt <mlflow.entities.Prompt>` in the MLflow Prompt Registry.

    Args:
        name: The name of the prompt.
        alias: The alias to delete for the prompt.
    """
    return MlflowClient().delete_prompt_alias(name=name, alias=alias)
