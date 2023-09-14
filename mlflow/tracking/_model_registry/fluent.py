from typing import Any, Dict, List, Optional

from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import ALREADY_EXISTS, RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.store.model_registry import (
    SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
)
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.client import MlflowClient
from mlflow.utils import get_results_from_paginated_fn
from mlflow.utils.logging_utils import eprint


def register_model(
    model_uri,
    name,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    *,
    tags: Optional[Dict[str, Any]] = None,
) -> ModelVersion:
    """
    Create a new model version in model registry for the model files specified by ``model_uri``.
    Note that this method assumes the model registry backend URI is the same as that of the
    tracking backend.

    :param model_uri: URI referring to the MLmodel directory. Use a ``runs:/`` URI if you want to
                      record the run ID with the model in model registry. ``models:/`` URIs are
                      currently not supported.
    :param name: Name of the registered model under which to create a new model version. If a
                 registered model with the given name does not exist, it will be created
                 automatically.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.
    :param tags: A dictionary of key-value pairs that are converted into
                 :py:class:`mlflow.entities.model_registry.ModelVersionTag` objects.
    :return: Single :py:class:`mlflow.entities.model_registry.ModelVersion` object created by
             backend.

    .. testcode:: python
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
    tags: Optional[Dict[str, Any]] = None,
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
                "Registered model '%s' already exists. Creating a new version of this model..."
                % name
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
    order_by: Optional[List[str]] = None,
) -> List[RegisteredModel]:
    """
    Search for registered models that satisfy the filter criteria.

    :param filter_string: Filter query string
        (e.g., ``"name = 'a_model_name' and tag.key = 'value1'"``),
        defaults to searching for all registered models. The following identifiers, comparators,
        and logical operators are supported.

        Identifiers
          - ``name``: registered model name.
          - ``tags.<tag_key>``: registered model tag. If ``tag_key`` contains spaces, it must be
            wrapped with backticks (e.g., ``"tags.`extra key`"``).

        Comparators
          - ``=``: Equal to.
          - ``!=``: Not equal to.
          - ``LIKE``: Case-sensitive pattern match.
          - ``ILIKE``: Case-insensitive pattern match.

        Logical operators
          - ``AND``: Combines two sub-queries and returns True if both of them are True.

    :param max_results: If passed, specifies the maximum number of models desired. If not
                        passed, all models will be returned.
    :param order_by: List of column names with ASC|DESC annotation, to be used for ordering
                     matching search results.
    :return: A list of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
             that satisfy the search expressions.

    .. testcode:: python
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
    order_by: Optional[List[str]] = None,
) -> List[ModelVersion]:
    """
    Search for model versions that satisfy the filter criteria.

    :param filter_string: Filter query string
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

    :param max_results: If passed, specifies the maximum number of models desired. If not
                        passed, all models will be returned.
    :param order_by: List of column names with ASC|DESC annotation, to be used for ordering
                    matching search results.
    :return: A list of :py:class:`mlflow.entities.model_registry.ModelVersion` objects
            that satisfy the search expressions.

    .. testcode:: python
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
