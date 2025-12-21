import logging
import os
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable

import mlflow
from mlflow.data.dataset import Dataset
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.evaluation_dataset import EvaluationDataset as EntityEvaluationDataset
from mlflow.entities.logged_model_input import LoggedModelInput
from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets.evaluation_dataset import EvaluationDataset
from mlflow.genai.evaluation.constant import InputDatasetColumn
from mlflow.genai.evaluation.session_utils import validate_session_level_evaluation_inputs
from mlflow.genai.evaluation.utils import (
    _convert_to_eval_set,
    _get_eval_data_size_and_fields,
)
from mlflow.genai.scorers import Scorer
from mlflow.genai.scorers.builtin_scorers import BuiltInScorer
from mlflow.genai.scorers.validation import valid_data_for_builtin_scorers, validate_scorers
from mlflow.genai.utils.display_utils import display_evaluation_output
from mlflow.genai.utils.trace_utils import convert_predict_fn
from mlflow.models.evaluation.base import (
    EvaluationResult,
    _is_model_deployment_endpoint_uri,
    _start_run_or_reuse_active_run,
)
from mlflow.models.evaluation.utils.trace import configure_autologging_for_evaluation
from mlflow.telemetry.events import GenAIEvaluateEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.telemetry.utils import _log_error
from mlflow.tracing.constant import (
    DATABRICKS_OPTIONS_KEY,
    DATABRICKS_OUTPUT_KEY,
    RETURN_TRACE_OPTION_KEY,
)
from mlflow.tracing.utils.copy import copy_trace_to_experiment
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.fluent import _get_experiment_id, _set_active_model
from mlflow.utils.mlflow_tags import MLFLOW_RUN_IS_EVALUATION

if TYPE_CHECKING:
    from mlflow.genai.evaluation.utils import EvaluationDatasetTypes


logger = logging.getLogger(__name__)


def evaluate(
    data: "EvaluationDatasetTypes",
    scorers: list[Scorer],
    predict_fn: Callable[..., Any] | None = None,
    model_id: str | None = None,
) -> EvaluationResult:
    """
    Evaluate the performance of a generative AI model/application using specified
    data and scorers.

    This function allows you to evaluate a model's performance on a given dataset
    using various scoring criteria. It supports both built-in scorers provided by
    MLflow and custom scorers. The evaluation results include metrics and detailed
    per-row assessments.

    There are three different ways to use this function:

    **1. Use Traces to evaluate the model/application.**

    The `data` parameter takes a DataFrame with `trace` column, which contains a
    single trace object corresponding to the prediction for the row. This dataframe
    is easily obtained from the existing traces stored in MLflow, by using the
    :py:func:`mlflow.search_traces` function.

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import Correctness, Safety
        import pandas as pd

        # model_id is a string starting with "m-", e.g. "m-074689226d3b40bfbbdf4c3ff35832cd"
        trace_df = mlflow.search_traces(model_id="<my-model-id>")

        mlflow.genai.evaluate(
            data=trace_df,
            scorers=[Correctness(), Safety()],
        )

    Built-in scorers will understand the model inputs, outputs, and other intermediate
    information e.g. retrieved context, from the trace object. You can also access to
    the trace object from the custom scorer function by using the `trace` parameter.

    .. code-block:: python

        from mlflow.genai.scorers import scorer


        @scorer
        def faster_than_one_second(inputs, outputs, trace):
            return trace.info.execution_duration < 1000

    **2. Use DataFrame or dictionary with "inputs", "outputs", "expectations" columns.**

    Alternatively, you can pass inputs, outputs, and expectations (ground truth) as
    a column in the dataframe (or equivalent list of dictionaries).

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import Correctness
        import pandas as pd

        data = pd.DataFrame(
            [
                {
                    "inputs": {"question": "What is MLflow?"},
                    "outputs": "MLflow is an ML platform",
                    "expectations": "MLflow is an ML platform",
                },
                {
                    "inputs": {"question": "What is Spark?"},
                    "outputs": "I don't know",
                    "expectations": "Spark is a data engine",
                },
            ]
        )

        mlflow.genai.evaluate(
            data=data,
            scorers=[Correctness()],
        )

    **3. Pass `predict_fn` and input samples (and optionally expectations).**

    If you want to generate the outputs and traces on-the-fly from your input samples,
    you can pass a callable to the `predict_fn` parameter. In this case, MLflow will
    pass the inputs to the `predict_fn` as keyword arguments. Therefore, the "inputs"
    column must be a dictionary with the parameter names as keys.

    .. code-block:: python

        import mlflow
        from mlflow.genai.scorers import Correctness, Safety
        import openai

        # Create a dataframe with input samples
        data = pd.DataFrame(
            [
                {"inputs": {"question": "What is MLflow?"}},
                {"inputs": {"question": "What is Spark?"}},
            ]
        )


        # Define a predict function to evaluate. The "inputs" column will be
        # passed to the prediction function as keyword arguments.
        def predict_fn(question: str) -> str:
            response = openai.OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": question}],
            )
            return response.choices[0].message.content


        mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=[Correctness(), Safety()],
        )

    Args:
        data: Dataset for the evaluation. Must be one of the following formats:

            * An EvaluationDataset entity
            * Pandas DataFrame
            * Spark DataFrame
            * List of dictionaries
            * List of `Trace` objects

            The dataset must include either of the following columns:

            1. `trace` column that contains a single trace object corresponding
                to the prediction for the row.

                If this column is present, MLflow extracts inputs, outputs, assessments,
                and other intermediate information e.g. retrieved context, from the trace
                object and uses them for scoring. When this column is present, the
                `predict_fn` parameter must not be provided.

            2. `inputs`, `outputs`, `expectations` columns.

                Alternatively, you can pass inputs, outputs, and expectations(ground
                truth) as a column in the dataframe (or equivalent list of dictionaries).

                - inputs (required): Column containing inputs for evaluation. The value
                  must be a dictionary. When `predict_fn` is provided, MLflow will pass
                  the inputs to the `predict_fn` as keyword arguments. For example,

                  * predict_fn: `def predict_fn(question: str, context: str) -> str`
                  * inputs: `{"question": "What is MLflow?", "context": "MLflow is an ML platform"}`
                  * `predict_fn` will receive "What is MLflow?" as the first argument
                    (`question`) and "MLflow is an ML platform" as the second argument (`context`)

                - outputs (optional): Column containing model or app outputs.
                  If this column is present, `predict_fn` must not be provided.

                - expectations (optional): Column containing a dictionary of ground truths.

            For list of dictionaries, each dict should follow the above schema.

            Optional columns:
                - tags (optional): Column containing a dictionary of tags. The tags will be logged
                                   to the respective traces.

        scorers: A list of Scorer objects that produces evaluation scores from
            inputs, outputs, and other additional contexts. MLflow provides pre-defined
            scorers, but you can also define custom ones.

        predict_fn: The target function to be evaluated. The specified function will be
            executed for each row in the input dataset, and outputs will be used for
            scoring.

            The function must emit a single trace per call. If it doesn't, decorate
            the function with @mlflow.trace decorator to ensure a trace to be emitted.

            Both synchronous and asynchronous (async def) functions are supported. Async
            functions are automatically detected and wrapped to run synchronously with a
            configurable timeout (default: 300 seconds). Set the timeout using the
            MLFLOW_GENAI_EVAL_ASYNC_TIMEOUT environment variable.

        model_id: Optional model identifier (e.g. "m-074689226d3b40bfbbdf4c3ff35832cd")
            to associate with the evaluation results. Can be also set globally via the
            :py:func:`mlflow.set_active_model` function.

    Returns:
        An :py:class:`mlflow.models.EvaluationResult~` object.

    Note:
        Certain advanced features of this function are only supported on Databricks.
        The tracking URI must be set to Databricks to use these features.

    .. warning::

        This function is not thread-safe. Please do not use it in multi-threaded
        environments.
    """
    result, _ = _run_harness(data, scorers, predict_fn, model_id)
    return result


@record_usage_event(GenAIEvaluateEvent)
def _run_harness(data, scorers, predict_fn, model_id) -> tuple[EvaluationResult, dict[str, Any]]:
    """
    Internal harness for running evaluation.

    Returns:
        A tuple containing:
            - EvaluationResult: The evaluation result object with metrics and assessments
            - dict: Telemetry data dictionary containing evaluation metadata (data size, fields,
              etc.). This is used by the @record_usage_event decorator.
    """
    from mlflow.genai.evaluation import harness

    is_managed_dataset = isinstance(data, (EvaluationDataset, EntityEvaluationDataset))

    scorers = validate_scorers(scorers)

    # Validate session-level input if session-level scorers are present
    validate_session_level_evaluation_inputs(scorers, predict_fn)

    df = _convert_to_eval_set(data)

    builtin_scorers = [scorer for scorer in scorers if isinstance(scorer, BuiltInScorer)]
    valid_data_for_builtin_scorers(df, builtin_scorers, predict_fn)

    sample_input = df.iloc[0][InputDatasetColumn.INPUTS]

    # Only check 'inputs' column when it is not derived from the trace object
    if "trace" not in df.columns and not isinstance(sample_input, dict):
        raise MlflowException.invalid_parameter_value(
            "The 'inputs' column must be a dictionary of field names and values. "
            "For example: {'query': 'What is MLflow?'}"
        )

    # If the input dataset is a managed dataset, we pass the original dataset
    # to the evaluate function to preserve metadata like dataset name.
    data = data if is_managed_dataset else df

    if predict_fn:
        predict_fn = convert_predict_fn(predict_fn=predict_fn, sample_input=sample_input)

    # NB: The "RAG_EVAL_MAX_WORKERS" env var is used in the DBX agent harness, but is
    # deprecated in favor of the new "MLFLOW_GENAI_EVAL_MAX_WORKERS" env var. The old
    # one is not publicly documented, but we keep it for backward compatibility.
    if "RAG_EVAL_MAX_WORKERS" in os.environ:
        logger.warning(
            "The `RAG_EVAL_MAX_WORKERS` environment variable is deprecated. "
            f"Please use `{MLFLOW_GENAI_EVAL_MAX_WORKERS.name}` instead."
        )
        os.environ[MLFLOW_GENAI_EVAL_MAX_WORKERS.name] = os.environ["RAG_EVAL_MAX_WORKERS"]

    if isinstance(data, (EvaluationDataset, EntityEvaluationDataset)):
        mlflow_dataset = data
        df = data.to_df()
    else:
        # Use default name for evaluation dataset when converting from DataFrame
        mlflow_dataset = mlflow.data.from_pandas(df=data, name="dataset")
        df = data

    try:
        telemetry_data = _get_eval_data_size_and_fields(df)
    except Exception:
        _log_error("Failed to get evaluation data size and fields for GenAIEvaluateEvent")
        telemetry_data = {}

    with (
        _start_run_or_reuse_active_run() as run_id,
        _set_active_model(model_id=model_id) if model_id else nullcontext(),
        # NB: Auto-logging should be enabled outside the thread pool to avoid race conditions.
        configure_autologging_for_evaluation(enable_tracing=True),
    ):
        _log_dataset_input(mlflow_dataset, run_id, model_id)

        # NB: Set this tag before run finishes to suppress the generic run URL printing.
        MlflowClient().set_tag(run_id, MLFLOW_RUN_IS_EVALUATION, "true")

        result = harness.run(
            predict_fn=predict_fn,
            eval_df=df,
            scorers=scorers,
            run_id=run_id,
        )

    try:
        display_evaluation_output(run_id)
    except Exception:
        logger.debug("Failed to display summary and usage instructions", exc_info=True)

    return result, telemetry_data


def _log_dataset_input(
    data: Dataset,
    run_id: str,
    model_id: str | None = None,
):
    client = MlflowClient()
    dataset_input = DatasetInput(dataset=data._to_mlflow_entity())
    client.log_inputs(
        run_id=run_id,
        datasets=[dataset_input],
        models=[LoggedModelInput(model_id=model_id)] if model_id else None,
    )


def to_predict_fn(endpoint_uri: str) -> Callable[..., Any]:
    """
    Convert an endpoint URI to a predict function.

    Args:
        endpoint_uri: The endpoint URI to convert.

    Returns:
        A predict function that can be used to make predictions.

    Example:

        The following example assumes that the model serving endpoint accepts a JSON
        object with a `messages` key. Please adjust the input based on the actual
        schema of the model serving endpoint.

        .. code-block:: python

            from mlflow.genai.scorers import get_all_scorers

            data = [
                {
                    "inputs": {
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "What is MLflow?"},
                        ]
                    }
                },
                {
                    "inputs": {
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "What is Spark?"},
                        ]
                    }
                },
            ]
            predict_fn = mlflow.genai.to_predict_fn("endpoints:/chat")
            mlflow.genai.evaluate(
                data=data,
                predict_fn=predict_fn,
                scorers=get_all_scorers(),
            )

        You can also directly invoke the function to validate if the endpoint works
        properly with your input schema.

        .. code-block:: python

            predict_fn(**data[0]["inputs"])
    """
    if not _is_model_deployment_endpoint_uri(endpoint_uri):
        raise ValueError(
            f"Invalid endpoint URI: {endpoint_uri}. The endpoint URI must be a valid model "
            f"deployment endpoint URI."
        )

    from mlflow.deployments import get_deploy_client
    from mlflow.metrics.genai.model_utils import _parse_model_uri

    client = get_deploy_client("databricks")
    _, endpoint = _parse_model_uri(endpoint_uri)
    endpoint_info = client.get_endpoint(endpoint)

    # Databricks Foundation Model API does not allow passing "databricks_options" in the payload,
    # so we need to handle this case separately.
    is_fmapi = False
    if isinstance(endpoint_info, dict):
        is_fmapi = endpoint_info.get("endpoint_type") == "FOUNDATION_MODEL_API"

    # NB: Wrap the function to show better docstring and change signature to `model_inputs`
    #   to unnamed keyword arguments. This is necessary because we pass input samples as
    #   keyword arguments to the predict function.
    def predict_fn(**kwargs):
        start_time_ms = int(time.time_ns() / 1e6)
        # Inject `{"databricks_options": {"return_trace": True}}` to the input payload
        # to return the trace in the response.
        databricks_options = {DATABRICKS_OPTIONS_KEY: {RETURN_TRACE_OPTION_KEY: True}}
        payload = kwargs if is_fmapi else {**kwargs, **databricks_options}
        result = client.predict(endpoint=endpoint, inputs=payload)
        end_time_ms = int(time.time_ns() / 1e6)

        # If the endpoint returns a trace, check if we need to copy it to the current experiment.
        if trace_dict := result.pop(DATABRICKS_OUTPUT_KEY, {}).get("trace"):
            # Check if the trace is already in the current experiment (dual-write mode).
            # This happens when the endpoint has MLFLOW_EXPERIMENT_ID set and writes
            # traces to both inference table and MLflow experiment.
            trace_experiment_id = (
                (info := trace_dict.get("info"))
                and (trace_loc := info.get("trace_location"))
                and (ml_exp := trace_loc.get("mlflow_experiment"))
                and ml_exp.get("experiment_id")
            )
            current_experiment_id = _get_experiment_id()

            # If the trace is already in the current experiment, we can reuse it
            # instead of copying it again (avoiding duplicate traces).
            if trace_experiment_id and trace_experiment_id == current_experiment_id:
                logger.debug(
                    "Trace from endpoint is already in the current experiment "
                    f"(experiment_id={current_experiment_id}). Reusing existing trace "
                    "instead of copying."
                )
                return result

            # Otherwise, copy the trace to the current experiment.
            try:
                copy_trace_to_experiment(trace_dict)
                return result
            except Exception:
                logger.debug(
                    "Failed to copy trace from the endpoint response to the current experiment. "
                    "Trace will only have a root span with request and response.",
                    exc_info=True,
                )

        # If the endpoint doesn't return a trace, manually create a trace with request/response.
        span = mlflow.start_span_no_context(
            name="predict",
            inputs=kwargs,
            start_time_ns=start_time_ms * 1000000,
        )
        span.end(
            outputs=result,
            end_time_ns=end_time_ms * 1000000,
        )
        return result

    predict_fn.__doc__ = f"""
A wrapper function for invoking the model serving endpoint `{endpoint_uri}`.

Args:
    **kwargs: The input samples to be passed to the model serving endpoint.
        For example, if the endpoint accepts a JSON object with a `messages` key,
        the function also expects to get `messages` as an argument.
    """
    return predict_fn
