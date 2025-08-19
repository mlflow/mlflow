import logging
import os
import time
import warnings
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable

import mlflow
from mlflow.data.dataset import Dataset
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.logged_model_input import LoggedModelInput
from mlflow.environment_variables import MLFLOW_GENAI_EVAL_MAX_WORKERS
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import EvaluationDataset
from mlflow.genai.evaluation.constant import InputDatasetColumn
from mlflow.genai.evaluation.utils import (
    _convert_scorer_to_legacy_metric,
    _convert_to_eval_set,
)
from mlflow.genai.scorers import Scorer
from mlflow.genai.scorers.builtin_scorers import GENAI_CONFIG_NAME, BuiltInScorer
from mlflow.genai.scorers.validation import valid_data_for_builtin_scorers, validate_scorers
from mlflow.genai.utils.display_utils import display_evaluation_output
from mlflow.genai.utils.trace_utils import clean_up_extra_traces, convert_predict_fn
from mlflow.models.evaluation.base import (
    EvaluationResult,
    _is_model_deployment_endpoint_uri,
    _start_run_or_reuse_active_run,
)
from mlflow.models.evaluation.utils.trace import configure_autologging_for_evaluation
from mlflow.telemetry.events import GenAIEvaluateEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.tracing.constant import (
    DATABRICKS_OPTIONS_KEY,
    DATABRICKS_OUTPUT_KEY,
    RETURN_TRACE_OPTION_KEY,
)
from mlflow.tracing.utils.copy import copy_trace_to_experiment
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.fluent import _set_active_model
from mlflow.utils.annotations import experimental
from mlflow.utils.mlflow_tags import MLFLOW_RUN_IS_EVALUATION
from mlflow.utils.uri import is_databricks_uri

if TYPE_CHECKING:
    from mlflow.genai.evaluation.utils import EvaluationDatasetTypes


logger = logging.getLogger(__name__)


@experimental(version="3.0.0")
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

        scorers: A list of Scorer objects that produces evaluation scores from
            inputs, outputs, and other additional contexts. MLflow provides pre-defined
            scorers, but you can also define custom ones.

        predict_fn: The target function to be evaluated. The specified function will be
            executed for each row in the input dataset, and outputs will be used for
            scoring.

            The function must emit a single trace per call. If it doesn't, decorate
            the function with @mlflow.trace decorator to ensure a trace to be emitted.

        model_id: Optional model identifier (e.g. "m-074689226d3b40bfbbdf4c3ff35832cd")
            to associate with the evaluation results. Can be also set globally via the
            :py:func:`mlflow.set_active_model` function.

    Returns:
        An :py:class:`mlflow.models.EvaluationResult~` object.

    Note:
        This function is only supported on Databricks. The tracking URI must be
        set to Databricks.

    .. warning::

        This function is not thread-safe. Please do not use it in multi-threaded
        environments.
    """
    is_managed_dataset = isinstance(data, EvaluationDataset)

    scorers = validate_scorers(scorers)
    # convert into a pandas dataframe with expected evaluation set schema
    df = data.to_df() if is_managed_dataset else _convert_to_eval_set(data)

    builtin_scorers = [scorer for scorer in scorers if isinstance(scorer, BuiltInScorer)]
    valid_data_for_builtin_scorers(df, builtin_scorers, predict_fn)

    # "request" column must exist after conversion
    input_key = "inputs" if is_managed_dataset else "request"
    sample_input = df.iloc[0][input_key]

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

    eval_start_time = int(time.time() * 1000)

    if is_databricks_uri(mlflow.get_tracking_uri()):
        result = _evaluate_dbx(data, scorers, predict_fn, model_id)
    else:
        result = _evaluate_oss(data, scorers, predict_fn, model_id)

    # Clean up noisy traces generated during evaluation
    clean_up_extra_traces(result.run_id, eval_start_time)

    return result


@record_usage_event(GenAIEvaluateEvent)
def _evaluate_oss(data, scorers, predict_fn, model_id):
    from mlflow.genai.evaluation import harness

    if isinstance(data, EvaluationDataset):
        mlflow_dataset = data
        df = data.to_df()
    else:
        # Rename 'request' / 'response' column back to 'inputs' / 'outputs'.
        # This is a temporary hack to avoid branching _convert_to_eval_set()
        # into OSS and DBX implementation.
        data = data.rename(
            columns={
                "request": InputDatasetColumn.INPUTS,
                "response": InputDatasetColumn.OUTPUTS,
            }
        )
        # Use default name for evaluation dataset when converting from DataFrame
        mlflow_dataset = mlflow.data.from_pandas(df=data, name="dataset")
        df = data

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

    return result


def _evaluate_dbx(data, scorers, predict_fn, model_id):
    """In Databricks, we run GenAI evaluation using databricks-agents package and
    the mlflow.evaluate() function. This is a temporary migration state and we will
    eventually unify this into OSS flow.
    """
    import pandas as pd

    # NB: The "RAG_EVAL_MAX_WORKERS" env var is used in the DBX agent harness, but is
    # deprecated in favor of the new "MLFLOW_GENAI_EVAL_MAX_WORKERS" env var. The old
    # one is not publicly documented, but we keep it for backward compatibility.
    if MLFLOW_GENAI_EVAL_MAX_WORKERS.is_set() and "RAG_EVAL_MAX_WORKERS" not in os.environ:
        os.environ["RAG_EVAL_MAX_WORKERS"] = str(MLFLOW_GENAI_EVAL_MAX_WORKERS.get())
    elif "RAG_EVAL_MAX_WORKERS" in os.environ:
        logger.warning(
            "The `RAG_EVAL_MAX_WORKERS` environment variable is deprecated. "
            "Please use `MLFLOW_GENAI_EVAL_MAX_WORKERS` instead."
        )

    if isinstance(data, pd.DataFrame):
        from mlflow.data.evaluation_dataset import convert_data_to_mlflow_dataset

        data = convert_data_to_mlflow_dataset(data=data, name="dataset")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Hint: Inferred schema contains integer column\(s\).*",
            category=UserWarning,
        )
        # Suppress numpy warning about ragged nested sequences. This is raised when passing
        # a dataset that contains complex object to mlflow.evaluate(). MLflow converts data
        # into numpy array to compute dataset digest, which triggers the warning.
        warnings.filterwarnings(
            "ignore",
            message=r"Creating an ndarray from ragged nested sequences",
            module="mlflow.data.evaluation_dataset",
        )

        return mlflow.models.evaluate(
            model=predict_fn,
            data=data,
            evaluator_config={GENAI_CONFIG_NAME: {"metrics": []}},  # Turn off the default metrics
            # Scorers are passed to the eval harness as extra metrics
            extra_metrics=[_convert_scorer_to_legacy_metric(_scorer) for _scorer in scorers],
            model_type=GENAI_CONFIG_NAME,
            model_id=model_id,
            _called_from_genai_evaluate=True,
        )


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


@experimental(version="3.0.0")
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

        # If the endpoint returns a trace, copy it to the current experiment.
        if trace_dict := result.pop(DATABRICKS_OUTPUT_KEY, {}).get("trace"):
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
        mlflow.log_trace(
            name="predict",
            request=kwargs,
            response=result,
            start_time_ms=start_time_ms,
            execution_time_ms=end_time_ms - start_time_ms,
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
