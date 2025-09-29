import json
import logging
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from inspect import Parameter, Signature
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.metrics.base import MetricValue
from mlflow.metrics.genai import model_utils
from mlflow.metrics.genai.base import EvaluationExample
from mlflow.metrics.genai.prompt_template import PromptTemplate
from mlflow.metrics.genai.utils import _get_default_model, _get_latest_metric_version
from mlflow.models import EvaluationMetric, make_metric
from mlflow.models.evaluation.base import _make_metric
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    UNAUTHENTICATED,
    ErrorCode,
)
from mlflow.utils.class_utils import _get_class_from_string
from mlflow.version import VERSION

_logger = logging.getLogger(__name__)

_GENAI_CUSTOM_METRICS_FILE_NAME = "genai_custom_metrics.json"
_PROMPT_FORMATTING_WRAPPER = """

You must return the following fields in your response in two lines, one below the other:
score: Your numerical score based on the rubric
justification: Your reasoning for giving this score

Do not add additional new lines. Do not add any other fields."""


def _format_args_string(grading_context_columns: list[str] | None, eval_values, indx) -> str:
    import pandas as pd

    args_dict = {}
    for arg in grading_context_columns:
        if arg in eval_values:
            args_dict[arg] = (
                eval_values[arg].iloc[indx]
                if isinstance(eval_values[arg], pd.Series)
                else eval_values[arg][indx]
            )
        else:
            raise MlflowException(
                f"{arg} does not exist in the eval function {list(eval_values.keys())}."
            )

    return (
        ""
        if args_dict is None or len(args_dict) == 0
        else (
            "Additional information used by the model:\n"
            + "\n".join(
                [f"key: {arg}\nvalue:\n{arg_value}" for arg, arg_value in args_dict.items()]
            )
        )
    )


# Function to extract Score and Justification
def _extract_score_and_justification(text):
    if text:
        text = re.sub(r"score", "score", text, flags=re.IGNORECASE)
        text = re.sub(r"justification", "justification", text, flags=re.IGNORECASE)
        # Attempt to parse JSON
        try:
            data = json.loads(text)
            score = int(data.get("score"))
            justification = data.get("justification")
        except json.JSONDecodeError:
            # If parsing fails, use regex
            if (match := re.search(r"score: (\d+),?\s*justification: (.+)", text)) or (
                match := re.search(r"\s*score:\s*(\d+)\s*justification:\s*(.+)", text, re.DOTALL)
            ):
                score = int(match.group(1))
                justification = match.group(2)
            else:
                score = None
                justification = f"Failed to extract score and justification. Raw output: {text}"

        if not isinstance(score, (int, float)) or not isinstance(justification, str):
            return None, f"Failed to extract score and justification. Raw output: {text}"

        return score, justification

    return None, None


def _score_model_on_one_payload(
    payload: str,
    eval_model: str,
    parameters: dict[str, Any] | None,
    extra_headers: dict[str, str] | None = None,
    proxy_url: str | None = None,
):
    try:
        # If the endpoint does not specify type, default to chat format
        endpoint_type = model_utils.get_endpoint_type(eval_model) or "llm/v1/chat"
        raw_result = model_utils.score_model_on_payload(
            eval_model, payload, parameters, extra_headers, proxy_url, endpoint_type
        )
        return _extract_score_and_justification(raw_result)
    except ImportError:
        raise
    except MlflowException as e:
        if e.error_code in [
            ErrorCode.Name(BAD_REQUEST),
            ErrorCode.Name(UNAUTHENTICATED),
            ErrorCode.Name(INVALID_PARAMETER_VALUE),
        ]:
            raise
        else:
            return None, f"Failed to score model on payload. Error: {e!s}"
    except Exception as e:
        return None, f"Failed to score model on payload. Error: {e!s}"


def _score_model_on_payloads(
    grading_payloads, model, parameters, headers, proxy_url, max_workers
) -> tuple[list[int], list[str]]:
    scores = [None] * len(grading_payloads)
    justifications = [None] * len(grading_payloads)
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="MlflowGenAiScoring"
    ) as executor:
        futures = {
            executor.submit(
                _score_model_on_one_payload,
                payload,
                model,
                parameters,
                headers,
                proxy_url,
            ): indx
            for indx, payload in enumerate(grading_payloads)
        }

        as_comp = as_completed(futures)
        try:
            from tqdm.auto import tqdm

            as_comp = tqdm(as_comp, total=len(futures))
        except ImportError:
            pass

        for future in as_comp:
            indx = futures[future]
            score, justification = future.result()
            scores[indx] = score
            justifications[indx] = justification

    return scores, justifications


def _get_aggregate_results(scores, aggregations):
    # loop over the aggregations and compute the aggregate results on the scores
    def aggregate_function(aggregate_option, scores):
        import numpy as np

        options = {
            "min": np.min,
            "max": np.max,
            "mean": np.mean,
            "median": np.median,
            "variance": np.var,
            "p90": lambda x: np.percentile(x, 90) if x else None,
        }

        if aggregate_option not in options:
            raise MlflowException(
                message=f"Invalid aggregate option {aggregate_option}.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        return options[aggregate_option](scores)

    scores_for_aggregation = [score for score in scores if score is not None]

    return (
        {option: aggregate_function(option, scores_for_aggregation) for option in aggregations}
        if aggregations is not None
        else {}
    )


def make_genai_metric_from_prompt(
    name: str,
    judge_prompt: str | None = None,
    model: str | None = _get_default_model(),
    parameters: dict[str, Any] | None = None,
    aggregations: list[str] | None = None,
    greater_is_better: bool = True,
    max_workers: int = 10,
    metric_metadata: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    proxy_url: str | None = None,
) -> EvaluationMetric:
    """
    Create a genai metric used to evaluate LLM using LLM as a judge in MLflow. This produces
    a metric using only the supplied judge prompt without any pre-written system prompt.
    This can be useful for use cases that are not covered by the full grading prompt in any
    ``EvaluationModel`` version.

    Args:
        name: Name of the metric.
        judge_prompt: The entire prompt to be used for the judge model.
            The prompt will be minimally wrapped in formatting instructions to ensure
            scores can be parsed. The prompt may use f-string formatting to include variables.
            Corresponding variables must be passed as keyword arguments into the
            resulting metric's eval function.
        model: (Optional) Model uri of the judge model that will be used to compute the metric,
            e.g., ``openai:/gpt-4``. Refer to the `LLM-as-a-Judge Metrics <https://mlflow.org/docs/latest/llms/llm-evaluate/index.html#selecting-the-llm-as-judge-model>`_
            documentation for the supported model types and their URI format.
        parameters: (Optional) Parameters for the LLM used to compute the metric. By default, we
            set the temperature to 0.0, max_tokens to 200, and top_p to 1.0. We recommend
            setting the temperature to 0.0 for the LLM used as a judge to ensure consistent results.
        aggregations: (Optional) The list of options to aggregate the scores. Currently supported
            options are: min, max, mean, median, variance, p90.
        greater_is_better: (Optional) Whether the metric is better when it is greater.
        max_workers: (Optional) The maximum number of workers to use for judge scoring.
            Defaults to 10 workers.
        metric_metadata: (Optional) Dictionary of metadata to be attached to the
            EvaluationMetric object. Useful for model evaluators that require additional
            information to determine how to evaluate this metric.
        extra_headers: (Optional) Additional headers to be passed to the judge model.
        proxy_url: (Optional) Proxy URL to be used for the judge model. This is useful when the
            judge model is served via a proxy endpoint, not directly via LLM provider services.
            If not specified, the default URL for the LLM provider will be used
            (e.g., https://api.openai.com/v1/chat/completions for OpenAI chat models).

    Returns:
        A metric object.

    .. code-block:: python
        :test:
        :caption: Example for creating a genai metric

        import pandas as pd
        import mlflow
        from mlflow.metrics.genai import make_genai_metric_from_prompt

        metric = make_genai_metric_from_prompt(
            name="ease_of_understanding",
            judge_prompt=(
                "You must evaluate the output of a bot based on how easy it is to "
                "understand its outputs."
                "Evaluate the bot's output from the perspective of a layperson."
                "The bot was provided with this input: {input} and this output: {output}."
            ),
            model="openai:/gpt-4",
            parameters={"temperature": 0.0},
            aggregations=["mean", "variance", "p90"],
            greater_is_better=True,
        )

        data = pd.DataFrame(
            {
                "input": ["Where is the capital of France."],
                "ground_truth": ["Paris"],
                "output": ["The capital of France is Paris."],
            }
        )

        mlflow.evaluate(
            data=data,
            targets="ground_truth",
            predictions="output",
            evaluators="default",
            extra_metrics=[metric],
        )
    """
    import numpy as np

    prompt_template = PromptTemplate([judge_prompt, _PROMPT_FORMATTING_WRAPPER])
    allowed_variables = prompt_template.variables
    additional_variables = list(allowed_variables - {"predictions"})

    # When users create a custom metric using this function,the metric configuration
    # will be serialized and stored as an artifact. This enables us to later deserialize
    # the configuration, allowing users to understand their LLM evaluation results more clearly.
    genai_metric_args = {
        "name": name,
        "judge_prompt": judge_prompt,
        "model": model,
        "parameters": parameters,
        "aggregations": aggregations,
        "greater_is_better": greater_is_better,
        "max_workers": max_workers,
        "metric_metadata": metric_metadata,
        # Record the mlflow version for serialization in case the function signature changes later
        "mlflow_version": VERSION,
        "fn_name": make_genai_metric_from_prompt.__name__,
    }

    aggregations = aggregations or ["mean", "variance", "p90"]

    def eval_fn(
        *args,
        **kwargs,
    ) -> MetricValue:
        """
        This is the function that is called when the metric is evaluated.
        Note that default evaluator only passes positional arguments.
        """
        for i, arg in enumerate(args):
            if i == 0:
                kwargs["predictions"] = arg
            else:
                kwargs[additional_variables[i - 1]] = arg
        if missing_variables := allowed_variables - set(kwargs.keys()):
            raise MlflowException(
                message=f"Missing variable inputs to eval_fn: {missing_variables}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        kwargs = {k: [v] if np.isscalar(v) else v for k, v in kwargs.items()}
        grading_payloads = pd.DataFrame(kwargs).to_dict(orient="records")
        arg_strings = [prompt_template.format(**payload) for payload in grading_payloads]
        scores, justifications = _score_model_on_payloads(
            arg_strings, model, parameters, extra_headers, proxy_url, max_workers
        )

        aggregate_scores = _get_aggregate_results(scores, aggregations)

        return MetricValue(scores, justifications, aggregate_scores)

    # Add `predictions` to the parameters to be compatible with `eval_fn`` interface
    eval_fn_parameters = [
        Parameter(name="predictions", kind=Parameter.POSITIONAL_ONLY),
        *[
            Parameter(name=var, kind=Parameter.POSITIONAL_ONLY, default=None)
            for var in additional_variables
        ],
    ]
    eval_fn.__signature__ = Signature(parameters=eval_fn_parameters)

    return make_metric(
        eval_fn=eval_fn,
        greater_is_better=greater_is_better,
        name=name,
        metric_metadata=metric_metadata,
        genai_metric_args=genai_metric_args,
    )


def make_genai_metric(
    name: str,
    definition: str,
    grading_prompt: str,
    examples: list[EvaluationExample] | None = None,
    version: str | None = _get_latest_metric_version(),
    model: str | None = _get_default_model(),
    grading_context_columns: str | list[str] | None = None,
    include_input: bool = True,
    parameters: dict[str, Any] | None = None,
    aggregations: list[str] | None = None,
    greater_is_better: bool = True,
    max_workers: int = 10,
    metric_metadata: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    proxy_url: str | None = None,
) -> EvaluationMetric:
    """
    Create a genai metric used to evaluate LLM using LLM as a judge in MLflow. The full grading
    prompt is stored in the metric_details field of the ``EvaluationMetric`` object.

    Args:
        name: Name of the metric.
        definition: Definition of the metric.
        grading_prompt: Grading criteria of the metric.
        examples: (Optional) Examples of the metric.
        version: (Optional) Version of the metric. Currently supported versions are: v1.
        model: (Optional) Model uri of the judge model that will be used to compute the metric,
            e.g., ``openai:/gpt-4``. Refer to the `LLM-as-a-Judge Metrics <https://mlflow.org/docs/latest/llms/llm-evaluate/index.html#selecting-the-llm-as-judge-model>`_
            documentation for the supported model types and their URI format.
        grading_context_columns: (Optional) The name of the grading context column, or a list of
            grading context column names, required to compute the metric. The
            ``grading_context_columns`` are used by the LLM as a judge as additional information to
            compute the metric. The columns are extracted from the input dataset or output
            predictions based on ``col_mapping`` in the ``evaluator_config`` passed to
            :py:func:`mlflow.evaluate()`. They can also be the name of other evaluated metrics.
        include_input: (Optional) Whether to include the input
            when computing the metric.
        parameters: (Optional) Parameters for the LLM used to compute the metric. By default, we
            set the temperature to 0.0, max_tokens to 200, and top_p to 1.0. We recommend
            setting the temperature to 0.0 for the LLM used as a judge to ensure consistent results.
        aggregations: (Optional) The list of options to aggregate the scores. Currently supported
            options are: min, max, mean, median, variance, p90.
        greater_is_better: (Optional) Whether the metric is better when it is greater.
        max_workers: (Optional) The maximum number of workers to use for judge scoring.
            Defaults to 10 workers.
        metric_metadata: (Optional) Dictionary of metadata to be attached to the
            EvaluationMetric object. Useful for model evaluators that require additional
            information to determine how to evaluate this metric.
        extra_headers: (Optional) Additional headers to be passed to the judge model.
        proxy_url: (Optional) Proxy URL to be used for the judge model. This is useful when the
            judge model is served via a proxy endpoint, not directly via LLM provider services.
            If not specified, the default URL for the LLM provider will be used
            (e.g., https://api.openai.com/v1/chat/completions for OpenAI chat models).


    Returns:
        A metric object.

    .. code-block:: python
        :test:
        :caption: Example for creating a genai metric

        from mlflow.metrics.genai import EvaluationExample, make_genai_metric

        example = EvaluationExample(
            input="What is MLflow?",
            output=(
                "MLflow is an open-source platform for managing machine "
                "learning workflows, including experiment tracking, model packaging, "
                "versioning, and deployment, simplifying the ML lifecycle."
            ),
            score=4,
            justification=(
                "The definition effectively explains what MLflow is "
                "its purpose, and its developer. It could be more concise for a 5-score.",
            ),
            grading_context={
                "targets": (
                    "MLflow is an open-source platform for managing "
                    "the end-to-end machine learning (ML) lifecycle. It was developed by "
                    "Databricks, a company that specializes in big data and machine learning "
                    "solutions. MLflow is designed to address the challenges that data "
                    "scientists and machine learning engineers face when developing, training, "
                    "and deploying machine learning models."
                )
            },
        )
        metric = make_genai_metric(
            name="answer_correctness",
            definition=(
                "Answer correctness is evaluated on the accuracy of the provided output based on "
                "the provided targets, which is the ground truth. Scores can be assigned based on "
                "the degree of semantic similarity and factual correctness of the provided output "
                "to the provided targets, where a higher score indicates higher degree of accuracy."
            ),
            grading_prompt=(
                "Answer correctness: Below are the details for different scores:"
                "- Score 1: The output is completely incorrect. It is completely different from "
                "or contradicts the provided targets."
                "- Score 2: The output demonstrates some degree of semantic similarity and "
                "includes partially correct information. However, the output still has significant "
                "discrepancies with the provided targets or inaccuracies."
                "- Score 3: The output addresses a couple of aspects of the input accurately, "
                "aligning with the provided targets. However, there are still omissions or minor "
                "inaccuracies."
                "- Score 4: The output is mostly correct. It provides mostly accurate information, "
                "but there may be one or more minor omissions or inaccuracies."
                "- Score 5: The output is correct. It demonstrates a high degree of accuracy and "
                "semantic similarity to the targets."
            ),
            examples=[example],
            version="v1",
            model="openai:/gpt-4",
            grading_context_columns=["targets"],
            parameters={"temperature": 0.0},
            aggregations=["mean", "variance", "p90"],
            greater_is_better=True,
        )
    """
    # When users create a custom metric using this function,the metric configuration
    # will be serialized and stored as an artifact. This enables us to later deserialize
    # the configuration, allowing users to understand their LLM evaluation results more clearly.
    genai_metric_args = {
        "name": name,
        "definition": definition,
        "grading_prompt": grading_prompt,
        "examples": examples,
        "version": version,
        "model": model,
        "grading_context_columns": grading_context_columns,
        "include_input": include_input,
        "parameters": parameters,
        "aggregations": aggregations,
        "greater_is_better": greater_is_better,
        "max_workers": max_workers,
        "metric_metadata": metric_metadata,
        # Record the mlflow version for serialization in case the function signature changes later
        "mlflow_version": VERSION,
        "fn_name": make_genai_metric.__name__,
    }

    aggregations = aggregations or ["mean", "variance", "p90"]
    grading_context_columns = grading_context_columns or []

    if not isinstance(grading_context_columns, list):
        grading_context_columns = [grading_context_columns]

    def process_example(example):
        if example.grading_context is None and len(grading_context_columns) == 0:
            grading_context = {}
        elif isinstance(example.grading_context, dict):
            grading_context = example.grading_context
        else:
            # The grading context is string-like. Assume that it corresponds to the first
            # grading context column and update the example accordingly
            grading_context = {grading_context_columns[0]: example.grading_context}
            example.grading_context = grading_context

        if set(grading_context.keys()) != set(grading_context_columns):
            raise MlflowException.invalid_parameter_value(
                f"Example grading context does not contain required columns.\n"
                f" Example grading context columns: {list(grading_context.keys())}\n"
                f" Required grading context columns: {grading_context_columns}\n"
            )

        if not include_input:
            return EvaluationExample(
                output=example.output,
                score=example.score,
                justification=example.justification,
                grading_context=example.grading_context,
            )

        return example

    if examples is not None:
        examples = [process_example(example) for example in examples]

    class_name = f"mlflow.metrics.genai.prompts.{version}.EvaluationModel"
    try:
        evaluation_model_class_module = _get_class_from_string(class_name)
    except ModuleNotFoundError:
        raise MlflowException(
            f"Failed to find evaluation model for version {version}."
            f" Please check the correctness of the version",
            error_code=INVALID_PARAMETER_VALUE,
        ) from None
    except Exception as e:
        raise MlflowException(
            f"Failed to construct evaluation model {version}. Error: {e!r}",
            error_code=INTERNAL_ERROR,
        ) from None

    evaluation_context = evaluation_model_class_module(
        name,
        definition,
        grading_prompt,
        examples,
        model,
        *(parameters,) if parameters is not None else (),
    ).to_dict()

    def eval_fn(
        predictions: "pd.Series",
        metrics: dict[str, MetricValue],
        inputs: "pd.Series",
        *args,
    ) -> MetricValue:
        """
        This is the function that is called when the metric is evaluated.
        """
        eval_values = dict(zip(grading_context_columns, args))

        outputs = predictions.to_list()
        inputs = inputs.to_list()
        eval_model = evaluation_context["model"]
        eval_parameters = evaluation_context["parameters"]

        # TODO: Save the metric definition in a yaml file for model monitoring

        if not isinstance(eval_model, str):
            raise MlflowException(
                message="The model argument must be a string URI referring to an openai model "
                "(openai:/gpt-4o-mini) or an MLflow Deployments endpoint "
                f"(endpoints:/my-endpoint), passed {eval_model} instead",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # generate grading payloads
        grading_payloads = []
        for indx, (input, output) in enumerate(zip(inputs, outputs)):
            try:
                arg_string = _format_args_string(grading_context_columns, eval_values, indx)
            except Exception as e:
                raise MlflowException(
                    f"Values for grading_context_columns are malformed and cannot be "
                    f"formatted into a prompt for metric '{name}'.\n"
                    f"Required columns: {grading_context_columns}\n"
                    f"Values: {eval_values}\n"
                    f"Error: {e!r}\n"
                    f"Please check the following: \n"
                    "- predictions and targets (if required) are provided correctly\n"
                    "- grading_context_columns are mapped correctly using the evaluator_config "
                    "parameter\n"
                    "- input and output data are formatted correctly."
                )
            grading_payloads.append(
                evaluation_context["eval_prompt"].format(
                    input=(input if include_input else None),
                    output=output,
                    grading_context_columns=arg_string,
                )
            )

        scores = [None] * len(inputs)
        justifications = [None] * len(inputs)

        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="MlflowGenAiEvaluation"
        ) as executor:
            futures = {
                executor.submit(
                    _score_model_on_one_payload,
                    payload,
                    eval_model,
                    eval_parameters,
                    extra_headers,
                    proxy_url,
                ): indx
                for indx, payload in enumerate(grading_payloads)
            }

            as_comp = as_completed(futures)
            try:
                from tqdm.auto import tqdm

                as_comp = tqdm(as_comp, total=len(futures))
            except ImportError:
                pass

            for future in as_comp:
                indx = futures[future]
                score, justification = future.result()
                scores[indx] = score
                justifications[indx] = justification

        aggregate_results = _get_aggregate_results(scores, aggregations)
        return MetricValue(scores, justifications, aggregate_results)

    signature_parameters = [
        Parameter("predictions", Parameter.POSITIONAL_OR_KEYWORD, annotation="pd.Series"),
        Parameter("metrics", Parameter.POSITIONAL_OR_KEYWORD, annotation=dict[str, MetricValue]),
        Parameter("inputs", Parameter.POSITIONAL_OR_KEYWORD, annotation="pd.Series"),
    ]

    # Add grading_context_columns to signature list
    for var in grading_context_columns:
        signature_parameters.append(Parameter(var, Parameter.POSITIONAL_OR_KEYWORD))

    # Note: this doesn't change how python allows calling the function
    # extra params in grading_context_columns can only be passed as positional args
    eval_fn.__signature__ = Signature(signature_parameters)

    return _make_metric(
        eval_fn=eval_fn,
        greater_is_better=greater_is_better,
        name=name,
        version=version,
        metric_details=evaluation_context["eval_prompt"].__str__(),
        metric_metadata=metric_metadata,
        genai_metric_args=genai_metric_args,
        require_strict_signature=True,
    )


def _filter_by_field(df, field_name, value):
    return df[df[field_name] == value]


def _deserialize_genai_metric_args(args_dict):
    mlflow_version_at_ser = args_dict.pop("mlflow_version", None)
    fn_name = args_dict.pop("fn_name", None)
    if fn_name is None or mlflow_version_at_ser is None:
        raise MlflowException(
            message="The artifact JSON file appears to be corrupted and cannot be deserialized. "
            "Please regenerate the custom metrics and rerun the evaluation. "
            "Ensure that the file is correctly formatted and not tampered with.",
            error_code=INTERNAL_ERROR,
        )

    if mlflow_version_at_ser != VERSION:
        warnings.warn(
            f"The custom metric definitions were serialized using MLflow {mlflow_version_at_ser}. "
            f"Deserializing them with the current version {VERSION} might cause mismatches. "
            "Please ensure compatibility or consider regenerating the metrics "
            "using the current version.",
            UserWarning,
            stacklevel=2,
        )

    if fn_name == make_genai_metric_from_prompt.__name__:
        return make_genai_metric_from_prompt(**args_dict)

    examples = args_dict["examples"]
    if examples is not None:
        args_dict["examples"] = [EvaluationExample(**example) for example in examples]

    return make_genai_metric(**args_dict)


def retrieve_custom_metrics(
    run_id: str,
    name: str | None = None,
    version: str | None = None,
) -> list[EvaluationMetric]:
    """
    Retrieve the custom metrics created by users through `make_genai_metric()` or
    `make_genai_metric_from_prompt()` that are associated with a particular evaluation run.

    Args:
        run_id: The unique identifier for the run.
        name: (Optional) The name of the custom metric to retrieve.
            If None, retrieve all metrics.
        version: (Optional) The version of the custom metric to retrieve.
            If None, retrieve all metrics.

    Returns:
        A list of EvaluationMetric objects that match the retrieval criteria.

    .. code-block:: python
        :caption: Example for retrieving a custom genai metric

        import pandas as pd

        import mlflow
        from mlflow.metrics.genai.genai_metric import (
            make_genai_metric_from_prompt,
            retrieve_custom_metrics,
        )

        eval_df = pd.DataFrame(
            {
                "inputs": ["foo"],
                "ground_truth": ["bar"],
            }
        )
        with mlflow.start_run() as run:
            system_prompt = "Answer the following question in two sentences"
            basic_qa_model = mlflow.openai.log_model(
                model="gpt-4o-mini",
                task="chat.completions",
                name="model",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "{question}"},
                ],
            )
            custom_metric = make_genai_metric_from_prompt(
                name="custom_llm_judge",
                judge_prompt="This is a custom judge prompt.",
                greater_is_better=False,
                parameters={"temperature": 0.0},
            )
            results = mlflow.evaluate(
                basic_qa_model.model_uri,
                eval_df,
                targets="ground_truth",
                model_type="question-answering",
                evaluators="default",
                extra_metrics=[custom_metric],
            )
        metrics = retrieve_custom_metrics(
            run_id=run.info.run_id,
            name="custom_llm_judge",
        )
    """
    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run_id)]
    if _GENAI_CUSTOM_METRICS_FILE_NAME not in artifacts:
        _logger.warning("No custom metric definitions were found for this evaluation run.")
        return []

    with TemporaryDirectory() as tmpdir:
        downloaded_artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=_GENAI_CUSTOM_METRICS_FILE_NAME,
            dst_path=tmpdir,
        )
        custom_metrics = client._read_from_file(downloaded_artifact_path)

    if name is not None:
        custom_metrics = _filter_by_field(custom_metrics, "name", name)
    if version is not None:
        custom_metrics = _filter_by_field(custom_metrics, "version", version)
    metric_args_list = custom_metrics["metric_args"].tolist()
    if len(metric_args_list) == 0:
        _logger.warning("No matching custom metric definitions were found.")
        return []

    return [_deserialize_genai_metric_args(a) for a in metric_args_list]
