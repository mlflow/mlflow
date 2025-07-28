import functools
import inspect
import logging
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Optional, Union

from pydantic import BaseModel, PrivateAttr

import mlflow
from mlflow.entities import Assessment, Feedback
from mlflow.entities.assessment import DEFAULT_FEEDBACK_NAME
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)

# Serialization version for tracking changes to the serialization format
_SERIALIZATION_VERSION = 1


@dataclass
class SerializedScorer:
    """
    Dataclass defining the serialization schema for Scorer objects.
    """

    # Core scorer fields
    name: str
    aggregations: Optional[list[str]] = None

    # Version metadata
    mlflow_version: str = mlflow.__version__
    serialization_version: int = _SERIALIZATION_VERSION

    # Builtin scorer fields (for scorers from mlflow.genai.scorers.builtin_scorers)
    builtin_scorer_class: Optional[str] = None
    builtin_scorer_pydantic_data: Optional[dict] = None

    # Decorator scorer fields (for @scorer decorated functions)
    call_source: Optional[str] = None
    call_signature: Optional[str] = None
    original_func_name: Optional[str] = None


@experimental
class Scorer(BaseModel):
    name: str
    aggregations: Optional[list] = None

    _cached_dump: Optional[dict[str, Any]] = PrivateAttr(default=None)

    def model_dump(self, **kwargs) -> dict:
        """Override model_dump to include source code."""
        # Create serialized scorer with core fields

        # Return cached dump if available (prevents re-serialization issues with dynamic functions)
        if self._cached_dump is not None:
            return self._cached_dump

        serialized = SerializedScorer(
            name=self.name,
            aggregations=self.aggregations,
            mlflow_version=mlflow.__version__,
            serialization_version=_SERIALIZATION_VERSION,
        )

        # Check if this is a decorator scorer
        if hasattr(self, "_original_func") and self._original_func:
            # Decorator scorer - extract and store source code
            source_info = self._extract_source_code_info()
            serialized.call_source = source_info.get("call_source")
            serialized.call_signature = source_info.get("call_signature")
            serialized.original_func_name = source_info.get("original_func_name")
        else:
            # BuiltInScorer overrides `model_dump`, so this is neither a builtin scorer nor a
            # decorator scorer
            raise MlflowException.invalid_parameter_value(
                f"Unsupported scorer type: {self.__class__.__name__}. "
                f"Scorer serialization only supports:\n"
                f"1. Builtin scorers (from mlflow.genai.scorers.builtin_scorers)\n"
                f"2. Decorator-created scorers (using @scorer decorator)\n"
                f"Direct subclassing of Scorer is not supported for serialization. "
                f"Please use the @scorer decorator instead."
            )

        return asdict(serialized)

    def _extract_source_code_info(self) -> dict:
        """Extract source code information for the original decorated function."""
        from mlflow.genai.scorers.scorer_utils import extract_function_body

        result = {"call_source": None, "call_signature": None, "original_func_name": None}

        # Extract original function source
        call_body, _ = extract_function_body(self._original_func)
        result["call_source"] = call_body
        result["original_func_name"] = self._original_func.__name__

        # Store the signature of the original function
        result["call_signature"] = str(inspect.signature(self._original_func))

        return result

    @classmethod
    def model_validate(cls, obj: Any) -> "Scorer":
        """Override model_validate to reconstruct scorer from source code."""
        if not isinstance(obj, dict):
            raise MlflowException.invalid_parameter_value(
                f"Invalid scorer data: expected a dictionary, got {type(obj).__name__}. "
                f"Scorer data must be a dictionary containing serialized scorer information."
            )

        # Parse the serialized data using our dataclass
        try:
            serialized = SerializedScorer(**obj)
        except Exception as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to parse serialized scorer data: {e}"
            )

        # Log version information for debugging
        if serialized.mlflow_version:
            _logger.debug(
                f"Deserializing scorer created with MLflow version: {serialized.mlflow_version}"
            )
        if serialized.serialization_version:
            _logger.debug(f"Scorer serialization version: {serialized.serialization_version}")

        if serialized.builtin_scorer_class:
            # Import here to avoid circular imports
            from mlflow.genai.scorers.builtin_scorers import BuiltInScorer

            return BuiltInScorer.model_validate(obj)

        # Handle decorator scorers
        elif serialized.call_source and serialized.call_signature and serialized.original_func_name:
            return cls._reconstruct_decorator_scorer(serialized)

        # Invalid serialized data
        else:
            raise MlflowException.invalid_parameter_value(
                f"Failed to load scorer '{serialized.name}'. The scorer is serialized in an "
                f"unknown format that cannot be deserialized. Please make sure you are using "
                f"a compatible MLflow version or recreate the scorer. "
                f"Scorer was created with MLflow version: "
                f"{serialized.mlflow_version or 'unknown'}, "
                f"serialization version: {serialized.serialization_version or 'unknown'}, "
                f"current MLflow version: {mlflow.__version__}."
            )

    @classmethod
    def _reconstruct_decorator_scorer(cls, serialized: SerializedScorer) -> "Scorer":
        """Reconstruct a decorator scorer from serialized data."""
        from mlflow.genai.scorers.scorer_utils import recreate_function

        # Recreate the original function from source code
        recreated_func = recreate_function(
            serialized.call_source, serialized.call_signature, serialized.original_func_name
        )

        if not recreated_func:
            raise MlflowException.invalid_parameter_value(
                f"Failed to recreate function from source code. "
                f"Scorer was created with MLflow version: "
                f"{serialized.mlflow_version or 'unknown'}, "
                f"serialization version: {serialized.serialization_version or 'unknown'}. "
                f"Current MLflow version: {mlflow.__version__}"
            )

        # Apply the scorer decorator to recreate the scorer
        # Rather than serializing and deserializing the `run` method of `Scorer`, we recreate the
        # Scorer using the original function and the `@scorer` decorator. This should be safe so
        # long as `@scorer` is a stable API.
        scorer_instance = scorer(
            recreated_func, name=serialized.name, aggregations=serialized.aggregations
        )
        # Cache the serialized data to prevent re-serialization issues with dynamic functions
        original_serialized_data = asdict(serialized)
        object.__setattr__(scorer_instance, "_cached_dump", original_serialized_data)
        return scorer_instance

    def run(self, *, inputs=None, outputs=None, expectations=None, trace=None):
        from mlflow.evaluation import Assessment as LegacyAssessment

        merged = {
            "inputs": inputs,
            "outputs": outputs,
            "expectations": expectations,
            "trace": trace,
        }
        # Filter to only the parameters the function actually expects
        sig = inspect.signature(self.__call__)
        filtered = {k: v for k, v in merged.items() if k in sig.parameters}
        result = self(**filtered)
        if not (
            # TODO: Replace 'Assessment' with 'Feedback' once we migrate from the agent eval harness
            isinstance(result, (int, float, bool, str, Assessment, LegacyAssessment))
            or (
                isinstance(result, list)
                and all(isinstance(item, (Assessment, LegacyAssessment)) for item in result)
            )
            # Allow None to represent an empty assessment from the scorer.
            or result is None
        ):
            if isinstance(result, list) and len(result) > 0:
                result_type = "list[" + type(result[0]).__name__ + "]"
            else:
                result_type = type(result).__name__
            raise MlflowException.invalid_parameter_value(
                f"{self.name} must return one of int, float, bool, str, "
                f"Feedback, or list[Feedback]. Got {result_type}"
            )

        if isinstance(result, Feedback) and result.name == DEFAULT_FEEDBACK_NAME:
            # NB: Overwrite the returned feedback name to the scorer name. This is important
            # so we show a consistent name for the feedback regardless of whether the scorer
            # succeeds or fails. For example, let's say we have a scorer like this:
            #
            # @scorer
            # def my_scorer():
            #     # do something
            #     ...
            #     return Feedback(value=True)
            #
            # If the scorer succeeds, the returned feedback name will be default "feedback".
            # However, if the scorer fails, it doesn't return a Feedback object, and we
            # only know the scorer name. To unify this behavior, we overwrite the feedback
            # name to the scorer name in the happy path.
            # This will not apply when the scorer returns a list of Feedback objects.
            # or users explicitly specify the feedback name via Feedback constructor.
            result.name = self.name

        return result

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: Optional[dict[str, Any]] = None,
        trace: Optional[Trace] = None,
    ) -> Union[int, float, bool, str, Feedback, list[Feedback]]:
        """
        Implement the custom scorer's logic here.


        The scorer will be called for each row in the input evaluation dataset.

        Your scorer doesn't need to have all the parameters defined in the base
        signature. You can define a custom scorer with only the parameters you need.
        See the parameter details below for what values are passed for each parameter.

        .. list-table::
            :widths: 20 20 20
            :header-rows: 1

            * - Parameter
              - Description
              - Source

            * - ``inputs``
              - A single input to the target model/app.
              - Derived from either dataset or trace.

                * When the dataset contains ``inputs`` column, the value will be
                  passed as is.
                * When traces are provided as evaluation dataset, this will be derived
                  from the ``inputs`` field of the trace (i.e. inputs captured as the
                  root span of the trace).

            * - ``outputs``
              - A single output from the target model/app.
              - Derived from either dataset, trace, or output of ``predict_fn``.

                * When the dataset contains ``outputs`` column, the value will be
                  passed as is.
                * When ``predict_fn`` is provided, MLflow will make a prediction using the
                  ``inputs`` and the ``predict_fn``, and pass the result as the ``outputs``.
                * When traces are provided as evaluation dataset, this will be derived
                  from the ``response`` field of the trace (i.e. outputs captured as the
                  root span of the trace).

            * - ``expectations``
              - Ground truth or any expectation for each prediction, e.g. expected retrieved docs.
              - Derived from either dataset or trace.

                * When the dataset contains ``expectations`` column, the value will be
                  passed as is.
                * When traces are provided as evaluation dataset, this will be a dictionary
                  that contains a set of assessments in the format of
                  [assessment name]: [assessment value].

            * - ``trace``
              - A trace object corresponding to the prediction for the row.
              - Specified as a ``trace`` column in the dataset, or generated during the prediction.

        Example:

            .. code-block:: python

                class NotEmpty(BaseScorer):
                    name = "not_empty"

                    def __call__(self, *, outputs) -> bool:
                        return outputs != ""


                class ExactMatch(BaseScorer):
                    name = "exact_match"

                    def __call__(self, *, outputs, expectations) -> bool:
                        return outputs == expectations["expected_response"]


                class NumToolCalls(BaseScorer):
                    name = "num_tool_calls"

                    def __call__(self, *, trace) -> int:
                        spans = trace.search_spans(name="tool_call")
                        return len(spans)


                # Use the scorer in an evaluation
                mlflow.genai.evaluate(
                    data=data,
                    scorers=[NotEmpty(), ExactMatch(), NumToolCalls()],
                )
        """
        raise NotImplementedError("Implementation of __call__ is required for Scorer class")


@experimental
def scorer(
    func=None,
    *,
    name: Optional[str] = None,
    aggregations: Optional[
        list[Union[Literal["min", "max", "mean", "median", "variance", "p90", "p99"], Callable]]
    ] = None,
):
    """
    A decorator to define a custom scorer that can be used in ``mlflow.genai.evaluate()``.

    The scorer function should take in a **subset** of the following parameters:

    .. list-table::
        :widths: 20 20 20
        :header-rows: 1

        * - Parameter
          - Description
          - Source

        * - ``inputs``
          - A single input to the target model/app.
          - Derived from either dataset or trace.

            * When the dataset contains ``inputs`` column, the value will be passed as is.
            * When traces are provided as evaluation dataset, this will be derived
              from the ``inputs`` field of the trace (i.e. inputs captured as the
              root span of the trace).

        * - ``outputs``
          - A single output from the target model/app.
          - Derived from either dataset, trace, or output of ``predict_fn``.

            * When the dataset contains ``outputs`` column, the value will be passed as is.
            * When ``predict_fn`` is provided, MLflow will make a prediction using the
              ``inputs`` and the ``predict_fn`` and pass the result as the ``outputs``.
            * When traces are provided as evaluation dataset, this will be derived
              from the ``response`` field of the trace (i.e. outputs captured as the
              root span of the trace).

        * - ``expectations``
          - Ground truth or any expectation for each prediction e.g., expected retrieved docs.
          - Derived from either dataset or trace.

            * When the dataset contains ``expectations`` column, the value will be passed as is.
            * When traces are provided as evaluation dataset, this will be a dictionary
              that contains a set of assessments in the format of
              [assessment name]: [assessment value].

        * - ``trace``
          - A trace object corresponding to the prediction for the row.
          - Specified as a ``trace`` column in the dataset, or generated during the prediction.

    The scorer function should return one of the following:

    * A boolean value
    * An integer value
    * A float value
    * A string value
    * A single :class:`~mlflow.entities.Feedback` object
    * A list of :class:`~mlflow.entities.Feedback` objects

    .. note::

        The metric name will be determined by the scorer function's name or a custom name
        specified in the `name` parameter for the scorer.

    Example:

        .. code-block:: python

            import json
            from mlflow.genai.scorers import scorer
            from mlflow.entities import AssessmentSource, Feedback


            # Basic scorers that returns primitive values
            @scorer
            def not_empty(outputs) -> bool:
                return outputs != ""


            @scorer
            def exact_match(outputs, expectations) -> bool:
                return outputs == expectations["expected_response"]


            @scorer
            def num_tool_calls(trace) -> int:
                spans = trace.search_spans(name="tool_call")
                return len(spans)


            # Use `Feedback` object to return additional information about the scorer's
            # result, such as a rationale for the score.
            @scorer
            def harmfulness(outputs) -> Feedback:
                import openai

                prompt = f'''
                    Judge if the following text is harmful or not.

                    Text:
                    {outputs}

                    Return the answer in a JSON object with the following format:
                    {{
                        "harmful": true
                        "reason": "The text contains harmful content"
                    }}

                    Do not output any other characters than the json object.
                '''
                response = openai.OpenAI().chat.completions.create(
                    model="o4-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                payload = json.loads(response.choices[0].message.content)
                return Feedback(
                    value=payload["harmful"],
                    rationale=payload["reason"],
                    source=AssessmentSource(
                        source_type="LLM_JUDGE",
                        source_id="openai:/o4-mini",
                    ),
                )


            # Use the scorer in an evaluation
            mlflow.genai.evaluate(
                data=data,
                scorers=[not_empty, exact_match, num_tool_calls, harmfulness],
            )
    """

    if func is None:
        return functools.partial(scorer, name=name, aggregations=aggregations)

    class CustomScorer(Scorer):
        # Store reference to the original function
        _original_func: Optional[Callable] = PrivateAttr(default=None)

        def __init__(self, **data):
            super().__init__(**data)
            # Set the original function reference
            # Use object.__setattr__ to bypass Pydantic's attribute handling for private attributes
            # during model initialization, as direct assignment (self._original_func = func) may be
            # ignored or fail in this context
            object.__setattr__(self, "_original_func", func)

        def __call__(self, *args, **kwargs):
            return func(*args, **kwargs)

    # Update the __call__ method's signature to match the original function
    # but add 'self' as the first parameter. This is required for MLflow to
    # pass the correct set of parameters to the scorer.
    signature = inspect.signature(func)
    params = list(signature.parameters.values())
    new_params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)] + params
    new_signature = signature.replace(parameters=new_params)
    CustomScorer.__call__.__signature__ = new_signature

    return CustomScorer(
        name=name or func.__name__,
        aggregations=aggregations,
    )
