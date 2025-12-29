import functools
import inspect
import logging
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Literal, TypeAlias

from pydantic import BaseModel, PrivateAttr

import mlflow
from mlflow.entities import Assessment, Feedback
from mlflow.entities.assessment import DEFAULT_FEEDBACK_NAME
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.telemetry.events import ScorerCallEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.tracking import get_tracking_uri
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.uri import is_databricks_uri

_logger = logging.getLogger(__name__)

# Context variable to track if we're in a scorer call (prevents nested telemetry)
_in_scorer_call: ContextVar[bool] = ContextVar("mlflow_scorer_call_context", default=False)

# Serialization version for tracking changes to the serialization format
_SERIALIZATION_VERSION = 1
_AggregationFunc: TypeAlias = Callable[[list[int | float]], float]
_AggregationType: TypeAlias = (
    Literal["min", "max", "mean", "median", "variance", "p90"] | _AggregationFunc
)


class ScorerKind(Enum):
    CLASS = "class"
    BUILTIN = "builtin"
    DECORATOR = "decorator"
    INSTRUCTIONS = "instructions"
    GUIDELINES = "guidelines"
    THIRD_PARTY = "third_party"


_ALLOWED_SCORERS_FOR_REGISTRATION = [
    ScorerKind.BUILTIN,
    ScorerKind.DECORATOR,
    ScorerKind.INSTRUCTIONS,
    ScorerKind.GUIDELINES,
]


class ScorerStatus(Enum):
    """Status of a scorer.

    Scorer status is determined by the sample rate due to the backend not having
    a notion of whether a scorer is started or stopped.
    """

    UNREGISTERED = "UNREGISTERED"  # sampling config not set
    STARTED = "STARTED"  # sample_rate > 0
    STOPPED = "STOPPED"  # sample_rate == 0


@dataclass
class ScorerSamplingConfig:
    """Configuration for registered scorer sampling."""

    sample_rate: float | None = None
    filter_string: str | None = None


AggregationFunc = Callable[[list[float]], float]  # List of per-row value -> aggregated value


@dataclass
class SerializedScorer:
    """
    Dataclass defining the serialization schema for Scorer objects.
    """

    # Core scorer fields
    name: str
    aggregations: list[str] | None = None
    description: str | None = None
    is_session_level_scorer: bool = False

    # Version metadata
    mlflow_version: str = mlflow.__version__
    serialization_version: int = _SERIALIZATION_VERSION

    # Builtin scorer fields (for scorers from mlflow.genai.scorers.builtin_scorers)
    builtin_scorer_class: str | None = None
    builtin_scorer_pydantic_data: dict[str, Any] | None = None

    # Decorator scorer fields (for @scorer decorated functions)
    call_source: str | None = None
    call_signature: str | None = None
    original_func_name: str | None = None

    # InstructionsJudge fields (for make_judge created judges)
    instructions_judge_pydantic_data: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate that exactly one type of scorer fields is present."""
        has_builtin_fields = self.builtin_scorer_class is not None
        has_decorator_fields = self.call_source is not None
        has_instructions_fields = self.instructions_judge_pydantic_data is not None

        # Count how many field types are present
        field_count = sum([has_builtin_fields, has_decorator_fields, has_instructions_fields])

        if field_count == 0:
            raise ValueError(
                "SerializedScorer must have either builtin scorer fields "
                "(builtin_scorer_class), decorator scorer fields (call_source), "
                "or instructions judge fields (instructions_judge_pydantic_data) present"
            )

        if field_count > 1:
            raise ValueError(
                "SerializedScorer cannot have multiple types of scorer fields "
                "present simultaneously"
            )


def _record_scorer_call_with_context(func):
    """Wraps a scorer's __call__ method with telemetry, but only for top-level calls.

    Uses ContextVar to track nesting in a thread-safe manner. Only supports synchronous
    functions - async scorers are not supported and will raise an exception.
    """
    if inspect.iscoroutinefunction(func):
        raise TypeError(
            f"Async scorer '__call__' methods are not supported. "
            f"Scorer class {func.__qualname__} has an async __call__ method."
        )

    telemetry_wrapped = record_usage_event(ScorerCallEvent)(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if _in_scorer_call.get():
            # Nested call: skip telemetry, call original function
            return func(*args, **kwargs)

        # Top-level call: set context and use telemetry-wrapped version
        token = _in_scorer_call.set(True)
        try:
            return telemetry_wrapped(*args, **kwargs)
        finally:
            # CRITICAL: Use reset(token), NOT set(False)
            # This ensures proper context restoration
            _in_scorer_call.reset(token)

    return wrapper


class Scorer(BaseModel):
    name: str
    aggregations: list[_AggregationType] | None = None
    description: str | None = None

    _cached_dump: dict[str, Any] | None = PrivateAttr(default=None)
    _sampling_config: ScorerSamplingConfig | None = PrivateAttr(default=None)
    _registered_backend: str | None = PrivateAttr(default=None)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Only wrap __call__ if it's defined directly in this class (cls.__dict__),
        # not inherited from a parent. This prevents wrapping the same method multiple
        # times in the inheritance chain.
        if "__call__" in cls.__dict__:
            original_call = cls.__dict__["__call__"]

            # Check if it's already wrapped to avoid double-wrapping
            if not hasattr(original_call, "_telemetry_wrapped"):
                # Wrap with context-aware telemetry decorator
                wrapped_call = _record_scorer_call_with_context(original_call)
                # Mark it as wrapped to prevent double-wrapping
                wrapped_call._telemetry_wrapped = True
                setattr(cls, "__call__", wrapped_call)

    @property
    def is_session_level_scorer(self) -> bool:
        """Get whether this scorer is a session-level scorer.

        Defaults to False. Child classes can override this property to return True
        or compute the value dynamically based on their configuration.
        """
        return False

    @property
    def sample_rate(self) -> float | None:
        """Get the sample rate for this scorer. Available when registered for monitoring."""
        return self._sampling_config.sample_rate if self._sampling_config else None

    @property
    def filter_string(self) -> str | None:
        """Get the filter string for this scorer."""
        return self._sampling_config.filter_string if self._sampling_config else None

    @property
    def status(self) -> ScorerStatus:
        """Get the status of this scorer, using only the local state."""

        if self._registered_backend is None:
            return ScorerStatus.UNREGISTERED

        return ScorerStatus.STARTED if (self.sample_rate or 0) > 0 else ScorerStatus.STOPPED

    def __repr__(self) -> str:
        # Get the standard representation from the parent class
        base_repr = super().__repr__()
        filter_string = self.filter_string
        if filter_string is not None:
            filter_string = f"'{filter_string}'"

        # Inject the property's value into the repr string
        return f"{base_repr[:-1]}, sample_rate={self.sample_rate}, filter_string={filter_string})"

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Override model_dump to include source code."""

        # Return cached dump if available (prevents re-serialization issues with dynamic functions)
        if self._cached_dump is not None:
            return self._cached_dump

        # Check if this is a decorator scorer
        if not getattr(self, "_original_func", None):
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

        # Decorator scorer - extract and store source code
        source_info = self._extract_source_code_info()

        # Create serialized scorer with all fields at once
        serialized = SerializedScorer(
            name=self.name,
            description=self.description,
            aggregations=self.aggregations,
            is_session_level_scorer=self.is_session_level_scorer,
            mlflow_version=mlflow.__version__,
            serialization_version=_SERIALIZATION_VERSION,
            call_source=source_info.get("call_source"),
            call_signature=source_info.get("call_signature"),
            original_func_name=source_info.get("original_func_name"),
        )
        self._cached_dump = asdict(serialized)
        return self._cached_dump

    def _extract_source_code_info(self) -> dict[str, str | None]:
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
        # Handle SerializedScorer object
        if isinstance(obj, SerializedScorer):
            serialized = obj
        # Handle dict object
        elif isinstance(obj, dict):
            # Parse the serialized data using our dataclass
            try:
                serialized = SerializedScorer(**obj)
            except Exception as e:
                raise MlflowException.invalid_parameter_value(
                    f"Failed to parse serialized scorer data: {e}"
                )
        else:
            raise MlflowException.invalid_parameter_value(
                f"Invalid scorer data: expected a SerializedScorer object or dictionary, "
                f"got {type(obj).__name__}. Scorer data must be either a SerializedScorer object "
                "or a dictionary containing serialized scorer information."
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

        # Handle InstructionsJudge scorers
        elif serialized.instructions_judge_pydantic_data is not None:
            from mlflow.genai.judges.instructions_judge import InstructionsJudge

            data = serialized.instructions_judge_pydantic_data

            field_specs = {
                "instructions": str,
                "model": str,
            }

            errors = []
            for field, expected_type in field_specs.items():
                if field not in data:
                    errors.append(f"missing required field '{field}'")
                elif not isinstance(data[field], expected_type):
                    actual_type = type(data[field]).__name__
                    errors.append(
                        f"field '{field}' must be {expected_type.__name__}, got {actual_type}"
                    )

            if errors:
                raise MlflowException.invalid_parameter_value(
                    f"Failed to deserialize InstructionsJudge scorer '{serialized.name}': "
                    f"{'; '.join(errors)}"
                )

            feedback_value_type = str  # default to str
            if "feedback_value_type" in data and data["feedback_value_type"] is not None:
                feedback_value_type = InstructionsJudge._deserialize_feedback_value_type(
                    data["feedback_value_type"]
                )

            try:
                return InstructionsJudge(
                    name=serialized.name,
                    description=serialized.description,
                    instructions=data["instructions"],
                    model=data["model"],
                    feedback_value_type=feedback_value_type,
                    # TODO: add aggregations here once we support boolean/numeric judge outputs
                )
            except Exception as e:
                raise MlflowException.invalid_parameter_value(
                    f"Failed to create InstructionsJudge scorer '{serialized.name}': {e}"
                )

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
        from mlflow.genai.scorers.scorer_utils import recreate_function

        # NB: Custom (@scorer) scorers use exec() during deserialization, which poses a code
        # execution risk. Only allow loading in Databricks runtime environments where the
        # execution environment is controlled.
        if not is_in_databricks_runtime():
            code_snippet = (
                "\n\nfrom mlflow.genai import scorer\n\n"
                f"@scorer\ndef {serialized.original_func_name}{serialized.call_signature}:\n"
            )
            for line in serialized.call_source.split("\n"):
                code_snippet += f"    {line}\n"

            is_databricks_remote = is_databricks_uri(get_tracking_uri())

            if is_databricks_remote:
                error_msg = (
                    f"Loading custom scorer '{serialized.name}' via remote access is not "
                    "supported. You are connected to a Databricks workspace but executing code "
                    "outside of it. Custom scorers require arbitrary code execution during "
                    "deserialization and must be loaded within the Databricks workspace for "
                    "security reasons.\n\n"
                    "To use this scorer:\n"
                    "1. Run your code inside the Databricks workspace (notebook or job), or\n"
                    "2. Copy the code below and use it directly in your source code, or\n"
                    "3. Use built-in scorers or make_judge() scorers instead\n\n"
                    f"Registered scorer code:\n{code_snippet}"
                )
            else:
                error_msg = (
                    f"Loading custom scorer '{serialized.name}' is not supported outside of "
                    "Databricks runtime environments due to security concerns. Custom scorers "
                    "require arbitrary code execution during deserialization.\n\n"
                    "To use this scorer, please:\n"
                    "1. Copy the code below and save it in your source code repository\n"
                    "2. Import and use it directly in your code, or\n"
                    "3. Use built-in scorers or make_judge() scorers instead\n\n"
                    f"Registered scorer code:\n{code_snippet}"
                )

            raise MlflowException.invalid_parameter_value(error_msg)

        try:
            recreated_func = recreate_function(
                serialized.call_source, serialized.call_signature, serialized.original_func_name
            )
        except Exception as e:
            raise MlflowException.invalid_parameter_value(
                f"Failed to recreate function from source code. "
                f"Scorer was created with MLflow version: "
                f"{serialized.mlflow_version or 'unknown'}, "
                f"serialization version: {serialized.serialization_version or 'unknown'}. "
                f"Current MLflow version: {mlflow.__version__}. "
                f"Error: {e}"
            )

        # Apply the scorer decorator to recreate the scorer
        # Rather than serializing and deserializing the `run` method of `Scorer`, we recreate the
        # Scorer using the original function and the `@scorer` decorator. This should be safe so
        # long as `@scorer` is a stable API.
        scorer_instance = scorer(
            recreated_func,
            name=serialized.name,
            description=serialized.description,
            aggregations=serialized.aggregations,
        )
        # Cache the serialized data to prevent re-serialization issues with dynamic functions
        original_serialized_data = asdict(serialized)
        object.__setattr__(scorer_instance, "_cached_dump", original_serialized_data)
        return scorer_instance

    def run(self, *, inputs=None, outputs=None, expectations=None, trace=None, session=None):
        from mlflow.evaluation import Assessment as LegacyAssessment

        merged = {
            "inputs": inputs,
            "outputs": outputs,
            "expectations": expectations,
            "trace": trace,
            "session": session,
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
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
        session: list[Trace] | None = None,
    ) -> int | float | bool | str | Feedback | list[Feedback]:
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

            * - ``session``
              - A list of trace objects belonging to the same conversation session.
              - Specify this parameter only for session_level scorers
                (scorers with ``is_session_level_scorer = True``).
                * Only traces with the same ``mlflow.trace.session`` metadata value can be passed in
                  this parameter, otherwise an error will be raised.

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

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.CLASS

    def register(self, *, name: str | None = None, experiment_id: str | None = None) -> "Scorer":
        """
        Register this scorer with the MLflow server.

        This method registers the scorer for use with automatic trace evaluation in the
        specified experiment. Once registered, the scorer can be started to begin
        evaluating traces automatically.

        Args:
            name: Optional registered name for the scorer. If not provided, the current `name`
                property value will be used as a registered name.
            experiment_id: The ID of the MLflow experiment to register the scorer for.
                If None, uses the currently active experiment.

        Returns:
            A new Scorer instance with server registration information.

        Example:

            .. code-block:: python

                import mlflow
                from mlflow.genai.scorers import RelevanceToQuery

                # Register a built-in scorer
                mlflow.set_experiment("my_genai_app")
                registered_scorer = RelevanceToQuery().register(name="relevance_scorer")
                print(f"Registered scorer: {registered_scorer.name}")

                # Register a custom scorer
                from mlflow.genai.scorers import scorer


                @scorer
                def custom_length_check(outputs) -> bool:
                    return len(outputs) > 100


                registered_custom = custom_length_check.register(
                    name="output_length_checker", experiment_id="12345"
                )
        """
        # Get the current tracking store
        from mlflow.genai.scorers.registry import DatabricksStore, _get_scorer_store

        self._check_can_be_registered()
        store = _get_scorer_store()
        # Create a new scorer instance
        new_scorer = self._create_copy()

        # If name is provided, update the copy's name
        if name:
            new_scorer.name = name
            # Update cached dump to reflect the new name
            if new_scorer._cached_dump is not None:
                new_scorer._cached_dump["name"] = name

        store.register_scorer(experiment_id, new_scorer)

        if isinstance(store, DatabricksStore):
            new_scorer._registered_backend = "databricks"
        else:
            new_scorer._registered_backend = "tracking"
        return new_scorer

    def start(
        self,
        *,
        name: str | None = None,
        experiment_id: str | None = None,
        sampling_config: ScorerSamplingConfig,
    ) -> "Scorer":
        """
        Start registered scoring with the specified sampling configuration.

        This method activates automatic trace evaluation for the scorer. The scorer will
        evaluate traces based on the provided sampling configuration, including the
        sample rate and optional filter criteria.

        Args:
            name: Optional scorer name. If not provided, uses the scorer's registered
                name or default name.
            experiment_id: The ID of the MLflow experiment containing the scorer.
                If None, uses the currently active experiment.
            sampling_config: Configuration object containing:
                - sample_rate: Fraction of traces to evaluate (0.0 to 1.0). Required.
                - filter_string: Optional MLflow search_traces compatible filter string.

        Returns:
            A new Scorer instance with updated sampling configuration.

        Example:
            .. code-block:: python

                import mlflow
                from mlflow.genai.scorers import RelevanceToQuery, ScorerSamplingConfig

                # Start scorer with 50% sampling rate
                mlflow.set_experiment("my_genai_app")
                scorer = RelevanceToQuery().register()
                active_scorer = scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=0.5))
                print(f"Scorer is evaluating {active_scorer.sample_rate * 100}% of traces")

                # Start scorer with filter to only evaluate specific traces
                filtered_scorer = scorer.start(
                    sampling_config=ScorerSamplingConfig(
                        sample_rate=1.0, filter_string="YOUR_FILTER_STRING"
                    )
                )
        """
        from mlflow.genai.scorers.registry import DatabricksStore
        from mlflow.tracking._tracking_service.utils import get_tracking_uri
        from mlflow.utils.uri import is_databricks_uri

        if not is_databricks_uri(get_tracking_uri()):
            raise MlflowException(
                "Scheduling scorers is only supported by Databricks tracking URI."
            )

        self._check_can_be_registered()

        if sampling_config.sample_rate is not None and sampling_config.sample_rate <= 0:
            raise MlflowException.invalid_parameter_value(
                "When starting a scorer, provided sample rate must be greater than 0"
            )

        scorer_name = name or self.name

        # Update the scorer on the server
        return DatabricksStore.update_registered_scorer(
            name=scorer_name,
            scorer=self,
            sample_rate=sampling_config.sample_rate,
            filter_string=sampling_config.filter_string,
            experiment_id=experiment_id,
        )

    def update(
        self,
        *,
        name: str | None = None,
        experiment_id: str | None = None,
        sampling_config: ScorerSamplingConfig,
    ) -> "Scorer":
        """
        Update the sampling configuration for this scorer.

        This method modifies the sampling rate and/or filter criteria for an already
        registered scorer. It can be used to dynamically adjust how many traces are
        evaluated or change the filtering criteria without stopping and restarting
        the scorer.

        Args:
            name: Optional scorer name. If not provided, uses the scorer's registered name
                or default name.
            experiment_id: The ID of the MLflow experiment containing the scorer.
                If None, uses the currently active experiment.
            sampling_config: Configuration object containing:
                - sample_rate: New fraction of traces to evaluate (0.0 to 1.0). Optional.
                - filter_string: New MLflow search_traces compatible filter string. Optional.

        Returns:
            A new Scorer instance with updated configuration.

        Example:

            .. code-block:: python

                import mlflow
                from mlflow.genai.scorers import RelevanceToQuery, ScorerSamplingConfig

                # Start scorer with initial configuration
                mlflow.set_experiment("my_genai_app")
                scorer = RelevanceToQuery().register()
                active_scorer = scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=0.1))

                # Update to increase sampling rate during high traffic
                updated_scorer = active_scorer.update(
                    sampling_config=ScorerSamplingConfig(sample_rate=0.5)
                )
                print(f"Updated sample rate: {updated_scorer.sample_rate}")

                # Update to add filtering criteria
                filtered_scorer = updated_scorer.update(
                    sampling_config=ScorerSamplingConfig(filter_string="YOUR_FILTER_STRING")
                )
                print(f"Added filter: {filtered_scorer.filter_string}")
        """
        from mlflow.genai.scorers.registry import DatabricksStore
        from mlflow.tracking._tracking_service.utils import get_tracking_uri
        from mlflow.utils.uri import is_databricks_uri

        if not is_databricks_uri(get_tracking_uri()):
            raise MlflowException(
                "Updating scheduled scorers is only supported by Databricks tracking URI."
            )

        self._check_can_be_registered()

        scorer_name = name or self.name

        # Update the scorer on the server
        return DatabricksStore.update_registered_scorer(
            name=scorer_name,
            scorer=self,
            sample_rate=sampling_config.sample_rate,
            filter_string=sampling_config.filter_string,
            experiment_id=experiment_id,
        )

    def stop(self, *, name: str | None = None, experiment_id: str | None = None) -> "Scorer":
        """
        Stop registered scoring by setting sample rate to 0.

        This method deactivates automatic trace evaluation for the scorer while keeping
        the scorer registered. The scorer can be restarted later using the start() method.

        Args:
            name: Optional scorer name. If not provided, uses the scorer's registered name
                or default name.
            experiment_id: The ID of the MLflow experiment containing the scorer.
                If None, uses the currently active experiment.

        Returns:
            A new Scorer instance with sample rate set to 0.

        Example:

            .. code-block:: python

                import mlflow
                from mlflow.genai.scorers import RelevanceToQuery, ScorerSamplingConfig

                # Start and then stop a scorer
                mlflow.set_experiment("my_genai_app")
                scorer = RelevanceToQuery().register()
                active_scorer = scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=0.5))
                print(f"Scorer is active: {active_scorer.sample_rate > 0}")

                # Stop the scorer
                stopped_scorer = active_scorer.stop()
                print(f"Scorer is active: {stopped_scorer.sample_rate > 0}")

                # The scorer remains registered and can be restarted later
                restarted_scorer = stopped_scorer.start(
                    sampling_config=ScorerSamplingConfig(sample_rate=0.3)
                )
        """
        from mlflow.tracking._tracking_service.utils import get_tracking_uri
        from mlflow.utils.uri import is_databricks_uri

        if not is_databricks_uri(get_tracking_uri()):
            raise MlflowException(
                "Stopping scheduled scorers is only supported by Databricks tracking URI."
            )

        self._check_can_be_registered()

        scorer_name = name or self.name
        return self.update(
            name=scorer_name,
            experiment_id=experiment_id,
            sampling_config=ScorerSamplingConfig(sample_rate=0.0),
        )

    def _create_copy(self) -> "Scorer":
        """
        Create a copy of this scorer instance.
        """
        self._check_can_be_registered(
            error_message="Scorer must be a builtin or decorator scorer to be copied."
        )

        copy = self.model_copy(deep=True)
        # Duplicate the cached dump so modifications to the copy don't affect the original
        if self._cached_dump is not None:
            object.__setattr__(copy, "_cached_dump", dict(self._cached_dump))
        return copy

    def _check_can_be_registered(self, error_message: str | None = None) -> None:
        from mlflow.genai.scorers.registry import DatabricksStore, _get_scorer_store

        if self.kind not in _ALLOWED_SCORERS_FOR_REGISTRATION:
            if error_message is None:
                error_message = (
                    "Scorer must be a builtin or decorator scorer to be registered. "
                    f"Got {self.kind}."
                )
            raise MlflowException.invalid_parameter_value(error_message)

        # NB: Custom (@scorer) scorers use exec() during deserialization, which poses a code
        # execution risk. Only allow registration when using Databricks tracking URI.
        # Registration itself is safe (just stores code), but we restrict it to Databricks
        # to ensure loaded scorers can only be executed in controlled environments.
        if self.kind == ScorerKind.DECORATOR and not is_databricks_uri(get_tracking_uri()):
            raise MlflowException.invalid_parameter_value(
                "Custom scorer registration (using @scorer decorator) is not supported "
                "outside of Databricks tracking environments due to security concerns. "
                "Custom scorers require arbitrary code execution during deserialization.\n\n"
                "To use custom scorers:\n"
                "1. Configure MLflow to use a Databricks tracking URI, or\n"
                "2. Manage your custom scorer code in a source code repository "
                "(e.g., GitHub) and import it directly, or\n"
                "3. Use built-in scorers or make_judge() scorers instead"
            )

        store = _get_scorer_store()
        if (
            isinstance(store, DatabricksStore)
            and (model := getattr(self, "model", None))
            and not model.startswith("databricks")
        ):
            raise MlflowException.invalid_parameter_value(
                "The scorer's judge model must use Databricks as a model provider "
                "in order to be registered or updated. Please use the default judge model or "
                "specify a model value starting with `databricks:/`. "
                f"Got {model}."
            )


def scorer(
    func=None,
    *,
    name: str | None = None,
    description: str | None = None,
    aggregations: list[_AggregationType] | None = None,
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

    Args:
        func: The scorer function to be decorated.
        name: The name of the scorer.
        description: A description of what the scorer evaluates.
        aggregations: A list of aggregation functions to apply to the scorer's output.
            The aggregation functions can be either a string or a callable.

            * If a string, it must be one of `["min", "max", "mean", "median", "variance", "p90"]`.
            * If a callable, it must take a list of values and return a single value.

            By default, "mean" is used as the aggregation function.

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
        return functools.partial(
            scorer, name=name, description=description, aggregations=aggregations
        )

    class CustomScorer(Scorer):
        # Store reference to the original function
        _original_func: Callable[..., Any] | None = PrivateAttr(default=None)

        def __init__(self, **data):
            super().__init__(**data)
            # Set the original function reference
            # Use object.__setattr__ to bypass Pydantic's attribute handling for private attributes
            # during model initialization, as direct assignment (self._original_func = func) may be
            # ignored or fail in this context
            object.__setattr__(self, "_original_func", func)

        def __call__(self, *args, **kwargs):
            return func(*args, **kwargs)

        @property
        def kind(self) -> ScorerKind:
            return ScorerKind.DECORATOR

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
        description=description,
        aggregations=aggregations,
    )
