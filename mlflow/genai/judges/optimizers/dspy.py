"""DSPy-based alignment optimizer implementation."""

import json
import logging
from abc import abstractmethod
from typing import Any, Callable

from pydantic import PrivateAttr

from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.base import AlignmentOptimizer, Judge
from mlflow.genai.judges.make_judge import InstructionsJudge
from mlflow.genai.judges.utils import get_default_model
from mlflow.utils.annotations import experimental

logger = logging.getLogger(__name__)


@experimental(version="3.4.0")
class DSPyAlignmentOptimizer(AlignmentOptimizer):
    """
    Abstract base class for DSPy-based alignment optimizers.

    Provides common functionality for converting MLflow traces to DSPy examples
    and handling DSPy program compilation.
    """

    _logger: logging.Logger = PrivateAttr()
    _model: str = PrivateAttr()

    def __init__(self, model: str | None = None, **kwargs):
        """
        Initialize DSPy optimizer with common parameters.

        Args:
            model: Model to use for DSPy optimization. If None, uses get_default_model().
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        # Initialize private variables using PrivateAttr
        self._logger = logging.getLogger(self.__class__.__name__)
        # Private member variable for the model, defaulted to get_default_model()
        self._model = model if model is not None else get_default_model()

    @abstractmethod
    def _dspy_optimize(self, program, examples, metric_fn) -> Any:
        """
        Perform DSPy optimization with algorithm-specific parameters.

        Each implementation can decide how to split the data internally if needed.

        Args:
            program: The DSPy program to optimize
            examples: Examples for optimization (implementations decide how to split)
            metric_fn: Metric function for optimization

        Returns:
            Optimized DSPy program
        """

    def _extract_text_from_data(self, data: Any, field_type: str) -> str:
        """
        Extract text from trace data, handling various formats.

        Args:
            data: The data to extract from (can be dict, string, or other)
            field_type: Either 'request' or 'response' to determine which keys to try

        Returns:
            Extracted text as string
        """
        if data is None:
            return ""

        # If it's already a string, try to parse as JSON first
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
                data = parsed_data
            except (json.JSONDecodeError, ValueError):
                # If JSON parsing fails, return the string as-is
                return data

        # Handle dict case
        if isinstance(data, dict):
            # Define the keys to try based on field type
            if field_type == "request":
                keys_to_try = ["request", "inputs", "input", "prompt"]
            elif field_type == "response":
                keys_to_try = ["response", "outputs", "output", "content", "text"]
            else:
                raise AssertionError(
                    f"Invalid field_type: {field_type}. Must be 'request' or 'response'."
                )

            # Try each key in order
            for key in keys_to_try:
                if key in data:
                    value = data[key]
                    # If the value is a dict or list, convert to string
                    if isinstance(value, (dict, list)):
                        return json.dumps(value)
                    else:
                        return str(value)

            # If no specific keys found, return the full dict as string
            return json.dumps(data)

        # For any other type, convert to string
        return str(data)

    def _extract_request_from_trace(self, trace: Trace) -> str:
        """
        Extract request text from an MLflow trace object.

        Args:
            trace: MLflow trace object

        Returns:
            Extracted request text as string
        """
        # Try trace.data.request first, fall back to trace.info.request_preview
        request_data = (
            trace.data.request if hasattr(trace.data, "request") else trace.info.request_preview
        )
        return self._extract_text_from_data(request_data, "request")

    def _extract_response_from_trace(self, trace: Trace) -> str:
        """
        Extract response text from an MLflow trace object.

        Args:
            trace: MLflow trace object

        Returns:
            Extracted response text as string
        """
        # Try trace.data.response first, fall back to trace.info.response_preview
        response_data = (
            trace.data.response if hasattr(trace.data, "response") else trace.info.response_preview
        )
        return self._extract_text_from_data(response_data, "response")

    def _sanitize_judge_name(self, judge_name: str) -> str:
        """Sanitize judge name for consistent comparison."""
        return judge_name.lower().strip()

    def _trace_to_dspy_example(self, trace: Trace, judge_name: str) -> Any | None:
        """
        Convert MLflow trace to DSPy example format.

        Extracts:
        - inputs/outputs from trace spans
        - expected result from human assessments
        - rationale from assessment feedback

        Args:
            trace: MLflow trace object
            judge_name: Name of the judge to find assessments for

        Returns:
            DSPy example object or None if conversion fails
        """
        try:
            # Import dspy here to allow graceful failure
            import dspy

            # Extract request and response from trace
            request = self._extract_request_from_trace(trace)
            response = self._extract_response_from_trace(trace)

            if not request or not response:
                self._logger.warning(f"Missing request or response in trace {trace.info.trace_id}")
                return None

            # Find human assessment for this judge
            expected_result = None
            sanitized_judge_name = self._sanitize_judge_name(judge_name)

            if trace.info.assessments:
                for assessment in trace.info.assessments:
                    if (
                        assessment.name == sanitized_judge_name
                        and assessment.source.source_type == "HUMAN"
                    ):
                        expected_result = assessment
                        break

            if not expected_result:
                self._logger.warning(
                    f"No human assessment found for judge '{judge_name}' "
                    f"in trace {trace.info.trace_id}"
                )
                return None

            if not expected_result.feedback:
                self._logger.warning(
                    f"No feedback found in assessment for trace {trace.info.trace_id}"
                )
                return None

            # Create DSPy example
            example = dspy.Example(
                inputs=request,
                outputs=response,
                result=str(expected_result.feedback.value).lower(),
                rationale=expected_result.rationale if expected_result.rationale else "",
            )

            # Set inputs (what the model should use as input)
            return example.with_inputs("inputs", "outputs")

        except ImportError:
            raise MlflowException("DSPy library is required but not installed")
        except Exception as e:
            self._logger.error(f"Failed to create DSPy example from trace: {e}")
            return None

    def _extract_judge_instructions(self, judge: Judge) -> str:
        """
        Extract core instructions from judge for DSPy signature creation.

        Args:
            judge: The judge instance

        Returns:
            Instructions text for DSPy signature
        """
        # For now, use the judge's description as instructions
        # This can be extended to extract from judge prompts or other sources
        return judge.description

    def _create_dspy_signature(self, instructions: str) -> Any:
        """
        Create DSPy signature for judge evaluation.

        Args:
            instructions: Instructions text for the signature

        Returns:
            DSPy signature object
        """
        try:
            import dspy

            return dspy.make_signature(
                {
                    InstructionsJudge.get_template_variable_inputs(): (
                        str,
                        dspy.InputField(desc="Inputs to the model"),
                    ),
                    InstructionsJudge.get_template_variable_outputs(): (
                        str,
                        dspy.InputField(desc="Outputs from the model"),
                    ),
                    InstructionsJudge.get_template_variable_result(): (
                        str,
                        dspy.OutputField(desc="pass or fail based on the inputs and outputs"),
                    ),
                    InstructionsJudge.get_template_variable_rationale(): (
                        str,
                        dspy.OutputField(desc="Rationale explaining the pass or fail result"),
                    ),
                },
                instructions,
            )

        except ImportError:
            raise MlflowException("DSPy library is required but not installed")

    def _create_agreement_metric(self) -> Callable:
        """
        Create DSPy metric function for judge optimization.

        Returns:
            Metric function for DSPy optimization
        """

        def agreement_metric(example, pred, trace=None):
            """Simple agreement metric for judge optimization."""
            try:
                # Extract result from example and prediction
                expected = getattr(example, "result", None)
                predicted = getattr(pred, "result", None)

                if expected is None or predicted is None:
                    return False

                # Normalize both to consistent format
                expected_norm = str(expected).lower().strip()
                predicted_norm = str(predicted).lower().strip()

                return expected_norm == predicted_norm
            except Exception:
                # Return 0 for any errors
                return False

        return agreement_metric

    def _setup_dspy_model(self) -> Any:
        """
        Set up DSPy model from the optimizer's model URI.

        Returns:
            DSPy model instance configured for the optimizer's model
        """
        try:
            import dspy

            return dspy.LM(model=self._model)
        except ImportError:
            raise MlflowException("DSPy library is required but not installed")
        except Exception as e:
            raise MlflowException(
                f"Failed to initialize DSPy language model with model '{self._model}': {e!s}"
            )

    def align(self, judge: Judge, traces: list[Trace]) -> Judge:
        """
        Main alignment method that orchestrates the DSPy optimization process.

        1. Extract judge instructions and create DSPy signature
        2. Convert traces to DSPy examples
        3. Create and compile DSPy optimizer
        4. Generate optimized judge from results

        Args:
            judge: The judge to be optimized
            traces: List of traces containing alignment data.
                   The implementation will split these traces internally for train/validation.

        Returns:
            A new optimized Judge instance

        Raises:
            MlflowException: If optimization fails or insufficient data is provided
        """
        try:
            import dspy

            if not traces:
                raise MlflowException("No traces provided for alignment")

            # Set up DSPy context to use the optimizer's model
            self._logger.info(f"Setting up DSPy context with model: {self._model}")

            # Configure DSPy to use the optimizer's model
            # This ensures the optimizer uses its own model, separate from the judge's model
            dspy_model = self._setup_dspy_model()

            # Use DSPy context manager to ensure proper model usage
            with dspy.context(lm=dspy_model):
                # Extract judge instructions and create DSPy signature
                instructions = self._extract_judge_instructions(judge)
                self._logger.info(f"Extracted instructions: {instructions}")

                signature = self._create_dspy_signature(instructions)

                # Create DSPy program that will simulate the judge
                # The program should use the judge's model, not the optimizer's model
                judge_model = dspy.LM(model=judge.model)
                with dspy.context(lm=judge_model):
                    program = dspy.Predict(signature)
                self._logger.info("Created DSPy program with signature using judge's model")

                # Convert traces to DSPy format
                dspy_examples = []
                for trace in traces:
                    example = self._trace_to_dspy_example(trace, judge.name)
                    if example is not None:
                        dspy_examples.append(example)

                self._logger.info(
                    f"Created {len(dspy_examples)} valid examples from {len(traces)} traces"
                )

                if not dspy_examples:
                    raise MlflowException("No valid examples could be created from traces")

                if len(dspy_examples) < 2:
                    raise MlflowException("At least 2 valid examples are required for optimization")

                # Create metric function
                agreement_metric = self._create_agreement_metric()

                self._logger.info("Starting DSPy optimization...")

                # Use the algorithm-specific optimization method
                # Each implementation decides how to handle data splitting
                optimized_program = self._dspy_optimize(program, dspy_examples, agreement_metric)

                self._logger.info("DSPy optimization completed")

                # Create optimized judge - for now, return the original judge
                # This would be extended to create a new judge with optimized prompts
                return self._create_optimized_judge(judge, optimized_program.signature.instructions)

        except ImportError:
            raise MlflowException("DSPy library is required but not installed")
        except Exception as e:
            raise MlflowException(f"Alignment optimization failed: {e!s}")

    def _create_optimized_judge(self, original_judge: Judge, instructions: str) -> Judge:
        """
        Create a new optimized judge from the original judge and optimized instructions.

        Uses make_judge() to create a new InstructionsJudge with the optimized instructions
        from DSPy, preserving the original judge's name and model configuration.

        Args:
            original_judge: The original judge
            instructions: The optimized instructions from DSPy optimization

        Returns:
            A new optimized Judge instance with improved instructions
        """
        from mlflow.genai.judges import make_judge

        # Create a new judge with optimized instructions
        # Use the same name as the original judge
        optimized_name = original_judge.name

        # Get the model from the original judge using the model property
        judge_model = original_judge.model

        self._logger.info(
            f"Creating optimized judge '{optimized_name}' with DSPy-optimized instructions"
        )

        return make_judge(name=optimized_name, instructions=instructions, model=judge_model)
