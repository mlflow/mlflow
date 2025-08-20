from typing import Any

from pydantic import Field, PrivateAttr

from mlflow.entities.assessment import Assessment, Feedback
from mlflow.entities.trace import Trace
from mlflow.genai.scorers.base import Scorer, ScorerKind
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
class Judge(Scorer):
    """
    A special type of scorer that can be aligned with human feedback.

    Judges are LLM-based scorers that evaluate inputs/outputs based on
    declarative instructions and can be improved through alignment with
    human-labeled data.
    """

    instructions: str = Field(..., description="Human-readable instructions for what to evaluate")
    model: str = Field(..., description="LLM model identifier")

    _examples: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _prompt_template: str | None = PrivateAttr(default=None)

    def __init__(
        self,
        name: str,
        instructions: str,
        model: str,
        examples: list[dict[str, Any]] | None = None,
        aggregations: list[Any] | None = None,
        **kwargs,
    ):
        """
        Initialize a Judge.

        Args:
            name: Unique identifier for the judge
            instructions: Human-readable instructions defining what the judge evaluates
            model: LLM model identifier (LiteLLM format, e.g., "openai/gpt-4o-mini")
            examples: Optional few-shot examples for alignment
            aggregations: Optional aggregation functions for the judge's output
        """
        super().__init__(
            name=name, instructions=instructions, model=model, aggregations=aggregations, **kwargs
        )
        self._examples = examples or []
        self._prompt_template = None

    def align(self, traces: list[Trace]) -> "Judge":
        """
        Create an aligned version of this judge based on labeled traces.

        Args:
            traces: List of traces with expectations (human labels)

        Returns:
            A new Judge instance with improved alignment
        """
        # TODO: Implement alignment logic
        raise NotImplementedError(
            "Judge alignment is not yet implemented. This will be available in a future release."
        )

    @property
    def description(self) -> str:
        """Human-readable description of the judge."""
        return self.instructions

    def __call__(
        self,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Assessment | Feedback:
        """
        Evaluate inputs/outputs or a trace.

        Args:
            inputs: Input values for evaluation
            outputs: Output values for evaluation
            expectations: Expected values (optional)
            trace: MLflow trace (alternative to inputs/outputs)

        Returns:
            Assessment with the judge's evaluation
        """
        from mlflow.genai.judges.utils import invoke_judge_model

        prompt = self._prepare_prompt(inputs, outputs, expectations, trace)

        result = invoke_judge_model(
            model=self.model,
            prompt=prompt,
            name=self.name,
        )

        if isinstance(result, (str, bool, int, float)):
            result = Feedback(
                name=self.name,
                value=result,
                rationale=f"Evaluated by {self.name} judge using {self.model}",
            )

        return result

    def _prepare_prompt(
        self,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> str:
        """
        Prepare the prompt for the LLM based on instructions and data.
        """
        # TODO: Implement proper prompt generation logic
        return self.instructions

    @property
    def kind(self) -> ScorerKind:
        """Judge is a special kind of builtin scorer."""
        return ScorerKind.BUILTIN
