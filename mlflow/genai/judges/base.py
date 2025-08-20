from typing import Any, Dict, List, Optional, Union

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
    
    _examples: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _prompt_template: Optional[str] = PrivateAttr(default=None)
    
    def __init__(
        self, 
        name: str, 
        instructions: str, 
        model: str,
        examples: Optional[List[Dict[str, Any]]] = None,
        aggregations: Optional[List] = None,
        **kwargs
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
            name=name, 
            instructions=instructions,
            model=model,
            aggregations=aggregations,
            **kwargs
        )
        self._examples = examples or []
        self._prompt_template = None
        
    def align(self, traces: List[Trace]) -> "Judge":
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
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        expectations: Optional[Dict[str, Any]] = None,
        trace: Optional[Trace] = None,
    ) -> Union[Assessment, Feedback]:
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
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        expectations: Optional[Dict[str, Any]] = None,
        trace: Optional[Trace] = None,
    ) -> str:
        """
        Prepare the prompt for the LLM based on instructions and data.
        
        Args:
            inputs: Input values
            outputs: Output values
            expectations: Expected values
            trace: MLflow trace
            
        Returns:
            Formatted prompt string
        """
        if trace and trace.data and trace.data.spans:
            root_span = trace.data.spans[0]
            inputs = inputs or root_span.inputs
            outputs = outputs or root_span.outputs
        
        prompt_parts = [
            f"Instructions: {self.instructions}",
            "",
        ]
        
        if self._examples:
            prompt_parts.append("Examples:")
            for i, example in enumerate(self._examples, 1):
                prompt_parts.append(f"Example {i}:")
                if "inputs" in example:
                    prompt_parts.append(f"  Inputs: {example['inputs']}")
                if "outputs" in example:
                    prompt_parts.append(f"  Outputs: {example['outputs']}")
                if "assessment" in example:
                    prompt_parts.append(f"  Assessment: {example['assessment']}")
                prompt_parts.append("")
        
        prompt_parts.append("Current evaluation:")
        if inputs:
            prompt_parts.append(f"Inputs: {inputs}")
        if outputs:
            prompt_parts.append(f"Outputs: {outputs}")
        if expectations:
            prompt_parts.append(f"Expectations: {expectations}")
        
        prompt_parts.append("")
        prompt_parts.append("Please evaluate based on the instructions above.")
        
        return "\n".join(prompt_parts)
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """
        Serialize judge to dictionary, leveraging Scorer's serialization.
        """
        from dataclasses import asdict
        from mlflow.genai.scorers.base import SerializedScorer, _SERIALIZATION_VERSION
        import mlflow
        
        judge_data = {
            "instructions": self.instructions,
            "model": self.model,
            "examples": self._examples,
        }
        
        serialized = SerializedScorer(
            name=self.name,
            aggregations=self.aggregations,
            mlflow_version=mlflow.__version__,
            serialization_version=_SERIALIZATION_VERSION,
            builtin_scorer_class="mlflow.genai.judges.base.Judge",
            builtin_scorer_pydantic_data=judge_data,
        )
        
        return asdict(serialized)
    
    @classmethod
    def model_validate(cls, obj: Any) -> "Judge":
        """
        Deserialize judge from dictionary.
        """
        from mlflow.genai.scorers.base import SerializedScorer
        
        serialized = SerializedScorer(**obj)
        
        if serialized.builtin_scorer_class != "mlflow.genai.judges.base.Judge":
            raise ValueError(f"Not a Judge serialization: {serialized.builtin_scorer_class}")
        
        judge_data = serialized.builtin_scorer_pydantic_data or {}
        
        judge = cls(
            name=serialized.name,
            instructions=judge_data.get("instructions"),
            model=judge_data.get("model"),
            examples=judge_data.get("examples", []),
            aggregations=serialized.aggregations,
        )
        
        return judge
    
    @property
    def kind(self) -> ScorerKind:
        """Judge is a special kind of builtin scorer."""
        return ScorerKind.BUILTIN