"""
Factory functions for creating Judge instances.
"""

from typing import Any, Optional

from mlflow.genai.judges.base import Judge
from mlflow.genai.judges.utils import get_default_model
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
def make_judge(
    name: str,
    instructions: str,
    model: Optional[str] = None,
) -> Judge:
    """
    Create a new judge from declarative instructions.
    
    Args:
        name: Unique name for the judge
        instructions: Plain language instructions for what to evaluate
        model: LLM model to use (defaults to configured default on Databricks)
        
    Returns:
        A new Judge instance
        
    Example:
        >>> judge = make_judge(
        ...     name="formality_checker",
        ...     instructions="Check if the response uses formal language appropriate for business communication",
        ...     model="openai/gpt-4o-mini"
        ... )
        >>> 
        >>> # Use the judge to evaluate outputs
        >>> assessment = judge(
        ...     inputs={"question": "How do I reset my password?"},
        ...     outputs="Hey! Just click the forgot password link and you're good to go!"
        ... )
        >>> print(assessment.value)  # False (informal language)
        
        >>> # Create a judge without specifying model (uses default)
        >>> relevance_judge = make_judge(
        ...     name="relevance_checker",
        ...     instructions="Determine if the answer directly addresses the question asked"
        ... )
    """
    # Use default model if not specified
    if model is None:
        model = get_default_model()
    
    return Judge(
        name=name,
        instructions=instructions,
        model=model,
        version=1,
    )


@experimental(version="3.4.0")
def make_judge_from_dspy(
    name: str,
    program: Any,  # DSPy program
    model: Optional[str] = None,
) -> Judge:
    """
    Create a judge from a DSPy program.
    
    This is a future enhancement that will allow creating judges from
    DSPy optimization programs that can be automatically aligned.
    
    Args:
        name: Unique name for the judge
        program: DSPy program defining the judge behavior
        model: LLM model to use (defaults to configured default)
        
    Returns:
        A new Judge instance
        
    Example:
        >>> import dspy
        >>> 
        >>> # Define a DSPy signature for the judge
        >>> class AccuracyJudge(dspy.Signature):
        ...     \"\"\"Assess the factual accuracy of the response.\"\"\"
        ...     question = dspy.InputField()
        ...     answer = dspy.InputField()
        ...     assessment = dspy.OutputField(desc="accurate/inaccurate with reasoning")
        >>> 
        >>> # Create a DSPy program
        >>> program = dspy.ChainOfThought(AccuracyJudge)
        >>> 
        >>> # Convert to MLflow Judge
        >>> judge = make_judge_from_dspy(
        ...     name="accuracy_judge",
        ...     program=program,
        ...     model="openai/gpt-4"
        ... )
    """
    raise NotImplementedError(
        "Creating judges from DSPy programs is not yet implemented. "
        "This feature will be available in a future release. "
        "For now, please use make_judge() with plain language instructions."
    )