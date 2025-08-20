"""
Factory functions for creating Judge instances.
"""

from typing import Any, Optional

from mlflow.genai.judges.base import Judge
from mlflow.genai.judges.utils import get_default_model
from mlflow.genai.scorers.registry import _get_scorer_store
from mlflow.tracking.fluent import _get_experiment_id
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

    .. code-block:: python

        import mlflow
        from mlflow.genai.judges import make_judge

        # Create a judge with specific model
        judge = make_judge(
            name="formality_checker",
            instructions="Check if the response uses formal language appropriate for business communication",
            model="openai/gpt-4o-mini"
        )

        # Use the judge to evaluate outputs
        assessment = judge(
            inputs={"question": "How do I reset my password?"},
            outputs="Hey! Just click the forgot password link and you're good to go!"
        )
        print(assessment.value)  # False (informal language)

        # Create a judge without specifying model (uses default)
        relevance_judge = make_judge(
            name="relevance_checker",
            instructions="Determine if the answer directly addresses the question asked"
        )
    """
    if model is None:
        model = get_default_model()
    
    return Judge(
        name=name,
        instructions=instructions,
        model=model,
    )


@experimental(version="3.4.0")
def register_judge(
    name: str,
    instructions: str,
    model: Optional[str] = None,
    examples: Optional[list] = None,
    experiment_id: Optional[str] = None,
) -> int:
    """
    Register a judge as a scorer with versioning support.
    
    Args:
        name: Unique name for the judge
        instructions: Plain language instructions for what to evaluate
        model: LLM model to use (defaults to configured default)
        examples: Optional few-shot examples for alignment
        experiment_id: Experiment to register the judge in (defaults to active experiment)
        
    Returns:
        Version number of the registered judge
        
    Example:

    .. code-block:: python

        import mlflow
        from mlflow.genai.judges import register_judge

        # Register v1 of a judge
        v1 = register_judge(
            name="formality_judge",
            instructions="Check if response uses formal language",
            model="openai/gpt-4o-mini"
        )
        print(f"Registered judge version {v1}")

        # Register v2 with examples from alignment
        v2 = register_judge(
            name="formality_judge",
            instructions="Check if response uses formal language",
            examples=[
                {"inputs": {"q": "Hi"}, "outputs": "Hey!", "assessment": False},
                {"inputs": {"q": "Hello"}, "outputs": "Good day.", "assessment": True}
            ]
        )
        print(f"Registered judge version {v2}")
    """
    
    judge = make_judge(name=name, instructions=instructions, model=model)
    
    if examples:
        judge._examples = examples
    
    if experiment_id is None:
        experiment_id = _get_experiment_id()
    
    store = _get_scorer_store()
    version = store.register_scorer(experiment_id, judge)
    
    return version


@experimental(version="3.4.0")
def load_judge(
    name: str,
    version: Optional[int] = None,
    experiment_id: Optional[str] = None,
) -> Judge:
    """
    Load a judge from the scorer registry.
    
    Args:
        name: Name of the judge
        version: Optional version number (defaults to latest)
        experiment_id: Experiment ID (defaults to active experiment)
        
    Returns:
        Judge instance
        
    Example:

    .. code-block:: python

        import mlflow
        from mlflow.genai.judges import load_judge

        # Load latest version
        judge = load_judge("formality_judge")

        # Load specific version
        judge_v1 = load_judge("formality_judge", version=1)
    """
   
    if experiment_id is None:
        experiment_id = _get_experiment_id()
    
    store = _get_scorer_store()
    scorer = store.get_scorer(experiment_id, name, version)
    
    if not isinstance(scorer, Judge):
        raise ValueError(f"Scorer '{name}' is not a Judge")
    
    return scorer


@experimental(version="3.4.0")
def list_judge_versions(
    name: str,
    experiment_id: Optional[str] = None,
) -> list[tuple[Judge, int]]:
    """
    List all versions of a judge.
    
    Args:
        name: Name of the judge
        experiment_id: Experiment ID (defaults to active experiment)
        
    Returns:
        List of (judge, version) tuples
        
    Example:

    .. code-block:: python

        import mlflow
        from mlflow.genai.judges import list_judge_versions

        versions = list_judge_versions("formality_judge")
        for judge, version in versions:
            print(f"Version {version}: {judge.instructions[:50]}...")
    """
    
    if experiment_id is None:
        experiment_id = _get_experiment_id()
    
    store = _get_scorer_store()
    scorer_versions = store.list_scorer_versions(experiment_id, name)
    
    judge_versions = []
    for scorer, version in scorer_versions:
        if isinstance(scorer, Judge):
            judge_versions.append((scorer, version))
    
    return judge_versions


@experimental(version="3.4.0")
def make_judge_from_dspy(
    name: str,
    program: Any,
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

    .. code-block:: python

        import dspy
        import mlflow
        from mlflow.genai.judges import make_judge_from_dspy

        # Define a DSPy signature for the judge
        class AccuracyJudge(dspy.Signature):
            \"\"\"Assess the factual accuracy of the response.\"\"\"
            question = dspy.InputField()
            answer = dspy.InputField()
            assessment = dspy.OutputField(desc="accurate/inaccurate with reasoning")

        # Create a DSPy program
        program = dspy.ChainOfThought(AccuracyJudge)

        # Convert to MLflow Judge
        judge = make_judge_from_dspy(
            name="accuracy_judge",
            program=program,
            model="openai/gpt-4"
        )
    """
    # TODO: Implement this
    raise NotImplementedError(
        "Creating judges from DSPy programs is not yet implemented. "
        "This feature will be available in a future release. "
        "For now, please use make_judge() with plain language instructions."
    )