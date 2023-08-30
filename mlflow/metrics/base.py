from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MetricValue:
    """
    The value of a metric.
    :param scores: The value of the metric per row
    :param justifications: The justification (if applicable) for the respective score
    :param aggregate_results: A dictionary mapping the name of the aggregation to its value
    """

    scores: List[float] = None
    justifications: List[float] = None
    aggregate_results: Dict[str, float] = None


@dataclass
class EvaluationExample:
    """
    Stores the sample example during few shot learning during LLM evaluation
    """

    input: str
    output: str
    score: float
    justification: str = None
    variables: Dict[str, str] = None

    def toString(self) -> str:
        variables = (
            ""
            if self.variables is None
            else "\n".join([f"Provided {key}: {value}" for key, value in self.variables.items()])
        )

        justification = ""
        if self.justification is not None:
            justification = f"Justification: {self.justification}\n"

        return f"""
        Input: {self.input}
        Provided output: {self.output}
        {variables}
        Score: {self.score}
        {justification}
        """
