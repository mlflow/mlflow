from dataclasses import dataclass, field
from typing import Any, Dict, List

from mlflow.metrics.base import (
    EvaluationExample,
)
from mlflow.metrics.utils.prompt_template import (
    PromptTemplate,
)

# TODO: Update the default_mode and default_parameters to the correct values post experimentation
default_model = "openai:/gpt4"
default_parameters = {
    "temperature": 0.0,
    "max_tokens": 100,
    "top_p": 1.0,
}
grading_system_prompt_template = PromptTemplate(
    """
Please act as an impartial judge and evaluate the quality of the provided output which
attempts to produce output for the provided input based on a provided information.
You'll be given a grading format below which you'll call for each provided information,
input and provided output to submit your justification and score to compute the {name} of
the output.

Input:
{input}

Provided output:
{output}

{variables}

Metric definition:
{definition}

Grading criteria:
{grading_prompt}

{examples}

And you'll need to submit your grading for the {name} of the output,
using the following in json format:
Score: [your score number between 0 to 4 for the {name} of the output]
Justification: [your step by step reasoning about the {name} of the output]
    """
)


@dataclass
class EvaluationModel:
    """
    Useful to compute v1 prompt for make_genai_metric
    """

    name: str
    definition: str
    grading_prompt: str
    examples: List[EvaluationExample] = None
    model: str = default_model
    parameters: Dict[str, Any] = field(default_factory=lambda: default_parameters)

    def to_dict(self):
        examples_str = "" if self.examples is None else f"Examples:\n{self._format_examples()}"
        return {
            "model": self.model,
            "eval_prompt": grading_system_prompt_template.partial_fill(
                name=self.name,
                definition=self.definition,
                grading_prompt=self.grading_prompt,
                examples=examples_str,
            ),
            "parameters": self.parameters,
        }

    def _format_examples(self):
        return "\n".join(map(str, self.examples))


correctness_definition = (
    "Correctness refers to how well the generated output matches "
    "or aligns with the reference or ground truth text that is considered "
    "accurate and appropriate for the given input. The ground truth serves as "
    "a benchmark against which the provided output is compared to determine the "
    "level of accuracy and fidelity."
)

correctness_grading_prompt = (
    "Correctness: If the answer correctly answer the question, below "
    "are the details for different scores: "
    "- Score 1: the answer is completely incorrect, doesn’t mention anything about "
    "the question or is completely contrary to the correct answer. "
    "- Score 2: the answer provides some relevance to the question and answer "
    "one aspect of the question correctly. "
    "- Score 3: the answer mostly answer the question but is missing or hallucinating "
    "on one critical aspect. "
    "- Score 5: the answer correctly answer the question and not missing any "
    "major aspect"
)

correctness_variables = ["ground_truth"]
correctness_parameters = {"temperature": 1.0}
