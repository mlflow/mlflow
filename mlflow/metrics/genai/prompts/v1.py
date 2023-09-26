from dataclasses import dataclass, field
from typing import Any, Dict, List

from mlflow.metrics.base import (
    EvaluationExample,
)
from mlflow.metrics.genai.prompt_template import (
    PromptTemplate,
)

# TODO: Update the default_mode and default_parameters to the correct values post experimentation
default_model = "openai:/gpt4"
default_parameters = {
    "temperature": 0.0,
    "max_tokens": 200,
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

Below is your grading criteria:
{grading_prompt}

{examples}

And you'll need to submit your grading for the {name} of the output,
using the following in json format:
Score: [your score number between 1 to 5 for the {name} of the output]
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


@dataclass
class CorrectnessMetric:
    definition = (
        "Correctness is evaluated on the proximity of the provided output to the ground truth "
        "in terms of meaning and description similarity. Scores can be assigned based on the "
        "gradual similarity in meaning and description to the ground truth."
    )

    grading_prompt = (
        "Correctness: Below are the details for different scores:"
        "- Score 1: the output is completely incorrect, doesn't mention anything related to the "
        "input or is completely contrary to the provided ground truth."
        "- Score 2: the output provides some relevance to the input and answers one aspect of the "
        "question as in the ground truth."
        "- Score 3: the output mostly answers the question but is missing or hallucinating on "
        "one critical aspect."
        "- Score 5: the output correctly answers the question and is not missing any major aspect "
        "provided in the ground truth answer."
    )

    variables = ["ground_truth"]
    parameters = default_parameters

    example_score_2 = EvaluationExample(
        input="What is MLflow?",
        output="MLflow is an open-source platform.",
        score=2,
        justification="While the statement correctly identifies MLflow as an open-source platform, "
        "it lacks some critical aspects mentioned in the ground truth. Specifically, it doesn't "
        "provide information about MLflow's purpose in managing the end-to-end machine learning "
        "lifecycle, its development by Databricks, and its focus on addressing challenges faced by "
        "data scientists and machine learning engineers. Therefore, it answers one aspect of the "
        "question but is missing several critical aspects provided in the ground truth.",
        variables={"ground_truth": "MLflow is an open-source platform for managing the end-to-end "
                   "machine learning (ML) lifecycle. It was developed by Databricks, a company "
                   "that specializes in big data and machine learning solutions. MLflow is "
                   "designed to address the challenges that data scientists and machine learning "
                   "engineers face when developing, training, and deploying machine learning "
                   "models."}
    )

    example_score_4 = EvaluationExample(
        input="What is MLflow?",
        output="MLflow is an open-source platform for managing machine learning workflows, "
        "including experiment tracking, model packaging, versioning, and deployment, simplifying "
        "the ML lifecycle.",
        score=4,
        justification="The output effectively explains what MLflow is and its purpose. "
        "Information about the developer of MLflow could be included for a 5-score.",
        variables={"ground_truth": "MLflow is an open-source platform for managing the end-to-end "
                   "machine learning (ML) lifecycle. It was developed by Databricks, a company "
                   "that specializes in big data and machine learning solutions. MLflow is "
                   "designed to address the challenges that data scientists and machine learning "
                   "engineers face when developing, training, and deploying machine learning "
                   "models."}
    )
