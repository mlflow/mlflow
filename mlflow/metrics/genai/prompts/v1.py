from dataclasses import dataclass, field
from typing import Any, Dict, List

from mlflow.metrics.base import (
    EvaluationExample,
)
from mlflow.metrics.genai.prompt_template import (
    PromptTemplate,
)

# TODO: Update the default_mode and default_parameters to the correct values post experimentation
default_model = "openai:/gpt-3.5-turbo-16k"
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

{grading_context_columns}

Metric definition:
{definition}

Below is your grading criteria:
{grading_prompt}

{examples}

And you'll need to submit your grading for the {name} of the output,
using the following in json format:
Score: [your score number for the {name} of the output]
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
        examples_str = (
            ""
            if self.examples is None or len(self.examples) == 0
            else f"Examples:\n{self._format_examples()}"
        )
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
        "in terms of meaning and description similarity. Scores can be assigned from 1 to 5 based "
        "on the gradual similarity in meaning and description to the ground truth."
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

    grading_context_columns = ["targets"]
    parameters = default_parameters
    default_model = default_model

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
        grading_context={
            "targets": "MLflow is an open-source platform for managing the end-to-end "
            "machine learning (ML) lifecycle. It was developed by Databricks, a company "
            "that specializes in big data and machine learning solutions. MLflow is "
            "designed to address the challenges that data scientists and machine learning "
            "engineers face when developing, training, and deploying machine learning "
            "models."
        },
    )

    example_score_4 = EvaluationExample(
        input="What is MLflow?",
        output="MLflow is an open-source platform for managing machine learning workflows, "
        "including experiment tracking, model packaging, versioning, and deployment, simplifying "
        "the ML lifecycle.",
        score=4,
        justification="The output effectively explains what MLflow is and its purpose. "
        "Information about the developer of MLflow could be included for a 5-score.",
        grading_context={
            "targets": "MLflow is an open-source platform for managing the end-to-end "
            "machine learning (ML) lifecycle. It was developed by Databricks, a company "
            "that specializes in big data and machine learning solutions. MLflow is "
            "designed to address the challenges that data scientists and machine learning "
            "engineers face when developing, training, and deploying machine learning "
            "models."
        },
    )

    default_examples = [example_score_2, example_score_4]


@dataclass
class RelevanceMetric:
    definition = (
        "Relevance encompasses the appropriateness, significance, and applicability of the output "
        "with respect to the input and context. Scores should range from 1 to 5 and should reflect "
        "the extent to which the output directly addresses the question provided in the input, "
        "given the provided context."
    )

    grading_prompt = (
        "Relevance: Below are the details for different scores:"
        "- Score 1: the output doesn't mention anything about the question or is completely "
        "irrelevant to the provided context."
        "- Score 2: the output provides some relevance to the question and is somehow related to "
        "the provided context."
        "- Score 3: the output mostly answers the question and is consistent with the provided "
        "context."
        "- Score 5: the output answers the question comprehensively using the provided context."
    )

    grading_context_columns = ["context"]
    parameters = default_parameters
    default_model = default_model

    example_score_2 = EvaluationExample(
        input="How is MLflow related to Databricks?",
        output="Databricks is a data engineering and analytics platform designed to help "
        "organizations process and analyze large amounts of data. Databricks is a company "
        "specializing in big data and machine learning solutions.",
        score=2,
        justification="The output provides relevant information about Databricks, mentioning it as "
        "a company specializing in big data and machine learning solutions. However, it doesn't "
        "directly address how MLflow is related to Databricks, which is the specific question "
        "asked in the input. Therefore, the output is only somewhat related to the provided "
        "context.",
        grading_context={
            "context": "MLflow is an open-source platform for managing the end-to-end machine "
            "learning (ML) lifecycle. It was developed by Databricks, a company that specializes "
            "in big data and machine learning solutions. MLflow is designed to address the "
            "challenges that data scientists and machine learning engineers face when developing, "
            "training, and deploying machine learning models."
        },
    )

    example_score_4 = EvaluationExample(
        input="How is MLflow related to Databricks?",
        output="MLflow is a product created by Databricks to enhance the efficiency of machine "
        "learning processes.",
        score=4,
        justification="The output provides a relevant and accurate statement about the "
        "relationship between MLflow and Databricks. While it doesn't provide extensive detail, "
        "it still offers a substantial and meaningful response. To achieve a score of 5, the "
        "response could be further improved by providing additional context or details about"
        "how MLflow specifically functions within the Databricks ecosystem.",
        grading_context={
            "context": "MLflow is an open-source platform for managing the end-to-end "
            "machine learning (ML) lifecycle. It was developed by Databricks, a company "
            "that specializes in big data and machine learning solutions. MLflow is "
            "designed to address the challenges that data scientists and machine learning "
            "engineers face when developing, training, and deploying machine learning "
            "models."
        },
    )

    default_examples = [example_score_2, example_score_4]


@dataclass
class StrictCorrectnessMetric:
    definition = (
        "When a question demands a specific value, term, or description (e.g., math questions or "
        "fact-checking), correctness is binary. Strict correctness of the output is assessed on "
        "whether it aligns exactly with the ground truth. Scores are assigned to be 0 or 1."
    )

    grading_prompt = (
        "Strict Correctness: Below are the details for different scores:"
        "- Score 0: the output is completely incorrect, doesn't mention anything about the "
        "question or is completely contrary to the ground truth."
        "- Score 1: the output answers the question correctly as provided in the ground truth."
    )

    grading_context_columns = ["targets"]
    parameters = default_parameters
    default_model = default_model

    example_score_0 = EvaluationExample(
        input="Is MLflow open-source?",
        output="No, MLflow is not open-source.",
        score=0,
        justification="The output is incorrect. It states that MLflow is not open-source, which "
        "contradicts the provided context, where it is explicitly mentioned that MLflow is an "
        "open-source platform. This directly opposes the ground truth, resulting in a score of 0 "
        "for strict correctness.",
        grading_context={
            "targets": "MLflow is an open-source platform for managing the end-to-end machine "
            "learning (ML) lifecycle. It was developed by Databricks, a company that specializes "
            "in big data and machine learning solutions. MLflow is designed to address the "
            "challenges that data scientists and machine learning engineers face when developing, "
            "training, and deploying machine learning models."
        },
    )

    example_score_1 = EvaluationExample(
        input="Is MLflow open-source?",
        output="MLflow is open-source, which means it's freely available for anyone to use.",
        score=1,
        justification="The output correctly states that MLflow is open-source, aligning perfectly "
        "with the provided context. It accurately reflects the ground truth information, earning "
        "a score of 1 for strict correctness.",
        grading_context={
            "targets": "MLflow is an open-source platform for managing the end-to-end machine "
            "learning (ML) lifecycle. It was developed by Databricks, a company that specializes "
            "in big data and machine learning solutions. MLflow is designed to address the "
            "challenges that data scientists and machine learning engineers face when developing, "
            "training, and deploying machine learning models."
        },
    )

    default_examples = [example_score_0, example_score_1]
