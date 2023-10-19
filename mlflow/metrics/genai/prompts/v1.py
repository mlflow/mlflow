from dataclasses import dataclass, field
from typing import Any, Dict, List

from mlflow.metrics.base import (
    EvaluationExample,
)
from mlflow.metrics.genai.prompt_template import (
    PromptTemplate,
)

# TODO: Update the default_mode and default_parameters to the correct values post experimentation
default_model = "openai:/gpt-4"
default_parameters = {
    "temperature": 0.0,
    "max_tokens": 200,
    "top_p": 1.0,
}
grading_system_prompt_template = PromptTemplate(
    """
Task:
You are an impartial judge. You will be given an input that was sent to a machine
learning model, and you will be given an output that the model produced. You
may also be given additional information that was used by the model to generate the output.

Your task is to determine a numerical score called {name} based on the input and output.
A definition of {name} and a grading rubric are provided below.
You must use the grading rubric to determine your score. You must also justify your score.

Examples could be included below for reference. Make sure to use them as references and to
understand them before completing the task.

Input:
{input}

Output:
{output}

{grading_context_columns}

Metric definition:
{definition}

Grading rubric:
{grading_prompt}

{examples}

You must return the following fields in your response one below the other:
score: Your numerical score for the model's {name} based on the rubric
justification: Your step-by-step reasoning about the model's {name} score
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
class AnswerSimilarityMetric:
    definition = (
        "Answer similarity is evaluated on the degree of semantic similarity of the provided "
        "output to the provided targets, which is the ground truth. Scores can be assigned based "
        "on the gradual similarity in meaning and description to the provided targets, where a "
        "higher score indicates greater alignment between the provided output and provided targets."
    )

    grading_prompt = (
        "Answer similarity: Below are the details for different scores:\n"
        "- Score 1: the output has little to no semantic similarity to the provided targets.\n"
        "- Score 2: the output displays partial semantic similarity to the provided targets on "
        "some aspects.\n"
        "- Score 3: the output has moderate semantic similarity to the provided targets.\n"
        "- Score 4: the output aligns with the provided targets in most aspects and has "
        "substantial semantic similarity.\n"
        "- Score 5: the output closely aligns with the provided targets in all significant aspects."
    )

    grading_context_columns = ["targets"]
    parameters = default_parameters
    default_model = default_model

    example_score_2 = EvaluationExample(
        input="What is MLflow?",
        output="MLflow is an open-source platform.",
        score=2,
        justification="The provided output is partially similar to the target, as it captures the "
        "general idea that MLflow is an open-source platform. However, it lacks the comprehensive "
        "details and context provided in the target about MLflow's purpose, development, and "
        "challenges it addresses. Therefore, it demonstrates partial, but not complete, "
        "semantic similarity.",
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
        justification="The provided output aligns closely with the target. It covers various key "
        "aspects mentioned in the target, including managing machine learning workflows, "
        "experiment tracking, model packaging, versioning, and deployment. While it may not include"
        " every single detail from the target, it demonstrates substantial semantic similarity.",
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
class AnswerCorrectnessMetric:
    definition = (
        "Answer correctness is evaluated on the accuracy of the provided output based on the "
        "provided targets, which is the ground truth. Scores can be assigned based on the degree "
        "of semantic similarity and factual correctness of the provided output to the provided "
        "targets, where a higher score indicates higher degree of accuracy."
    )

    grading_prompt = (
        "Answer Correctness: Below are the details for different scores:\n"
        "- Score 1: the output is completely incorrect. It is completely different from or "
        "contradicts the provided targets.\n"
        "- Score 2: the output demonstrates some degree of semantic similarity and includes "
        "partially correct information. However, the output still has significant discrepancies "
        "with the provided targets or inaccuracies.\n"
        "- Score 3: the output addresses a couple of aspects of the input accurately, aligning "
        "with the provided targets. However, there are still omissions or minor inaccuracies.\n"
        "- Score 4: the output is mostly correct. It provides mostly accurate information, but "
        "there may be one or more minor omissions or inaccuracies.\n"
        "- Score 5: the output is correct. It demonstrates a high degree of accuracy and "
        "semantic similarity to the targets."
    )

    grading_context_columns = ["targets"]
    parameters = default_parameters
    default_model = default_model

    example_score_0 = EvaluationExample(
        input="Is MLflow open-source?",
        output="No, MLflow is not open-source.",
        score=1,
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
