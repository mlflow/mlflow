import logging
from typing import Any

import dspy
from jinja2 import Template
from pydantic import BaseModel

from mlflow.genai.judges.optimizers.dspy_utils import construct_dspy_lm
from mlflow.genai.judges.optimizers.memalign.prompts import DISTILLATION_PROMPT_TEMPLATE

_logger = logging.getLogger(__name__)


class Guideline(BaseModel):
    guideline_text: str
    source_ids: list[int] | None = None


class Guidelines(BaseModel):
    guidelines: list[Guideline]


def get_default_embedding_model() -> str:
    return "openai/text-embedding-3-small"


def distill_guidelines(
    examples: list["dspy.Example"],
    signature: "dspy.Signature",
    judge_instructions: str,
    distillation_model: str,
    existing_guidelines: list[str],
) -> list[str]:
    """Distill general guidelines from feedback examples.

    Args:
        examples: List of DSPy examples containing feedback
        signature: DSPy signature defining input/output fields
        judge_instructions: Original judge instructions
        distillation_model: Model to use for distillation
        existing_guidelines: Previously distilled guidelines

    Returns:
        List of newly distilled guidelines (not including existing ones)
    """
    if not examples:
        return []

    examples_data = [dict(example) for example in examples]

    template = Template(DISTILLATION_PROMPT_TEMPLATE)
    prompt = template.render(
        judge_instructions=judge_instructions,
        feedback_records=examples_data,
        ids=list(range(len(examples_data))),
        existing_guidelines=existing_guidelines,
        # Pass zip and len as globals for Jinja2 template to iterate over examples with indices
        zip=zip,
        len=len,
    )

    try:
        distillation_lm = construct_dspy_lm(distillation_model)
        response = distillation_lm(
            messages=[{"role": "user", "content": prompt}],
            response_format=Guidelines,
        )[0]
        result = Guidelines.model_validate_json(response)

        return [
            g.guideline_text
            for g in result.guidelines
            if g.guideline_text not in existing_guidelines
        ]

    except Exception as e:
        _logger.error(f"Failed to distill guidelines: {e}")
        return []


def retrieve_relevant_examples(
    search: "dspy.retrievers.Embeddings",
    examples: list["dspy.Example"],
    query_kwargs: dict[str, Any],
    signature: "dspy.Signature",
) -> tuple[list["dspy.Example"], list[int]]:
    """Retrieve relevant examples using semantic search.

    Args:
        search: DSPy Embeddings retriever
        examples: List of all examples
        query_kwargs: Query parameters to construct search query
        signature: DSPy signature defining input fields

    Returns:
        Tuple of (retrieved examples, their indices)
    """
    if not examples or search is None:
        return [], []

    query_parts = [
        str(query_kwargs[field_name])
        for field_name in signature.input_fields
        if field_name in query_kwargs and query_kwargs[field_name] is not None
    ]
    query = " ".join(query_parts)
    search_results = search(query)
    indices = [int(i) for i in search_results.indices]
    retrieved_examples = [examples[i] for i in indices]

    return retrieved_examples, indices
