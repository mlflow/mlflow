import json
import logging
from typing import TYPE_CHECKING, Any

from jinja2 import Template
from pydantic import BaseModel

from mlflow.genai.judges.optimizers.dspy_utils import construct_dspy_lm
from mlflow.genai.judges.optimizers.memalign.prompts import (
    DISTILLATION_PROMPT_TEMPLATE,
    create_examples_field,
    create_guidelines_field,
)

if TYPE_CHECKING:
    import dspy

_logger = logging.getLogger(__name__)


class Guideline(BaseModel):
    guideline_text: str
    source_trace_ids: list[str] | None = None


class Guidelines(BaseModel):
    guidelines: list[Guideline]


def get_default_embedding_model() -> str:
    return "openai/text-embedding-3-small"


def distill_guidelines(
    examples: list["dspy.Example"],
    signature: "dspy.Signature",
    judge_instructions: str,
    reflection_lm: str,
    existing_guidelines: list[str],
) -> list[Guideline]:
    """Distill general guidelines from feedback examples.

    Args:
        examples: List of DSPy examples containing feedback (with _trace_id attribute)
        signature: DSPy signature defining input/output fields
        judge_instructions: Original judge instructions
        reflection_lm: Model to use for distillation
        existing_guidelines: Previously distilled guidelines

    Returns:
        List of newly distilled Guideline objects (not including existing ones)

    TODO: Add batching logic when number of examples exceeds threshold (e.g., 50-100)
          to prevent context explosion during distillation.
    """
    if not examples:
        return []

    examples_data = [dict(example) for example in examples]
    # Create index to trace_id mapping
    indices = list(range(len(examples_data)))
    index_to_trace_id = {
        i: example._trace_id if hasattr(example, "_trace_id") else f"example_{i}"
        for i, example in enumerate(examples)
    }

    template = Template(DISTILLATION_PROMPT_TEMPLATE)
    prompt = template.render(
        judge_instructions=judge_instructions,
        feedback_records=examples_data,
        ids=indices,
        existing_guidelines=existing_guidelines,
        # Pass zip and len as globals for Jinja2 template to iterate over examples with indices
        zip=zip,
        len=len,
    )

    distillation_lm = construct_dspy_lm(reflection_lm)
    response = distillation_lm(
        messages=[{"role": "user", "content": prompt}],
        response_format=Guidelines,
    )[0]

    # Parse JSON manually to convert integer indices to trace IDs before Pydantic validation
    response_data = json.loads(response)

    guidelines = []
    for guideline_data in response_data.get("guidelines", []):
        guideline_text = guideline_data.get("guideline_text")

        if guideline_text and guideline_text not in existing_guidelines:
            source_trace_ids_raw = guideline_data.get("source_trace_ids")

            if source_trace_ids_raw is not None:
                # Convert integers (indices) to actual trace IDs
                trace_ids = [
                    index_to_trace_id.get(
                        int(idx) if isinstance(idx, (int, str)) else idx, f"unknown_{idx}"
                    )
                    for idx in source_trace_ids_raw
                ]
                guidelines.append(
                    Guideline(guideline_text=guideline_text, source_trace_ids=trace_ids)
                )
            else:
                guidelines.append(Guideline(guideline_text=guideline_text, source_trace_ids=None))

    return guidelines


def retrieve_relevant_examples(
    search: "dspy.retrievers.Embeddings",
    examples: list["dspy.Example"],
    query_kwargs: dict[str, Any],
    signature: "dspy.Signature",
) -> tuple[list["dspy.Example"], list[str]]:
    """Retrieve relevant examples using semantic search.

    Args:
        search: DSPy Embeddings retriever
        examples: List of all examples
        query_kwargs: Query parameters to construct search query
        signature: DSPy signature defining input fields

    Returns:
        Tuple of (retrieved examples, their trace IDs)
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

    # Convert indices to trace IDs
    trace_ids = [
        examples[i]._trace_id if hasattr(examples[i], "_trace_id") else f"example_{i}"
        for i in indices
    ]

    return retrieved_examples, trace_ids


def create_extended_signature(base_signature: "dspy.Signature") -> "dspy.Signature":
    """Create extended DSPy signature with guidelines and example judgements fields.

    Args:
        base_signature: Base DSPy signature to extend

    Returns:
        Extended signature with guidelines and example_judgements fields prepended
    """
    extended_sig = base_signature.prepend("guidelines", create_guidelines_field())
    return extended_sig.prepend("example_judgements", create_examples_field())
