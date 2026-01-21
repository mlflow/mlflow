import json
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from jinja2 import Template
from pydantic import BaseModel

from mlflow.genai.judges.optimizers.dspy_utils import construct_dspy_lm
from mlflow.genai.judges.optimizers.memalign.prompts import (
    DISTILLATION_PROMPT_TEMPLATE,
    create_examples_field,
    create_guidelines_field,
)

# Try to import litellm at module level
try:
    from litellm import get_max_tokens, token_counter

    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False

if TYPE_CHECKING:
    import dspy

_logger = logging.getLogger(__name__)

# Maximum tokens for model input (most models have this limit)
_MAX_MODEL_TOKENS = 8192


@lru_cache(maxsize=1)
def _get_model_max_tokens(model: str) -> int:
    """Get the maximum token limit for a model.

    Args:
        model: Model identifier (e.g., "openai/text-embedding-3-small")

    Returns:
        Maximum token limit for the model
    """
    if not _LITELLM_AVAILABLE:
        return _MAX_MODEL_TOKENS

    try:
        max_tokens = get_max_tokens(model)
        if max_tokens is not None:
            return max_tokens
    except Exception as e:
        _logger.debug(f"Error getting max tokens for model {model}: {e}", exc_info=True)
    return _MAX_MODEL_TOKENS


@lru_cache(maxsize=1000)
def truncate_to_token_limit(text: str, model: str) -> str:
    """Truncate text to fit within model's token limit.

    Args:
        text: Text to truncate
        model: Model identifier (e.g., "openai/text-embedding-3-small")

    Returns:
        Truncated text that fits within token limit
    """
    max_tokens = _get_model_max_tokens(model)

    if not _LITELLM_AVAILABLE:
        # Naive truncation to `max_tokens` characters in the text if litellm is not available
        _logger.warning(
            f"LiteLLM is required for accurate token counting, using naive truncation to "
            f"{max_tokens} characters. Please install litellm using: `pip install litellm`"
        )
        return text[:max_tokens]

    # Optimization to avoid token counting if number of characters is less than max tokens.
    if len(text) <= max_tokens:
        return text

    token_count = token_counter(model=model, text=text)
    if token_count <= max_tokens:
        return text

    original_token_count = token_count
    ratio = max_tokens / token_count
    truncated = text[: int(len(text) * ratio)]

    while token_counter(model=model, text=truncated) > max_tokens:
        truncated = truncated[: int(len(truncated) * 0.95)]

    _logger.debug(f"Truncated text from {original_token_count} to ~{max_tokens} tokens")
    return truncated


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

    # Truncate prompt to fit within model's token limit
    prompt = truncate_to_token_limit(prompt, reflection_lm)

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
                # Convert indices to actual trace IDs, handling both integer indices
                # and cases where the LLM returns trace IDs directly
                trace_ids = []
                for idx in source_trace_ids_raw:
                    if isinstance(idx, str) and idx.startswith("tr-"):
                        # LLM returned an actual trace ID
                        trace_ids.append(idx)
                    elif isinstance(idx, int):
                        trace_ids.append(index_to_trace_id.get(idx, f"unknown_{idx}"))
                    elif isinstance(idx, str) and idx.isdigit():
                        trace_ids.append(index_to_trace_id.get(int(idx), f"unknown_{idx}"))
                    else:
                        trace_ids.append(f"unknown_{idx}")
                guidelines.append(
                    Guideline(guideline_text=guideline_text, source_trace_ids=trace_ids)
                )
            else:
                guidelines.append(Guideline(guideline_text=guideline_text, source_trace_ids=None))

    return guidelines


def retrieve_relevant_examples(
    retriever: "dspy.retrievers.Embeddings",
    examples: list["dspy.Example"],
    query_kwargs: dict[str, Any],
    signature: "dspy.Signature",
) -> list[tuple["dspy.Example", str]]:
    """Retrieve relevant examples using semantic search.

    Args:
        retriever: DSPy Embeddings retriever
        examples: List of all examples
        query_kwargs: Query parameters to construct search query
        signature: DSPy signature defining input fields

    Returns:
        List of tuples of (retrieved example, trace ID)
    """
    if not examples or retriever is None:
        return []

    query_parts = [
        str(query_kwargs[field_name])
        for field_name in signature.input_fields
        if field_name in query_kwargs and query_kwargs[field_name] is not None
    ]
    query = " ".join(query_parts)
    search_results = retriever(query)
    indices = [int(i) for i in search_results.indices]

    # Return list of tuples for safer API
    return [
        (
            examples[i],
            examples[i]._trace_id if hasattr(examples[i], "_trace_id") else f"example_{i}",
        )
        for i in indices
    ]


def create_extended_signature(base_signature: "dspy.Signature") -> "dspy.Signature":
    """Create extended DSPy signature with guidelines and example judgements fields.

    Args:
        base_signature: Base DSPy signature to extend

    Returns:
        Extended signature with guidelines and example_judgements fields prepended
    """
    extended_sig = base_signature.prepend("guidelines", create_guidelines_field())
    return extended_sig.prepend("example_judgements", create_examples_field())
