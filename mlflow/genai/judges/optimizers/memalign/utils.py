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


def _find_optimal_batch_size(
    examples_data: list[dict],
    indices: list[int],
    judge_instructions: str,
    existing_guidelines: list[str],
    reflection_lm: str,
    distillation_lm: "dspy.LM",
    max_input_tokens: int,
) -> int:
    """Find the optimal batch size for distillation based on token limits.

    Starts with a maximum of 50 records and reduces the batch size using binary search until
    the prompt fits within token limits and the LM call succeeds.

    Args:
        examples_data: List of feedback example data dicts
        indices: List of indices corresponding to examples_data
        judge_instructions: Original judge instructions
        existing_guidelines: Previously distilled guidelines
        reflection_lm: Model to use for distillation
        distillation_lm: DSPy LM instance for distillation
        max_input_tokens: Maximum input tokens for the model
    
    Returns:
        Optimal number of records per batch that fits within token limits
    """
    # Reserve tokens for response and account for variance in prompt length
    flex_tokens = 5000
    prompt_tokens_limit = max_input_tokens - flex_tokens

    template = Template(DISTILLATION_PROMPT_TEMPLATE)
    records_per_group = min(len(examples_data), 50)

    while records_per_group > 0:
        prompt = template.render(
            judge_instructions=judge_instructions,
            feedback_records=examples_data[:records_per_group],
            ids=indices[:records_per_group],
            existing_guidelines=existing_guidelines,
            zip=zip,
            len=len,
        )

        if not _LITELLM_AVAILABLE:
            # Without litellm, use naive white-space-based estimation
            prompt_tokens_estimate = len(prompt.split())
        else:
            # Use litellm to estimate token count
            prompt_tokens_estimate = token_counter(model=reflection_lm, text=prompt)

        if prompt_tokens_estimate <= prompt_tokens_limit: # Found potentially acceptable batch size
            # Do a trial LM call to verify if prompt can fit into LM context window
            try:
                trial_response = distillation_lm(
                    messages=[{"role": "user", "content": prompt}],
                    response_format=Guidelines,
                )[0]
                Guidelines.model_validate_json(trial_response)
                return records_per_group
            except Exception:
                _logger.debug(
                    f"Trial LM call failed with batch size {records_per_group}, reducing"
                )
                records_per_group //= 2
        else: # Prompt still too large, reduce batch size
            records_per_group //= 2
    return 0


def _process_batch_response(
    response: str,
    index_to_trace_id: dict[int, str],
    existing_guideline_texts: set[str],
) -> list[Guideline]:
    """Parse LM response and convert to Guideline objects, filtering duplicates.
    
    Args:
        response: LM response in JSON format
        index_to_trace_id: Mapping from example indices to trace IDs
        existing_guideline_texts: Set of already existing guideline texts to avoid duplicates

    Returns:
        List of Guideline objects parsed from the response, excluding duplicates
    """
    response_data = json.loads(response)
    guidelines = []

    for guideline_data in response_data.get("guidelines", []):
        # Skip empty or duplicate guidelines
        guideline_text = guideline_data.get("guideline_text")
        if not guideline_text or guideline_text in existing_guideline_texts:
            continue
        
        # Skip guidelines without valid source trace IDs
        source_trace_ids_raw = guideline_data.get("source_trace_ids")
        if source_trace_ids_raw is None:
            continue
      
        # Map indices back to trace IDs, ignoring invalid indices
        trace_ids = [
            trace_id
            for idx in source_trace_ids_raw
            if (
                trace_id := index_to_trace_id.get(
                    int(idx) if isinstance(idx, (int, str)) else idx
                )
            )
            is not None
        ]
        # Only add guideline if there is at least one valid trace ID
        if trace_ids:
          guidelines.append(
              Guideline(
                  guideline_text=guideline_text,
                  source_trace_ids=trace_ids,
              )
          )

    return guidelines


def distill_guidelines(
    examples: list["dspy.Example"],
    judge_instructions: str,
    reflection_lm: str,
    existing_guidelines: list[str],
) -> list[Guideline]:
    """Distill general guidelines from feedback examples.

    Handles large batches by splitting them into smaller groups based on token limits.

    Args:
        examples: List of DSPy examples containing feedback (with _trace_id attribute)
        judge_instructions: Original judge instructions
        reflection_lm: Model to use for distillation
        existing_guidelines: Previously distilled guidelines

    Returns:
        List of newly distilled Guideline objects (not including existing ones)
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

    distillation_lm = construct_dspy_lm(reflection_lm)
    max_input_tokens = _get_model_max_tokens(reflection_lm)

    # Find optimal batch size based on token limits
    records_per_group = _find_optimal_batch_size(
        examples_data=examples_data,
        indices=indices,
        judge_instructions=judge_instructions,
        existing_guidelines=existing_guidelines,
        reflection_lm=reflection_lm,
        distillation_lm=distillation_lm,
        max_input_tokens=max_input_tokens,
    )

    if records_per_group == 0:
        _logger.warning(
            "Not a single trace can fit in the guideline distillation prompt. "
            "Please reduce the trace length."
        )
        return []

    # Distill guidelines from each batch of feedback records
    template = Template(DISTILLATION_PROMPT_TEMPLATE)
    all_guidelines = []
    existing_guideline_texts = set(existing_guidelines)

    try:
        from tqdm.auto import tqdm

        num_batches = (len(examples_data) + records_per_group - 1) // records_per_group
        batch_iter = tqdm(
            range(0, len(examples_data), records_per_group),
            total=num_batches,
            desc="Distilling guidelines",
        )
    except ImportError:
        batch_iter = range(0, len(examples_data), records_per_group)

    for i in batch_iter:
        batch_examples = examples_data[i : i + records_per_group]
        batch_indices = indices[i : i + records_per_group]

        prompt = template.render(
            judge_instructions=judge_instructions,
            feedback_records=batch_examples,
            ids=batch_indices,
            existing_guidelines=list(existing_guideline_texts),
            zip=zip,
            len=len,
        )

        try:
            response = distillation_lm(
                messages=[{"role": "user", "content": prompt}],
                response_format=Guidelines,
            )[0]

            batch_guidelines = _process_batch_response(
                response=response,
                index_to_trace_id=index_to_trace_id,
                existing_guideline_texts=existing_guideline_texts,
            )

            # Add new guidelines and update existing set to avoid duplicates across batches
            for guideline in batch_guidelines:
                all_guidelines.append(guideline)
                existing_guideline_texts.add(guideline.guideline_text)

        except Exception as e:
            _logger.error(f"Failed to generate/validate distilled guidelines for batch {i}: {e}")
            continue

    return all_guidelines


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
