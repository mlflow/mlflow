import copy
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

# Try to import jinja2 at module level
try:
    from jinja2 import Template

    _JINJA2_AVAILABLE = True
except ImportError:
    _JINJA2_AVAILABLE = False

from mlflow.entities.trace import Trace
from mlflow.environment_variables import MLFLOW_GENAI_OPTIMIZE_MAX_WORKERS
from mlflow.genai.judges.optimizers.dspy_utils import (
    construct_dspy_lm,
)
from mlflow.genai.judges.optimizers.memalign.prompts import (
    DISTILLATION_PROMPT_TEMPLATE,
    create_examples_field,
    create_guidelines_field,
)
from mlflow.genai.utils.trace_utils import (
    extract_request_from_trace,
    extract_response_from_trace,
)
from mlflow.metrics.genai.model_utils import convert_mlflow_uri_to_litellm

# Try to import litellm at module level
try:
    from litellm import get_model_info, token_counter

    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False

if TYPE_CHECKING:
    import dspy

_logger = logging.getLogger(__name__)

# Maximum input tokens for embedding models (most models have this limit or higher)
_MAX_EMBEDDING_MODEL_TOKENS = 8192
# Maximum input tokens for chat models (most models have this limit or higher)
_MAX_CHAT_MODEL_TOKENS = 128000
# Maximum records per batch for distillation
_MAX_RECORDS_PER_BATCH = 50
# Flexible tokens to reserve for response and variance in prompt length
_FLEX_TOKENS = 5000

# Priority list of fields to use for building corpus and retrieval queries
_QUERY_FIELD_PRIORITY = ["inputs", "outputs", "expectations", "conversation", "trace"]


def get_query_field(signature: "dspy.Signature") -> str | None:
    """Get the field name to use for building corpus and retrieval queries.

    Args:
        signature: DSPy signature defining input fields

    Returns:
        The first field from priority list that exists in signature's input_fields,
        or None if no matching field is found.
    """
    for field_name in _QUERY_FIELD_PRIORITY:
        if field_name in signature.input_fields:
            return field_name
    return None


@lru_cache(maxsize=1)
def _get_model_max_input_tokens(model: str, model_type: str) -> int:
    """Get the maximum input token limit for a model.

    Args:
        model: Model identifier (e.g., "openai:/text-embedding-3-small")
        model_type: Type of model ("embedding" or "chat")

    Returns:
        Maximum token limit for the model
    """

    if _LITELLM_AVAILABLE:
        litellm_model = convert_mlflow_uri_to_litellm(model)
        try:
            max_tokens = get_model_info(litellm_model)["max_input_tokens"]
            if max_tokens is not None:
                return max_tokens
        except Exception as e:
            _logger.debug(f"Error getting max tokens for model {model}: {e}", exc_info=True)

    if model_type == "embedding":
        return _MAX_EMBEDDING_MODEL_TOKENS
    elif model_type == "chat":
        return _MAX_CHAT_MODEL_TOKENS
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@lru_cache(maxsize=1000)
def truncate_to_token_limit(text: str, model: str, model_type: str) -> str:
    """Truncate text to fit within the model's token limit.

    Args:
        text: Text to truncate
        model: Model identifier (e.g., "openai:/text-embedding-3-small")
        model_type: Type of model ("embedding" or "chat")

    Returns:
        Truncated text that fits within token limit
    """
    max_tokens = _get_model_max_input_tokens(model, model_type=model_type)

    if not _LITELLM_AVAILABLE:
        # Naive truncation based on character count (1 token ~= 4 characters)
        # if litellm is not available
        _logger.warning(
            f"LiteLLM is required for accurate token counting, using naive truncation to "
            f"{max_tokens * 4} characters. Please install litellm using: `pip install litellm`"
        )
        return text[: max_tokens * 4]

    # Optimization to avoid token counting if number of characters is well below limit
    if len(text) <= max_tokens * 4 - _FLEX_TOKENS:
        return text

    litellm_model = convert_mlflow_uri_to_litellm(model)
    token_count = token_counter(model=litellm_model, text=text)
    if token_count <= max_tokens:
        return text

    original_token_count = token_count
    ratio = max_tokens / token_count
    truncated = text[: int(len(text) * ratio)]

    while token_counter(model=litellm_model, text=truncated) > max_tokens:
        truncated = truncated[: int(len(truncated) * 0.95)]

    _logger.debug(f"Truncated text from {original_token_count} to ~{max_tokens} tokens")
    return truncated


# These models describe the structured-output ``response_format`` for guideline distillation.
# ``extra="forbid"`` and the absence of field defaults are load-bearing: see
# ``_build_strict_response_format`` for the full set of schema constraints Databricks enforces.
class Guideline(BaseModel):
    model_config = {"extra": "forbid"}

    guideline_text: str
    # No default: a default drops the field from the schema's ``required`` list, which the
    # strict endpoint rejects. Stays nullable via ``| None`` so ``None`` remains valid.
    source_trace_ids: list[str | int] | None


class Guidelines(BaseModel):
    model_config = {"extra": "forbid"}

    guidelines: list[Guideline]


def _build_strict_response_format(model: type[BaseModel]) -> dict[str, Any]:
    """Build an OpenAI-strict ``json_schema`` response_format from a pydantic model.

    Databricks' structured-output endpoint enforces the strict-schema rules on every object
    and, unlike the OpenAI API, does not resolve ``$ref``/``$defs`` indirection ("reference
    can only point to definitions defined at the top level of the schema"). Pydantic's
    ``model_json_schema()`` emits nested models as ``$ref`` into ``$defs`` and omits
    ``additionalProperties``/full ``required``. This inlines every ``$ref`` and enforces
    ``additionalProperties: false`` plus a complete ``required`` list on each object, so the
    resulting schema is accepted regardless of whether the litellm version normalizes it.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                resolved = resolve(copy.deepcopy(defs[name]))
                # Merge any sibling keys (e.g. description) over the resolved definition.
                resolved.update({k: resolve(v) for k, v in node.items() if k != "$ref"})
                return resolved
            node = {k: resolve(v) for k, v in node.items()}
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                if "properties" in node:
                    node["required"] = list(node["properties"].keys())
            return node
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return {
        "type": "json_schema",
        "json_schema": {"name": model.__name__, "schema": resolve(schema), "strict": True},
    }


def get_default_embedding_model() -> str:
    return "openai:/text-embedding-3-small"


def _count_tokens(text: str, litellm_model: str | None) -> int:
    """Count tokens in text using litellm or naive whitespace estimation."""
    if litellm_model is not None and _LITELLM_AVAILABLE:
        return token_counter(model=litellm_model, text=text)
    # Fallback: heuristic estimation based on character count
    # Approximate 4 characters per token (see https://platform.openai.com/tokenizer)
    return len(text) // 4


def _make_json_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_json_serializable(item) for item in value]
    return value_to_embedding_text(value)


def _create_batches(
    examples_data: list[dict[str, Any]],
    indices: list[int],
    judge_instructions: str,
    existing_guidelines: list[str],
    reflection_lm: str,
) -> list[list[int]]:
    """Create batches using greedy bin-packing based on token counts.

    Computes token count for each example and greedily packs them into batches
    that fit within the model's token limit.

    Args:
        examples_data: List of feedback example data dicts
        indices: List of indices corresponding to examples_data
        judge_instructions: Original judge instructions
        existing_guidelines: Previously distilled guidelines
        reflection_lm: Model to use for distillation

    Returns:
        List of batches, where each batch is a list of indices into examples_data
    """
    max_input_tokens = _get_model_max_input_tokens(reflection_lm, model_type="chat")
    prompt_tokens_limit = max_input_tokens - _FLEX_TOKENS

    litellm_model = convert_mlflow_uri_to_litellm(reflection_lm) if _LITELLM_AVAILABLE else None

    # Compute base overhead (template + instructions + guidelines, without examples)
    template = Template(DISTILLATION_PROMPT_TEMPLATE)
    base_prompt = template.render(
        judge_instructions=judge_instructions,
        feedback_records=[],
        ids=[],
        existing_guidelines=existing_guidelines,
        zip=zip,
        len=len,
    )
    base_tokens = _count_tokens(base_prompt, litellm_model)

    # Compute token count for each example
    example_tokens = []
    for example in examples_data:
        example_str = json.dumps(_make_json_serializable(example))
        tokens = _count_tokens(example_str, litellm_model)
        example_tokens.append(tokens)

    # Greedy bin-packing
    batches = []
    current_batch = []
    current_tokens = base_tokens

    for idx, tokens in zip(indices, example_tokens):
        # Check if adding this example would exceed limits
        if current_batch and (
            current_tokens + tokens > prompt_tokens_limit
            or len(current_batch) >= _MAX_RECORDS_PER_BATCH
        ):
            # Start a new batch
            batches.append(current_batch)
            current_batch = []
            current_tokens = base_tokens

        current_batch.append(idx)
        current_tokens += tokens

    # Add the last batch
    if current_batch:
        batches.append(current_batch)

    return batches


def _extract_json_object(response: str) -> str:
    """Extract the JSON object from an LLM response.

    Structured-output (``response_format``) responses are raw JSON, but the unstructured
    fallback relies on the prompt to elicit JSON and models often wrap it in a
    ```` ```json ... ``` ```` fence and/or surround it with prose. Strips a fence when
    present, then narrows to the outermost ``{...}`` so leading/trailing text does not
    break parsing. Returns the original (stripped) string when no object is found, so the
    caller's ``json.loads`` raises a meaningful error.
    """
    stripped = response.strip()
    if match := re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", stripped, re.DOTALL):
        stripped = match.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _parse_batch_response(
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
    response_data = json.loads(_extract_json_object(response))
    guidelines = []
    trace_ids_set = set(index_to_trace_id.values())

    def resolve_trace_id(idx: Any) -> str | None:
        """Resolve an LLM-returned index to a trace ID."""
        if isinstance(idx, int):
            return index_to_trace_id.get(idx)
        if isinstance(idx, str):
            if idx in trace_ids_set:
                return idx
            try:
                return index_to_trace_id.get(int(idx))
            except ValueError:
                return None
        return None

    for guideline_data in response_data.get("guidelines", []):
        # Skip empty or duplicate guidelines
        guideline_text = guideline_data.get("guideline_text")
        if not guideline_text or guideline_text in existing_guideline_texts:
            continue

        # Skip guidelines without valid source trace IDs
        source_trace_ids_raw = guideline_data.get("source_trace_ids")
        if source_trace_ids_raw is None:
            continue

        # Map indices back to trace IDs, filtering out invalid values
        trace_ids = [
            resolved
            for idx in source_trace_ids_raw
            if (resolved := resolve_trace_id(idx)) is not None
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


def value_to_embedding_text(value: Any) -> str:
    """
    Convert an arbitrary value to text suitable for embedding.

    For Trace objects, extracts the request and response text. We do not use other attributes
    of the trace because the size is generally unbounded.
    For other types, returns the string representation.
    """
    if isinstance(value, Trace):
        parts = []
        if request := extract_request_from_trace(value):
            parts.append(request)
        if response := extract_response_from_trace(value):
            parts.append(response)
        return " ".join(parts) if parts else ""
    return str(value)


def distill_guidelines(
    examples: list["dspy.Example"],
    judge_instructions: str,
    reflection_lm: str,
    existing_guidelines: list[str],
) -> list[Guideline]:
    """Distill general guidelines from feedback examples.

    The number of parallel threads for LLM calls can be configured via the
    ``MLFLOW_GENAI_OPTIMIZE_MAX_WORKERS`` environment variable (default: 8).

    Args:
        examples: List of DSPy examples containing feedback (with _trace_id attribute)
        judge_instructions: Original judge instructions
        reflection_lm: Model to use for distillation
        existing_guidelines: Previously distilled guidelines

    Returns:
        List of newly distilled Guideline objects (not including existing ones)
    """
    if not _JINJA2_AVAILABLE:
        raise ImportError(
            "jinja2 is required for guideline distillation. "
            "Please install it using: `pip install jinja2`"
        )

    if not examples:
        return []

    examples_data = [_make_json_serializable(dict(example)) for example in examples]
    # Create index to trace_id mapping
    indices = list(range(len(examples_data)))
    index_to_trace_id = {
        i: example._trace_id if hasattr(example, "_trace_id") else f"example_{i}"
        for i, example in enumerate(examples)
    }

    distillation_lm = construct_dspy_lm(reflection_lm)

    # Create batches using greedy bin-packing
    batches = _create_batches(
        examples_data=examples_data,
        indices=indices,
        judge_instructions=judge_instructions,
        existing_guidelines=existing_guidelines,
        reflection_lm=reflection_lm,
    )

    if not batches:
        _logger.error(
            "Inputs to the judge are too large, please reduce the size of inputs for alignment. "
        )
        return []

    # Distill guidelines from each batch of feedback records in parallel
    template = Template(DISTILLATION_PROMPT_TEMPLATE)
    existing_guideline_texts = set(existing_guidelines)

    def process_batch(batch_indices: list[int]) -> list[Guideline]:
        batch_examples = [examples_data[i] for i in batch_indices]

        prompt = template.render(
            judge_instructions=judge_instructions,
            feedback_records=batch_examples,
            ids=batch_indices,
            existing_guidelines=list(existing_guideline_texts),
            zip=zip,
            len=len,
        )

        messages = [{"role": "user", "content": prompt}]
        try:
            try:
                response = distillation_lm(
                    messages=messages, response_format=_build_strict_response_format(Guidelines)
                )[0]
            except Exception as e:
                # Some models (e.g. some Databricks-served endpoints) reject or error on
                # structured-output requests. The prompt already specifies the exact JSON
                # format, so fall back to an unstructured request rather than dropping the
                # batch. Mirrors the judge adapter's response_format fallback (see
                # mlflow/genai/judges/adapters/litellm_adapter.py).
                _logger.debug(
                    f"Structured-output distillation failed for batch {batch_indices}; "
                    f"retrying without response_format. Error: {e}",
                    exc_info=True,
                )
                response = distillation_lm(messages=messages)[0]

            return _parse_batch_response(
                response=response,
                index_to_trace_id=index_to_trace_id,
                existing_guideline_texts=existing_guideline_texts,
            )
        except Exception as e:
            _logger.error(
                f"Failed to generate/validate distilled guidelines for batch "
                f"with indices {batch_indices}: {e}"
            )
            return []

    # Process batches in parallel using ThreadPoolExecutor
    all_guidelines = []
    try:
        from tqdm.auto import tqdm

        use_tqdm = True
    except ImportError:
        use_tqdm = False

    with ThreadPoolExecutor(
        max_workers=MLFLOW_GENAI_OPTIMIZE_MAX_WORKERS.get(),
        thread_name_prefix="MLflowMemAlignDistillation",
    ) as executor:
        futures = {executor.submit(process_batch, batch): batch for batch in batches}

        if use_tqdm:
            futures_iter = tqdm(
                as_completed(futures), total=len(futures), desc="Distilling guidelines"
            )
        else:
            futures_iter = as_completed(futures)

        for future in futures_iter:
            batch_guidelines = future.result()
            all_guidelines.extend(batch_guidelines)

    # Deduplicate guidelines (since batches ran in parallel with the same existing_guideline_texts)
    seen_texts = set(existing_guidelines)
    new_guidelines = []
    for guideline in all_guidelines:
        if guideline.guideline_text not in seen_texts:
            seen_texts.add(guideline.guideline_text)
            new_guidelines.append(guideline)

    return new_guidelines


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

    query_field = get_query_field(signature)
    value = query_kwargs.get(query_field) if query_field else None
    query = value_to_embedding_text(value) if value else ""
    if not query:
        return []

    search_results = retriever(query)
    indices = [int(i) for i in search_results.indices]

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
