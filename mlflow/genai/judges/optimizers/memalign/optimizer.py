import logging
from typing import TYPE_CHECKING

from mlflow.entities.assessment import Assessment, AssessmentSource, Feedback
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.base import AlignmentOptimizer, Judge, JudgeField
from mlflow.genai.judges.optimizers.dspy_utils import (
    construct_dspy_lm,
    create_dspy_signature,
    trace_to_dspy_example,
)
from mlflow.genai.judges.optimizers.memalign.prompts import (
    create_examples_field,
    create_guidelines_field,
)
from mlflow.genai.judges.optimizers.memalign.utils import (
    distill_guidelines,
    get_default_embedding_model,
    retrieve_relevant_examples,
)
from mlflow.genai.judges.utils import get_default_model
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring

if TYPE_CHECKING:
    import dspy

_logger = logging.getLogger(__name__)

_MODEL_API_DOC = {
    "reflection_lm": """Model to use for distilling guidelines from feedback.
Supported formats:

* `"databricks"` for Databricks-native integration
* `"databricks:/<endpoint-name>"` or `"endpoints:/<endpoint-name>"` for
  Databricks model serving endpoints
* `<provider>:/<model-name>` for other providers (e.g.,
  `"openai:/gpt-4o-mini"`, `"anthropic:/claude-3.5-sonnet-20240620"`)

MLflow natively supports `["openai", "anthropic", "bedrock", "mistral"]`,
and more providers are supported through
`LiteLLM <https://docs.litellm.ai/docs/providers>`_.

Default model depends on the tracking URI setup:

* Databricks: `databricks`
* Otherwise: `openai:/gpt-4o-mini`.
""",
    "embedding_model": """Model to use for generating embeddings for
example retrieval. Must be a form of `<provider>/<model-name>`, such as
`"openai/text-embedding-3-small"`. Supported providers include OpenAI and
others via LiteLLM. Default: `"openai/text-embedding-3-small"`.
""",
}
# Maximum tokens for embedding model input (most embedding models have this limit)
_MAX_EMBEDDING_TOKENS = 8192


def _get_max_tokens_for_model(embedding_model: str) -> int:
    """Get maximum token limit for embedding model."""
    try:
        from litellm import get_max_tokens

        max_tokens = get_max_tokens(embedding_model)
        if max_tokens is not None:
            return max_tokens
    except Exception as e:
        _logger.debug(f"Error getting max tokens for model {embedding_model}: {e}", exc_info=True)
    return _MAX_EMBEDDING_TOKENS


def _truncate_to_token_limit(text: str, embedding_model: str) -> str:
    from litellm import token_counter

    max_tokens = _get_max_tokens_for_model(embedding_model)
    token_count = token_counter(model=embedding_model, text=text)
    if token_count <= max_tokens:
        return text

    original_token_count = token_count
    ratio = max_tokens / token_count
    truncated = text[: int(len(text) * ratio)]

    while token_counter(model=embedding_model, text=truncated) > max_tokens:
        truncated = truncated[: int(len(truncated) * 0.95)]

    _logger.debug(f"Truncated example from {original_token_count} to ~{max_tokens} tokens")
    return truncated


@experimental(version="3.9.0")
@format_docstring(_MODEL_API_DOC)
class MemoryAugmentedJudge(Judge):
    """
    A judge augmented with dual memory systems.

    This judge enhances evaluation with:
    - Semantic Memory: Distilled guidelines from past feedback
    - Episodic Memory: Retrieved similar examples from past feedback

    The judge maintains state across evaluations and provides the exact same
    interface as the original judge. Memories are managed internally.

    Args:
        base_judge: Base judge to augment with memory systems
        reflection_lm: {{ reflection_lm }}
        retrieval_k: Number of similar examples to retrieve from episodic memory
        embedding_model: {{ embedding_model }}
        embedding_dim: Dimension of embeddings
        examples: Initial examples to add to memory
        inherit_guidelines: Whether to inherit guidelines from base_judge if it's a
            MemoryAugmentedJudge. If False, guidelines are redistilled from scratch.
    """

    def __init__(
        self,
        base_judge: Judge,
        reflection_lm: str | None = None,
        retrieval_k: int = 5,
        embedding_model: str | None = None,
        embedding_dim: int = 512,
        examples: list["dspy.Example"] | None = None,
        inherit_guidelines: bool = True,
    ):
        import dspy

        effective_base_judge = (
            base_judge._base_judge if isinstance(base_judge, MemoryAugmentedJudge) else base_judge
        )
        initial_guidelines = (
            list(base_judge._semantic_memory)
            if isinstance(base_judge, MemoryAugmentedJudge) and inherit_guidelines
            else []
        )

        super().__init__(
            name=effective_base_judge.name,
            description=effective_base_judge.description,
            aggregations=effective_base_judge.aggregations,
        )

        self._base_judge = effective_base_judge
        self._base_signature = create_dspy_signature(effective_base_judge)
        self._retrieval_k = retrieval_k
        self._examples: list["dspy.Example"] = []
        self._semantic_memory: list[str] = initial_guidelines

        self._reflection_lm = reflection_lm if reflection_lm is not None else get_default_model()

        self._embedding_model = (
            embedding_model if embedding_model is not None else get_default_embedding_model()
        )
        self._embedding_dim = embedding_dim
        self._embedder = dspy.Embedder(self._embedding_model, dimensions=self._embedding_dim)
        self._search = None

        extended_signature = self._create_extended_signature()
        self._predict_module = dspy.Predict(extended_signature)
        self._predict_module.set_lm(construct_dspy_lm(effective_base_judge.model))

        if examples:
            self._add_examples(examples)

    def __call__(self, **kwargs) -> Assessment:
        guidelines = self._semantic_memory
        relevant_examples, retrieved_indices = retrieve_relevant_examples(
            search=self._search,
            examples=self._examples,
            query_kwargs=kwargs,
            signature=self._base_signature,
        )

        prediction = self._predict_module(
            guidelines=guidelines,
            example_judgements=relevant_examples,
            **kwargs,
        )

        return Feedback(
            name=self._base_judge.name,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=self._base_judge.model,
            ),
            value=prediction.result,
            rationale=prediction.rationale,
            metadata={"retrieved_example_indices": retrieved_indices} if retrieved_indices else {},
        )

    @property
    def name(self) -> str:
        return self._base_judge.name

    @property
    def instructions(self) -> str:
        instructions = self._base_judge.instructions
        if self._semantic_memory:
            instructions += f"\n\nDistilled Guidelines ({len(self._semantic_memory)}):\n"
            for guideline in self._semantic_memory:
                instructions += f"  - {guideline}\n"
        return instructions

    @property
    def model(self) -> str:
        return self._base_judge.model

    def get_input_fields(self) -> list[JudgeField]:
        return self._base_judge.get_input_fields()

    @experimental(version="3.9.0")
    def unalign(self, traces: list[Trace]) -> "MemoryAugmentedJudge":
        """
        Remove specific traces from memory and return new judge.

        This method allows you to selectively remove feedback examples from the judge's
        memory systems. This is useful when you want to:

        - Remove incorrect or low-quality feedback that negatively impacts performance
        - Update the judge in case your evaluation criteria change
        - Remove feedback from specific users or time periods

        The returned judge will have guidelines automatically redistributed from the
        remaining examples, ensuring the semantic memory reflects only the retained feedback.

        Args:
            traces: Traces containing feedback to remove from memory. Only traces with
                feedback matching this judge's name will be removed.

        Returns:
            A new MemoryAugmentedJudge with the specified traces removed from memory.

        Example:
            .. code-block:: python
                aligned_judge = judge.align(traces=all_traces, optimizer=optimizer)
                aligned_judge_v2 = aligned_judge.unalign(traces=bad_traces)
                # The judge now only reflects feedback from `set(all_traces) - set(bad_traces)`
        """
        trace_ids_to_remove = {trace.info.trace_id for trace in traces}

        filtered_examples = [
            example
            for example in self._examples
            if not (hasattr(example, "_trace_id") and example._trace_id in trace_ids_to_remove)
        ]

        if len(filtered_examples) == len(self._examples):
            _logger.warning("No feedback records found for the provided traces")
            return self

        return MemoryAugmentedJudge(
            base_judge=self._base_judge,
            reflection_lm=self._reflection_lm,
            retrieval_k=self._retrieval_k,
            embedding_model=self._embedding_model,
            embedding_dim=self._embedding_dim,
            examples=filtered_examples,
            inherit_guidelines=False,
        )

    def _create_extended_signature(self) -> "dspy.Signature":
        extended_sig = self._base_signature.prepend("guidelines", create_guidelines_field())
        return extended_sig.prepend("example_judgements", create_examples_field())

    def _add_examples(self, examples: list["dspy.Example"]) -> None:
        import dspy.retrievers

        self._examples.extend(examples)

        # Distill new guidelines from all examples
        new_guidelines = distill_guidelines(
            examples=self._examples,
            signature=self._base_signature,
            judge_instructions=self._base_judge.instructions,
            reflection_lm=self._reflection_lm,
            existing_guidelines=self._semantic_memory,
        )
        self._semantic_memory.extend(new_guidelines)
        _logger.debug(
            f"Distilled {len(new_guidelines)} new guidelines from {len(self._examples)} examples"
        )

        # Build episodic memory corpus from input fields
        corpus = []
        for example in self._examples:
            query_parts = []
            for field_name in self._base_signature.input_fields:
                if hasattr(example, field_name):
                    value = getattr(example, field_name)
                    if value is not None:
                        query_parts.append(str(value))
            query = " ".join(query_parts)
            query = _truncate_to_token_limit(query, self._embedding_model)
            corpus.append(query)

        self._search = dspy.retrievers.Embeddings(
            embedder=self._embedder, corpus=corpus, k=self._retrieval_k
        )
        _logger.debug(f"Episodic memory corpus contains {len(corpus)} examples")


@experimental(version="3.9.0")
@format_docstring(_MODEL_API_DOC)
class MemAlignOptimizer(AlignmentOptimizer):
    """
    MemAlign alignment optimizer using dual memory systems.

    This optimizer creates a memory-augmented judge that learns from feedback
    through two complementary mechanisms:

    **Semantic Memory** - Distills general guidelines from feedback:
        - LLM extracts patterns from feedback records
        - Guidelines describe user preferences and expectations
        - Applied as context to all future evaluations

    **Episodic Memory** - Retrieves similar past examples:
        - Stores feedback records with embeddings
        - Finds most similar examples during evaluation
        - Provides concrete examples as evaluation context

    The returned judge is a MemoryAugmentedJudge that maintains memory state.

    Args:
        reflection_lm: {{ reflection_lm }}
        retrieval_k: Number of similar examples to retrieve from episodic memory
        embedding_model: {{ embedding_model }}
        embedding_dim: Dimension of embeddings

    Example:
        .. code-block:: python

            judge = make_judge(name="quality", instructions="...", model="openai:/gpt-4")
            optimizer = MemAlignOptimizer(
                reflection_lm="openai:/gpt-4o-mini",
                retrieval_k=3,
                embedding_model="openai/text-embedding-3-small",
            )
            optimized_judge = judge.align(traces=traces, optimizer=optimizer)
            result = optimized_judge(inputs={...}, outputs={...})
    """

    def __init__(
        self,
        reflection_lm: str | None = None,
        retrieval_k: int = 5,
        embedding_model: str | None = None,
        embedding_dim: int = 512,
    ):
        self._reflection_lm = reflection_lm if reflection_lm is not None else get_default_model()
        self._retrieval_k = retrieval_k
        self._embedding_model = (
            embedding_model if embedding_model is not None else get_default_embedding_model()
        )
        self._embedding_dim = embedding_dim

    def align(self, judge: Judge, traces: list[Trace]) -> Judge:
        """
        Align judge with human feedback from traces.

        Args:
            judge: Judge to align
            traces: Traces containing human feedback

        Returns:
            Memory-augmented judge aligned with feedback
        """
        try:
            if not traces:
                raise MlflowException(
                    "No traces provided for alignment", error_code=INVALID_PARAMETER_VALUE
                )

            _logger.debug(f"Starting MemAlign alignment with {len(traces)} traces")

            existing_examples = (
                list(judge._examples) if isinstance(judge, MemoryAugmentedJudge) else []
            )

            new_examples = []
            for trace in traces:
                example = trace_to_dspy_example(trace, judge)
                if example is not None:
                    example._trace_id = trace.info.trace_id
                    new_examples.append(example)

            if not new_examples:
                raise MlflowException(
                    f"No valid feedback records found in traces. "
                    f"Ensure traces contain human assessments with name '{judge.name}'",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            _logger.debug(
                f"Created {len(new_examples)} new feedback records from {len(traces)} traces"
            )

            all_examples = existing_examples + new_examples

            memory_judge = MemoryAugmentedJudge(
                base_judge=judge,
                reflection_lm=self._reflection_lm,
                retrieval_k=self._retrieval_k,
                embedding_model=self._embedding_model,
                embedding_dim=self._embedding_dim,
                examples=all_examples,
            )

            _logger.debug(
                f"MemAlign alignment completed successfully. Aligned {len(new_examples)} examples."
            )
            return memory_judge

        except Exception as e:
            _logger.error(f"MemAlign alignment failed: {e}", exc_info=True)
            raise MlflowException(
                f"Alignment optimization failed: {e!s}", error_code=INTERNAL_ERROR
            ) from e
