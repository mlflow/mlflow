import copy
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import mlflow
from mlflow.entities.assessment import Assessment, AssessmentSource, Feedback
from mlflow.entities.assessment_source import AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.base import AlignmentOptimizer, Judge, JudgeField
from mlflow.genai.judges.optimizers.dspy_utils import (
    _check_dspy_installed,
    construct_dspy_lm,
    convert_mlflow_uri_to_litellm,
    create_dspy_signature,
    trace_to_dspy_example,
)
from mlflow.genai.judges.optimizers.memalign.utils import (
    Guideline,
    create_extended_signature,
    distill_guidelines,
    get_default_embedding_model,
    get_query_field,
    retrieve_relevant_examples,
    truncate_to_token_limit,
    value_to_embedding_text,
)
from mlflow.genai.judges.utils import get_default_model
from mlflow.genai.scorers.base import (
    _SERIALIZATION_VERSION,
    ScorerKind,
    SerializedScorer,
)
from mlflow.genai.utils.trace_utils import (
    resolve_expectations_from_trace,
    resolve_inputs_from_trace,
    resolve_outputs_from_trace,
)
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring

if TYPE_CHECKING:
    import dspy

_logger = logging.getLogger(__name__)

_CONFIG_FIELDS = ("reflection_lm", "retrieval_k", "embedding_model", "embedding_dim")

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
example retrieval. Must be a form of `<provider>:/<model-name>`, such as
`"openai:/text-embedding-3-small"`. Supported providers include OpenAI and
others via LiteLLM. Default: `"openai:/text-embedding-3-small"`.
""",
}


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
        retrieval_k: Number of similar examples to retrieve from episodic memory (default: 5)
        embedding_model: {{ embedding_model }}
        embedding_dim: Dimension of embeddings (default: 512)
    """

    def __init__(
        self,
        base_judge: Judge,
        reflection_lm: str | None = None,
        retrieval_k: int = 5,
        embedding_model: str | None = None,
        embedding_dim: int = 512,
        *,
        _defer_init: bool = False,
    ):
        effective_base_judge = (
            base_judge._base_judge if isinstance(base_judge, MemoryAugmentedJudge) else base_judge
        )

        super().__init__(
            name=effective_base_judge.name,
            description=effective_base_judge.description,
            aggregations=effective_base_judge.aggregations,
        )

        self._base_judge = effective_base_judge
        self._retrieval_k = retrieval_k
        self._reflection_lm = reflection_lm if reflection_lm is not None else get_default_model()
        self._embedding_model = (
            embedding_model if embedding_model is not None else get_default_embedding_model()
        )
        self._embedding_dim = embedding_dim

        # Always store trace IDs for serialization
        self._episodic_trace_ids: list[str] = []

        if _defer_init:
            # Defer creating heavyweight DSPy objects until first use (_embedder=None signals this)
            self._base_signature = None
            self._embedder = None
            self._retriever = None
            self._predict_module = None
            self._episodic_memory: list["dspy.Example"] = []
            self._semantic_memory: list[Guideline] = []
        else:
            self._initialize_dspy_components(base_judge)

    def _initialize_dspy_components(self, base_judge: Judge | None = None) -> None:
        """Initialize heavyweight DSPy components (embedder, predict module, memory index)."""
        import dspy

        effective_base_judge = base_judge or self._base_judge

        self._base_signature = create_dspy_signature(effective_base_judge)
        litellm_embedding_model = convert_mlflow_uri_to_litellm(self._embedding_model)
        self._embedder = dspy.Embedder(
            litellm_embedding_model, dimensions=self._embedding_dim, drop_params=True
        )
        self._retriever = None

        # Inherit memory from base_judge if it's a MemoryAugmentedJudge
        if isinstance(base_judge, MemoryAugmentedJudge):
            self._semantic_memory = copy.deepcopy(base_judge._semantic_memory)
            self._episodic_memory = copy.deepcopy(base_judge._episodic_memory)
            self._build_episodic_memory()
        else:
            self._episodic_memory: list["dspy.Example"] = []
            self._semantic_memory: list[Guideline] = []

        extended_signature = create_extended_signature(self._base_signature)
        self._predict_module = dspy.Predict(extended_signature)
        self._predict_module.set_lm(construct_dspy_lm(effective_base_judge.model))

    def __call__(
        self,
        *,
        inputs: Any = None,
        outputs: Any = None,
        expectations: dict[str, Any] | None = None,
        trace: Trace | None = None,
    ) -> Assessment:
        self._lazy_init()

        if trace is not None:
            inputs = resolve_inputs_from_trace(inputs, trace)
            outputs = resolve_outputs_from_trace(outputs, trace)
            expectations = resolve_expectations_from_trace(expectations, trace)

        guidelines = [g.guideline_text for g in self._semantic_memory]
        query_kwargs = {
            "inputs": inputs,
            "outputs": outputs,
            "expectations": expectations,
            "trace": trace,
        }
        retrieved_results = retrieve_relevant_examples(
            retriever=self._retriever,
            examples=self._episodic_memory,
            query_kwargs=query_kwargs,
            signature=self._base_signature,
        )

        relevant_examples = [example for example, _ in retrieved_results]
        retrieved_trace_ids = [trace_id for _, trace_id in retrieved_results]

        prediction = self._predict_module(
            guidelines=guidelines,
            example_judgements=relevant_examples,
            inputs=inputs,
            outputs=outputs,
            expectations=expectations,
            trace=value_to_embedding_text(trace) if trace is not None else None,
        )

        return Feedback(
            name=self._base_judge.name,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=self._base_judge.model,
            ),
            value=prediction.result,
            rationale=prediction.rationale,
            metadata={"retrieved_example_trace_ids": retrieved_trace_ids}
            if retrieved_trace_ids
            else {},
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
                instructions += f"  - {guideline.guideline_text}\n"
        return instructions

    @property
    def model(self) -> str:
        return self._base_judge.model

    @property
    def feedback_value_type(self) -> Any:
        return self._base_judge.feedback_value_type

    def get_input_fields(self) -> list[JudgeField]:
        return self._base_judge.get_input_fields()

    @property
    def kind(self) -> ScorerKind:
        return ScorerKind.MEMORY_AUGMENTED

    def model_dump(self, **kwargs) -> dict[str, Any]:
        base_judge_data = self._base_judge.model_dump(**kwargs)

        memory_augmented_data = {
            "base_judge": base_judge_data,
            "episodic_trace_ids": self._episodic_trace_ids,
            "semantic_memory": [g.model_dump() for g in self._semantic_memory],
            **{field: getattr(self, f"_{field}") for field in _CONFIG_FIELDS},
        }

        serialized = SerializedScorer(
            name=self.name,
            description=self.description,
            aggregations=self.aggregations,
            mlflow_version=mlflow.__version__,
            serialization_version=_SERIALIZATION_VERSION,
            memory_augmented_judge_data=memory_augmented_data,
        )
        return asdict(serialized)

    @classmethod
    def _from_serialized(
        cls,
        serialized: SerializedScorer,
    ) -> "MemoryAugmentedJudge":
        # Import here to avoid circular dependency: base.py imports MemoryAugmentedJudge
        from mlflow.genai.scorers.base import Scorer

        data = serialized.memory_augmented_judge_data

        base_judge_serialized = SerializedScorer(**data["base_judge"])
        base_judge = Scorer.model_validate(base_judge_serialized)

        # Use constructor with _defer_init=True to skip heavyweight DSPy initialization
        instance = cls(
            base_judge=base_judge,
            reflection_lm=data.get("reflection_lm"),
            retrieval_k=data.get("retrieval_k", 5),
            embedding_model=data.get("embedding_model"),
            embedding_dim=data.get("embedding_dim", 512),
            _defer_init=True,
        )

        # Restore semantic memory and episodic trace IDs for lazy loading
        instance._semantic_memory = [Guideline(**g) for g in data["semantic_memory"]]
        instance._episodic_trace_ids = data.get("episodic_trace_ids") or []

        return instance

    def _create_copy(self) -> "MemoryAugmentedJudge":
        """
        Override base _create_copy for Scorer.register().

        The base implementation uses model_copy(deep=True), which fails because
        DSPy objects (_embedder, _retriever, _predict_module) contain thread locks
        that can't be pickled. We create a new instance with _defer_init=True and
        store trace IDs for lazy reconstruction.
        """
        judge_copy = MemoryAugmentedJudge(
            base_judge=self._base_judge,
            reflection_lm=self._reflection_lm,
            retrieval_k=self._retrieval_k,
            embedding_model=self._embedding_model,
            embedding_dim=self._embedding_dim,
            _defer_init=True,
        )
        judge_copy._semantic_memory = copy.deepcopy(self._semantic_memory)
        judge_copy._episodic_trace_ids = self._episodic_trace_ids.copy()

        return judge_copy

    def _lazy_init(self) -> None:
        """
        Lazily initialize DSPy components and episodic memory from stored trace IDs.

        This method is called on first use (e.g., __call__) when the judge was created
        with _defer_init=True. It:
        1. Creates DSPy components (embedder, predict module)
        2. Fetches traces by ID and reconstructs episodic memory
        3. Builds the episodic memory search index

        No-op if already initialized (checked via _embedder not being None).
        """
        if self._embedder is not None:
            return

        import dspy

        self._base_signature = create_dspy_signature(self._base_judge)

        litellm_embedding_model = convert_mlflow_uri_to_litellm(self._embedding_model)
        self._embedder = dspy.Embedder(
            litellm_embedding_model, dimensions=self._embedding_dim, drop_params=True
        )

        extended_signature = create_extended_signature(self._base_signature)
        self._predict_module = dspy.Predict(extended_signature)
        self._predict_module.set_lm(construct_dspy_lm(self._base_judge.model))

        # Fetch traces by ID using mlflow.get_trace which handles location context
        traces = []
        missing_ids = []
        for trace_id in self._episodic_trace_ids:
            trace = mlflow.get_trace(trace_id, silent=True)
            if trace is not None:
                traces.append(trace)
            else:
                missing_ids.append(trace_id)

        if missing_ids:
            _logger.warning(
                f"Could not find {len(missing_ids)} traces for episodic memory reconstruction. "
                f"Missing trace IDs: {missing_ids[:5]}"
                f"{'...' if len(missing_ids) > 5 else ''}. "
                f"Judge will operate with partial memory "
                f"({len(traces)}/{len(self._episodic_trace_ids)} traces)."
            )

        for trace in traces:
            if example := trace_to_dspy_example(trace, self._base_judge):
                example._trace_id = trace.info.trace_id
                self._episodic_memory.append(example)

        if self._episodic_memory:
            self._build_episodic_memory()

    @experimental(version="3.9.0")
    def unalign(self, traces: list[Trace]) -> "MemoryAugmentedJudge":
        """
        Remove specific traces from memory and return an updated judge.

        This method allows you to selectively remove feedback examples from the judge's
        memory systems. This is useful when you want to:

        - Remove incorrect or low-quality feedback that negatively impacts performance
        - Update the judge in case your evaluation criteria change
        - Remove feedback from specific users or time periods

        The returned judge will have guidelines selectively deleted based on source_trace_ids:
        - Guidelines where all source traces were removed are deleted
        - Guidelines with at least one remaining source trace are retained

        Args:
            traces: Traces containing feedback to remove from memory. Only traces with
                feedback matching this judge's name will be removed.

        Returns:
            Updated MemoryAugmentedJudge with specified traces removed from memory.

        Example:
            .. code-block:: python

                import mlflow
                from mlflow.genai.judges import make_judge
                from mlflow.genai.judges.optimizers import MemAlignOptimizer

                # Assuming `all_traces` contains human feedback for the judge
                aligned_judge = judge.align(traces=all_traces, optimizer=MemAlignOptimizer())
                aligned_judge_v2 = aligned_judge.unalign(traces=bad_traces)
                # aligned_judge_v2 now only retains feedback from
                # `set(all_traces) - set(bad_traces)`
        """
        trace_ids_to_remove = {trace.info.trace_id for trace in traces}

        # Filter examples to retain based on trace ids
        examples_to_retain = [
            example
            for example in self._episodic_memory
            if not (hasattr(example, "_trace_id") and example._trace_id in trace_ids_to_remove)
        ]
        if len(examples_to_retain) == len(self._episodic_memory):
            _logger.warning("No feedback records found for the provided traces")
            return self

        # Filter guidelines to retain based on source_trace_ids
        # - Always retain user-provided guidelines (those without source_trace_ids)
        # - Delete guideline only if ALL of its source traces were removed
        guidelines_to_retain = [
            guideline
            for guideline in self._semantic_memory
            if guideline.source_trace_ids is None
            or any(tid not in trace_ids_to_remove for tid in guideline.source_trace_ids)
        ]

        # Reinitialize new judge
        new_judge = MemoryAugmentedJudge(
            base_judge=self._base_judge,
            reflection_lm=self._reflection_lm,
            retrieval_k=self._retrieval_k,
            embedding_model=self._embedding_model,
            embedding_dim=self._embedding_dim,
        )

        new_judge._semantic_memory = guidelines_to_retain
        new_judge._episodic_memory = examples_to_retain
        new_judge._build_episodic_memory()

        _logger.debug(
            f"Removed {len(traces)} traces from memory. "
            f"Episodic memory size: {len(new_judge._episodic_memory)} examples, "
            f"Semantic memory size: {len(new_judge._semantic_memory)} guidelines."
        )
        return new_judge

    def _distill_new_guidelines(self, new_examples: list["dspy.Example"]) -> None:
        """
        Distill new guidelines from newly added examples and add to semantic memory.

        Args:
            new_examples: The examples that were just added (not all examples)
        """
        existing_guideline_texts = [g.guideline_text for g in self._semantic_memory]
        new_guidelines = distill_guidelines(
            examples=new_examples,
            judge_instructions=self._base_judge.instructions,
            reflection_lm=self._reflection_lm,
            existing_guidelines=existing_guideline_texts,
        )
        self._semantic_memory.extend(new_guidelines)
        _logger.debug(
            f"Distilled {len(new_guidelines)} new guidelines from {len(new_examples)} new "
            f"examples. Semantic memory now has {len(self._semantic_memory)} guidelines."
        )

    def _build_episodic_memory(self) -> None:
        """Build episodic memory search index from examples."""
        import dspy.retrievers

        query_field = get_query_field(self._base_signature)
        if query_field is None:
            raise MlflowException(
                "Unable to build episodic memory: no suitable input field found in judge "
                "instructions. Please ensure the judge instructions reference at least one of "
                "the following fields: inputs, outputs, expectations, conversation, trace.",
                error_code=INTERNAL_ERROR,
            )

        # Build corpus and filter examples with empty query field
        filtered_memory = []
        corpus = []
        for example in self._episodic_memory:
            if value := getattr(example, query_field, None):
                query = truncate_to_token_limit(
                    value_to_embedding_text(value), self._embedding_model, model_type="embedding"
                )
                corpus.append(query)
                filtered_memory.append(example)
        self._episodic_memory = filtered_memory

        self._retriever = dspy.retrievers.Embeddings(
            embedder=self._embedder, corpus=corpus, k=self._retrieval_k
        )
        _logger.debug(f"Episodic memory corpus contains {len(corpus)} examples")

    def _add_examples_to_memory(self, examples: list["dspy.Example"]) -> None:
        """Add examples by updating both episodic memory and semantic memory.

        Args:
            examples: Examples to add
        """
        # Update episodic memory and trace IDs
        self._episodic_memory.extend(examples)
        self._episodic_trace_ids.extend(ex._trace_id for ex in examples if hasattr(ex, "_trace_id"))
        self._build_episodic_memory()

        # Update semantic memory
        self._distill_new_guidelines(examples)


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
        retrieval_k: Number of similar examples to retrieve from episodic memory (default: 5)
        embedding_model: {{ embedding_model }}
        embedding_dim: Dimension of embeddings (default: 512)

    Note:
        The number of parallel threads for LLM calls during guideline distillation can be
        configured via the ``MLFLOW_GENAI_OPTIMIZE_MAX_WORKERS`` environment variable
        (default: 8). Increasing this value can speed up alignment when processing many
        feedback examples, but may increase API rate limit errors.

    Example:
        .. code-block:: python

            import mlflow
            from mlflow.genai.judges import make_judge
            from mlflow.genai.judges.optimizers import MemAlignOptimizer

            judge = make_judge(name="quality", instructions="...", model="openai:/gpt-4")
            optimizer = MemAlignOptimizer(
                reflection_lm="openai:/gpt-4o-mini",
                retrieval_k=3,
                embedding_model="openai:/text-embedding-3-small",
            )

            # Assuming `traces` contains human feedback for the judge
            optimized_judge = judge.align(traces=traces, optimizer=optimizer)
            result = optimized_judge(inputs="...", outputs="...")
    """

    def __init__(
        self,
        reflection_lm: str | None = None,
        retrieval_k: int = 5,
        embedding_model: str | None = None,
        embedding_dim: int = 512,
    ):
        _check_dspy_installed()
        self._reflection_lm = reflection_lm if reflection_lm is not None else get_default_model()
        self._retrieval_k = retrieval_k
        self._embedding_model = (
            embedding_model if embedding_model is not None else get_default_embedding_model()
        )
        self._embedding_dim = embedding_dim

    def align(self, judge: Judge, traces: list[Trace]) -> MemoryAugmentedJudge:
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

            memory_judge = MemoryAugmentedJudge(
                base_judge=judge,
                reflection_lm=self._reflection_lm,
                retrieval_k=self._retrieval_k,
                embedding_model=self._embedding_model,
                embedding_dim=self._embedding_dim,
            )

            memory_judge._add_examples_to_memory(new_examples)

            _logger.debug(f"MemAlign alignment completed successfully on {len(traces)} examples.")
            return memory_judge

        except Exception as e:
            _logger.error(f"MemAlign alignment failed: {e}", exc_info=True)
            raise MlflowException(
                f"Alignment optimization failed: {e!s}", error_code=INTERNAL_ERROR
            ) from e
