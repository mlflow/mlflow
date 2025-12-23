"""MemAlign optimizer with dual memory system."""

import logging
from collections import OrderedDict
from typing import Any

import dspy
import dspy.retrievers
from litellm import token_counter
from pydantic import ValidationError, create_model

from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.base import AlignmentOptimizer, Judge, JudgeField
from mlflow.genai.judges.optimizers.dspy_utils import (
    construct_dspy_lm,
    create_dspy_signature,
)
from mlflow.genai.judges.optimizers.memalign_config import MemAlignConfig
from mlflow.genai.judges.optimizers.memalign_utils import (
    Guideline,
    Guidelines,
    load_memalign_template,
    trace_to_feedback_record,
)
from mlflow.genai.judges.utils import get_default_model
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


class MemoryAugmentedJudge(Judge):
    """A judge augmented with dual memory systems.

    This judge wraps a base judge and enhances evaluation with:
    - Semantic Memory: Distilled guidelines from past feedback
    - Episodic Memory: Retrieved similar examples from past feedback

    The judge maintains state across evaluations and provides the exact same
    interface as the original judge. Memories are managed internally.
    """

    def __init__(
        self,
        base_judge: Judge,
        config: MemAlignConfig,
        base_signature: "dspy.Signature",
        feedback_records: list[dict[str, Any]],
    ):
        """Initialize memory-augmented judge.

        Args:
            base_judge: Original judge to wrap
            config: MemAlign configuration
            base_signature: DSPy signature from base judge
            feedback_records: Initial feedback records to populate memories
        """
        self._base_judge = base_judge
        self._config = config
        self._base_signature = base_signature

        self._InputModel, self._OutputModel, self._Example = (
            self._infer_pydantic_models_from_signature(base_signature)
        )

        self._feedback_history = OrderedDict()
        self._next_id = 1

        self._lm = construct_dspy_lm(config.model)

        if not config.disable_semantic_memory:
            self._semantic_memory: list[Guideline] = []
            self._distill_template = load_memalign_template(config.distill_prompt_template_name)
        else:
            self._semantic_memory = None

        if not config.disable_episodic_memory:
            self._episodic_memory = OrderedDict()
            self._embedder = dspy.Embedder(config.embedder_name, dimensions=config.embed_dim)
            self._k = config.retrieval_k
            self._search = None
        else:
            self._episodic_memory = None

        self._extended_signature = self._create_extended_signature()

        self._predict_module = dspy.Predict(self._extended_signature)
        self._predict_module.set_lm(self._lm)

        if feedback_records:
            self._add_feedback_records(feedback_records)

    def __call__(self, **kwargs) -> Feedback:
        """Evaluate with memory augmentation."""
        input_example = self._InputModel.model_validate(kwargs)

        guidelines = self._get_guideline_texts()
        relevant_examples = self._retrieve_relevant_feedback_records(input_example)

        prediction = self._predict_module(
            guidelines=guidelines,
            example_judgements=relevant_examples,
            **input_example.model_dump(),
        )

        return Feedback(
            value=prediction.result,
            rationale=prediction.rationale,
        )

    @property
    def name(self) -> str:
        return self._base_judge.name

    @property
    def instructions(self) -> str:
        base = self._base_judge.instructions
        memory_desc = "\n\nThis judge uses a dual memory system:"
        if not self._config.disable_semantic_memory and self._semantic_memory:
            memory_desc += f"\n- Semantic Memory: {len(self._semantic_memory)} guidelines"
        if not self._config.disable_episodic_memory and self._episodic_memory:
            memory_desc += f"\n- Episodic Memory: {len(self._episodic_memory)} examples"
        return base + memory_desc

    @property
    def model(self) -> str:
        return self._base_judge.model

    def get_input_fields(self) -> list[JudgeField]:
        return self._base_judge.get_input_fields()

    def unalign(self, traces: list[Trace]) -> "MemoryAugmentedJudge":
        """Remove specific traces from the judge's memory and return a new judge.

        Args:
            traces: List of traces to remove from the judge's memory

        Returns:
            A new MemoryAugmentedJudge with the traces removed
        """
        trace_ids_to_remove = {trace.info.trace_id for trace in traces}

        ids_to_remove = []
        for feedback_id, feedback_record in self._feedback_history.items():
            if feedback_record.get("_trace_id") in trace_ids_to_remove:
                ids_to_remove.append(feedback_id)

        if not ids_to_remove:
            _logger.warning("No feedback records found for the provided traces")
            return self

        new_feedback_history = OrderedDict()
        for feedback_id, record in self._feedback_history.items():
            if feedback_id not in ids_to_remove:
                new_feedback_history[feedback_id] = record

        feedback_records = [
            {k: v for k, v in record.items() if k != "_trace_id"}
            for record in new_feedback_history.values()
        ]

        return MemoryAugmentedJudge(
            base_judge=self._base_judge,
            config=self._config,
            base_signature=self._base_signature,
            feedback_records=feedback_records,
        )

    def _create_extended_signature(self) -> "dspy.Signature":
        """Create extended signature with memory fields."""
        extended_sig = self._base_signature

        if not self._config.disable_semantic_memory:
            guidelines_field = dspy.InputField(
                desc=(
                    "General guidelines you should always consider when evaluating an input. "
                    "IMPORTANT: Your output fields should NEVER directly refer to the presence "
                    "of these guidelines. Instead, weave the learned lessons into your reasoning."
                ),
                type=list[str],
            )
            extended_sig = extended_sig.prepend("guidelines", guidelines_field, type_=list[str])

        if not self._config.disable_episodic_memory:
            examples_field = dspy.InputField(
                desc=(
                    "Some example judgements (certain input fields might be omitted for "
                    "brevity). When evaluating the new input, try to align your judgements "
                    "with these examples. IMPORTANT: Your output fields should NEVER directly "
                    "refer to the presence of these examples. Instead, weave the learned "
                    "lessons into your reasoning."
                ),
                type=list[self._Example],
            )
            extended_sig = extended_sig.prepend(
                "example_judgements", examples_field, type_=list[self._Example]
            )

        return extended_sig

    def _add_feedback_records(self, feedback_records: list[dict[str, Any]]) -> None:
        """Add feedback records to memories."""
        try:
            validated_records = [
                self._Example.model_validate(record) for record in feedback_records
            ]
        except ValidationError as e:
            _logger.error(f"Failed to validate feedback records: {e}")
            raise MlflowException(
                "Invalid feedback records. Please ensure each record contains all the input "
                "and output fields defined in the signature.",
                error_code=INVALID_PARAMETER_VALUE,
            ) from e

        ids_for_current_records = []
        for record in feedback_records:
            feedback_id = self._next_id
            self._feedback_history[feedback_id] = record
            ids_for_current_records.append(feedback_id)
            self._next_id += 1

        if not self._config.disable_semantic_memory:
            self._distill_guidelines_into_semantic_memory(
                validated_records, ids_for_current_records
            )

        if not self._config.disable_episodic_memory:
            self._persist_feedback_records(validated_records, ids_for_current_records)

        _logger.info(f"Added {len(feedback_records)} feedback records to memories")

    def _distill_guidelines_into_semantic_memory(
        self, feedback_records: list[Any], ids_for_current_records: list[int]
    ) -> None:
        """Distill guidelines from feedback records."""
        flex_tokens = self._config.max_output_tokens + 5000
        prompt_tokens_limit = self._config.max_input_tokens - flex_tokens

        records_per_group = min(len(feedback_records), 50)
        while records_per_group > 0:
            tmp_prompt = self._distill_template.render(
                judge_instructions=self._base_judge.instructions,
                feedback_records=feedback_records[:records_per_group],
                ids=ids_for_current_records[:records_per_group],
                existing_guidelines=self._get_guideline_texts(),
                zip=zip,
                len=len,
            )
            try:
                tmp_prompt_tokens = token_counter(model=self._config.model, text=tmp_prompt)
            except Exception as e:
                _logger.warning(
                    f"Token counting failed for model {self._config.model}: {e}. "
                    "Using character-based estimate."
                )
                tmp_prompt_tokens = len(tmp_prompt) // 4

            if tmp_prompt_tokens <= prompt_tokens_limit:
                try:
                    trial_response = self._lm(
                        messages=[{"role": "user", "content": tmp_prompt}],
                        response_format=Guidelines,
                    )[0]
                    Guidelines.model_validate_json(trial_response)
                    break
                except ValidationError as e:
                    _logger.debug(f"Validation failed, reducing batch size: {e}")
                    records_per_group //= 2
                except Exception as e:
                    _logger.error(f"Trial LM call failed with non-retriable error: {e}")
                    raise MlflowException(
                        f"Failed to distill guidelines: {e!s}", error_code=INTERNAL_ERROR
                    ) from e
            else:
                records_per_group //= 2

        if records_per_group == 0:
            if not self._config.disable_episodic_memory:
                _logger.warning(
                    "No feedback records fit in guideline distillation prompt. "
                    "Continuing with episodic memory only."
                )
                return
            else:
                raise MlflowException(
                    "Feedback records too large to fit in prompt and episodic memory is disabled.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        for i in range(0, len(feedback_records), records_per_group):
            group = feedback_records[i : i + records_per_group]
            ids_for_group = ids_for_current_records[i : i + records_per_group]
            existing_guideline_texts = self._get_guideline_texts()

            distill_prompt = self._distill_template.render(
                judge_instructions=self._base_judge.instructions,
                feedback_records=group,
                ids=ids_for_group,
                existing_guidelines=existing_guideline_texts,
                zip=zip,
                len=len,
            )

            try:
                response = self._lm(
                    messages=[{"role": "user", "content": distill_prompt}],
                    response_format=Guidelines,
                )[0]
                new_guidelines = Guidelines.model_validate_json(response).guidelines
            except Exception as e:
                _logger.error(f"Failed to distill guidelines: {e}")
                continue

            if self._config.dedup_guidelines:
                new_guidelines = [
                    guideline
                    for guideline in new_guidelines
                    if guideline.guideline_text not in existing_guideline_texts
                ]

            self._semantic_memory.extend(new_guidelines)

        _logger.info(f"Semantic memory now contains {len(self._semantic_memory)} guidelines")

    def _persist_feedback_records(
        self, feedback_records: list[Any], ids_for_current_records: list[int]
    ) -> None:
        """Store feedback records in episodic memory."""
        self._episodic_memory.update(dict(zip(ids_for_current_records, feedback_records)))

        corpus = [
            self._extract_query_from_input(record) for record in self._episodic_memory.values()
        ]
        self._search = dspy.retrievers.Embeddings(embedder=self._embedder, corpus=corpus, k=self._k)

        _logger.info(f"Episodic memory now contains {len(self._episodic_memory)} examples")

    def _retrieve_relevant_feedback_records(self, input_example: Any) -> list[Any]:
        """Retrieve relevant examples from episodic memory."""
        if not self._episodic_memory or self._search is None:
            return []

        query = self._extract_query_from_input(input_example)
        search_results = self._search(query)
        search_results_ids = search_results.indices

        memory_values = list(self._episodic_memory.values())
        return [memory_values[i] for i in search_results_ids if 0 <= i < len(memory_values)]

    def _extract_query_from_input(self, input_example: Any) -> str:
        """Extract query-like field from input for similarity matching."""
        if hasattr(input_example, "query"):
            return str(input_example.query)
        elif hasattr(input_example, "input"):
            return str(input_example.input)
        elif hasattr(input_example, "inputs"):
            return str(input_example.inputs)
        elif hasattr(input_example, "text"):
            return str(input_example.text)
        elif hasattr(input_example, "question"):
            return str(input_example.question)
        else:
            return str(input_example)

    def _get_guideline_texts(self) -> list[str]:
        """Get list of guideline texts from semantic memory."""
        if not self._semantic_memory:
            return []
        return [guideline.guideline_text for guideline in self._semantic_memory]

    def _infer_pydantic_models_from_signature(self, signature_class):
        """Generate Pydantic models from DSPy signature."""
        input_fields = {}
        for name, field_info in signature_class.input_fields.items():
            field_type = field_info.annotation or str
            input_fields[name] = (field_type, field_info.default)

        output_fields = {}
        for name, field_info in signature_class.output_fields.items():
            field_type = field_info.annotation or str
            output_fields[name] = (field_type, field_info.default)

        InputModel = create_model(f"{signature_class.__name__}Input", **input_fields)
        OutputModel = create_model(f"{signature_class.__name__}Output", **output_fields)
        FullModel = create_model(
            f"{signature_class.__name__}Example", **input_fields, **output_fields
        )

        return InputModel, OutputModel, FullModel


@experimental(version="3.6.0")
class MemAlignOptimizer(AlignmentOptimizer):
    """MemAlign alignment optimizer using dual memory systems.

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

    Usage:
        Works with the standard Judge.align() interface - no API changes needed.

        >>> judge = make_judge(name="quality", instructions="...", model="openai:/gpt-4")
        >>> optimizer = MemAlignOptimizer(model="openai:/gpt-4", retrieval_k=3)
        >>> optimized_judge = judge.align(traces=traces, optimizer=optimizer)
        >>> result = optimized_judge(inputs={...}, outputs={...})
    """

    def __init__(
        self,
        model: str | None = None,
        disable_semantic_memory: bool = False,
        disable_episodic_memory: bool = False,
        distill_prompt_template: str = "memalign_distill_guidelines.txt",
        dedup_guidelines: bool = True,
        retrieval_k: int = 5,
        embedder_name: str = "openai/text-embedding-3-small",
        embed_dim: int = 512,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        **kwargs,
    ):
        """Initialize MemAlign optimizer.

        Args:
            model: Model for guideline distillation. If None, uses default.
            disable_semantic_memory: Disable guideline distillation.
            disable_episodic_memory: Disable example retrieval.
            distill_prompt_template: Template for guideline extraction.
            dedup_guidelines: Remove duplicate guidelines.
            retrieval_k: Number of examples to retrieve.
            embedder_name: Embedding model name (LiteLLM format).
            embed_dim: Embedding dimension.
            temperature: Temperature for LLM calls.
            max_output_tokens: Max tokens for guideline distillation.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self._model = model if model is not None else get_default_model()

        if disable_semantic_memory and disable_episodic_memory:
            raise MlflowException(
                "At least one memory system must be enabled. "
                "Cannot disable both semantic and episodic memory.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        self._config = MemAlignConfig(
            model=self._model,
            disable_semantic_memory=disable_semantic_memory,
            disable_episodic_memory=disable_episodic_memory,
            distill_prompt_template_name=distill_prompt_template,
            dedup_guidelines=dedup_guidelines,
            retrieval_k=retrieval_k,
            embedder_name=embedder_name,
            embed_dim=embed_dim,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def align(self, judge: Judge, traces: list[Trace]) -> Judge:
        """Align a judge using dual memory systems.

        Args:
            judge: The judge to augment with memory
            traces: List of traces containing human feedback

        Returns:
            A MemoryAugmentedJudge with populated memories
        """
        try:
            if not traces:
                raise MlflowException(
                    "No traces provided for alignment", error_code=INVALID_PARAMETER_VALUE
                )

            _logger.info(f"Starting MemAlign alignment with {len(traces)} traces")

            signature = create_dspy_signature(judge)

            feedback_records = []
            for trace in traces:
                record = trace_to_feedback_record(trace, judge)
                if record is not None:
                    feedback_records.append(record)

            if not feedback_records:
                raise MlflowException(
                    f"No valid feedback records found in traces. "
                    f"Ensure traces contain human assessments with name '{judge.name}'",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            _logger.info(
                f"Created {len(feedback_records)} feedback records from {len(traces)} traces"
            )

            memory_judge = MemoryAugmentedJudge(
                base_judge=judge,
                config=self._config,
                base_signature=signature,
                feedback_records=feedback_records,
            )

            _logger.info("MemAlign alignment completed successfully")
            return memory_judge

        except Exception as e:
            _logger.error(f"MemAlign alignment failed: {e}")
            raise MlflowException(
                f"Alignment optimization failed: {e!s}", error_code=INTERNAL_ERROR
            ) from e
