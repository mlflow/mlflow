"""MemAlign optimizer implementation."""

import logging
import os
from collections import OrderedDict
from typing import Any

import dspy
import dspy.retrievers

from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.base import AlignmentOptimizer, Judge, JudgeField
from mlflow.genai.judges.optimizers.dspy_utils import (
    construct_dspy_lm,
    create_dspy_signature,
    trace_to_dspy_example,
)
from mlflow.genai.judges.utils import get_default_model
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


def _load_distillation_template():
    """Load the guideline distillation template."""
    from jinja2 import Environment, FileSystemLoader

    template_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(template_dir))
    return env.get_template("distillation_guidelines.txt")


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
        base_signature: "dspy.Signature",
        distillation_model: str,
        retrieval_k: int,
        embedder_name: str,
        embed_dim: int,
        examples: list["dspy.Example"],
    ):
        """Initialize memory-augmented judge.

        Args:
            base_judge: Original judge to wrap
            base_signature: DSPy signature from base judge
            distillation_model: Model for guideline distillation
            retrieval_k: Number of examples to retrieve
            embedder_name: Embedding model name
            embed_dim: Embedding dimension
            examples: Initial DSPy examples to populate memories
        """
        self._base_judge = base_judge
        self._base_signature = base_signature
        self._retrieval_k = retrieval_k

        self._examples = OrderedDict()
        self._next_id = 1

        self._distillation_model = distillation_model
        self._distillation_lm = construct_dspy_lm(distillation_model)
        self._distill_template = _load_distillation_template()

        self._semantic_memory: list[str] = []

        self._embedder_name = embedder_name
        self._embed_dim = embed_dim
        self._embedder = dspy.Embedder(embedder_name, dimensions=embed_dim)
        self._search = None

        self._extended_signature = self._create_extended_signature()

        self._predict_module = dspy.Predict(self._extended_signature)
        self._predict_module.set_lm(construct_dspy_lm(base_judge.model))

        if examples:
            self._add_examples(examples)

    def __call__(self, **kwargs) -> Feedback:
        """Evaluate with memory augmentation."""
        guidelines = self._semantic_memory
        relevant_examples = self._retrieve_relevant_examples(kwargs)

        prediction = self._predict_module(
            guidelines=guidelines,
            example_judgements=relevant_examples,
            **kwargs,
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
        instructions = self._base_judge.instructions
        if self._semantic_memory:
            instructions += f"\n\nDistilled Guidelines ({len(self._semantic_memory)}):\n"
            for guideline in self._semantic_memory[:5]:
                instructions += f"  - {guideline}\n"
        return instructions

    @property
    def model(self) -> str:
        return self._base_judge.model

    def get_input_fields(self) -> list[JudgeField]:
        return self._base_judge.get_input_fields()

    def unalign(self, traces: list[Trace]) -> "MemoryAugmentedJudge":
        """Remove specific traces from memory and return new judge.

        Args:
            traces: Traces to remove from memory

        Returns:
            A new MemoryAugmentedJudge with the traces removed
        """
        trace_ids_to_remove = {trace.info.trace_id for trace in traces}

        ids_to_remove = []
        for example_id, example in self._examples.items():
            if hasattr(example, "_trace_id") and example._trace_id in trace_ids_to_remove:
                ids_to_remove.append(example_id)

        if not ids_to_remove:
            _logger.warning("No feedback records found for the provided traces")
            return self

        new_examples = OrderedDict()
        for example_id, example in self._examples.items():
            if example_id not in ids_to_remove:
                new_examples[example_id] = example

        filtered_examples = list(new_examples.values())

        return MemoryAugmentedJudge(
            base_judge=self._base_judge,
            base_signature=self._base_signature,
            distillation_model=self._distillation_model,
            retrieval_k=self._retrieval_k,
            embedder_name=self._embedder_name,
            embed_dim=self._embed_dim,
            examples=filtered_examples,
        )

    def _create_extended_signature(self) -> "dspy.Signature":
        """Create extended signature with memory fields."""
        guidelines_field = dspy.InputField(
            desc=(
                "General guidelines you should always consider when evaluating an input. "
                "IMPORTANT: Your output fields should NEVER directly refer to the presence "
                "of these guidelines. Instead, weave the learned lessons into your reasoning."
            )
        )
        extended_sig = self._base_signature.prepend("guidelines", guidelines_field)

        examples_field = dspy.InputField(
            desc=(
                "Some example judgements (certain input fields might be omitted for "
                "brevity). When evaluating the new input, try to align your judgements "
                "with these examples. IMPORTANT: Your output fields should NEVER directly "
                "refer to the presence of these examples. Instead, weave the learned "
                "lessons into your reasoning."
            )
        )
        return extended_sig.prepend("example_judgements", examples_field)

    def _add_examples(self, examples: list["dspy.Example"]) -> None:
        """Add examples to memories."""
        for example in examples:
            example_id = self._next_id
            self._examples[example_id] = example
            self._next_id += 1

        self._distill_guidelines(list(self._examples.values()))
        self._update_episodic_memory(list(self._examples.values()))

        _logger.info(f"Added {len(examples)} examples to memories")

    def _distill_guidelines(self, examples: list["dspy.Example"]) -> None:
        """Distill guidelines from examples using LLM."""
        if not examples:
            return

        existing_guidelines = self._semantic_memory[:]

        examples_data = []
        for example in examples:
            example_dict = {}
            for field_name in self._base_signature.input_fields:
                if hasattr(example, field_name):
                    example_dict[field_name] = getattr(example, field_name)
            for field_name in self._base_signature.output_fields:
                if hasattr(example, field_name):
                    example_dict[field_name] = getattr(example, field_name)
            examples_data.append(example_dict)

        prompt = self._distill_template.render(
            judge_instructions=self._base_judge.instructions,
            feedback_records=examples_data,
            ids=list(range(len(examples_data))),
            existing_guidelines=existing_guidelines,
            zip=zip,
            len=len,
        )

        try:
            from pydantic import BaseModel

            class Guideline(BaseModel):
                guideline_text: str
                source_ids: list[int] | None = None

            class Guidelines(BaseModel):
                guidelines: list[Guideline]

            response = self._distillation_lm(
                messages=[{"role": "user", "content": prompt}],
                response_format=Guidelines,
            )[0]

            result = Guidelines.model_validate_json(response)

            new_guidelines = [g.guideline_text for g in result.guidelines]

            new_guidelines = [
                guideline for guideline in new_guidelines if guideline not in existing_guidelines
            ]

            self._semantic_memory.extend(new_guidelines)

            _logger.info(f"Semantic memory now contains {len(self._semantic_memory)} guidelines")

        except Exception as e:
            _logger.error(f"Failed to distill guidelines: {e}")

    def _update_episodic_memory(self, examples: list["dspy.Example"]) -> None:
        """Update episodic memory with embeddings."""
        corpus = []
        for example in examples:
            query_parts = []
            for field_name in self._base_signature.input_fields:
                if hasattr(example, field_name):
                    value = getattr(example, field_name)
                    if value is not None:
                        query_parts.append(str(value))
            corpus.append(" ".join(query_parts))

        self._search = dspy.retrievers.Embeddings(
            embedder=self._embedder, corpus=corpus, k=self._retrieval_k
        )

        _logger.info(f"Episodic memory now contains {len(examples)} examples")

    def _retrieve_relevant_examples(self, query_kwargs: dict[str, Any]) -> list["dspy.Example"]:
        """Retrieve relevant examples from episodic memory."""
        if not self._examples or self._search is None:
            return []

        query_parts = [
            str(query_kwargs[field_name])
            for field_name in self._base_signature.input_fields
            if field_name in query_kwargs and query_kwargs[field_name] is not None
        ]
        query = " ".join(query_parts)
        search_results = self._search(query)
        search_results_ids = search_results.indices

        example_values = list(self._examples.values())
        return [example_values[i] for i in search_results_ids if 0 <= i < len(example_values)]


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
        >>> optimizer = MemAlignOptimizer(distillation_model="openai:/gpt-4", retrieval_k=3)
        >>> optimized_judge = judge.align(traces=traces, optimizer=optimizer)
        >>> result = optimized_judge(inputs={...}, outputs={...})
    """

    def __init__(
        self,
        distillation_model: str | None = None,
        retrieval_k: int = 5,
        embedder_name: str = "openai/text-embedding-3-small",
        embed_dim: int = 512,
        **kwargs,
    ):
        """Initialize MemAlign optimizer.

        Args:
            distillation_model: Model for guideline distillation. If None, uses default.
            retrieval_k: Number of examples to retrieve from episodic memory.
            embedder_name: Embedding model name in LiteLLM format.
            embed_dim: Embedding dimension.
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self._distillation_model = (
            distillation_model if distillation_model is not None else get_default_model()
        )
        self._retrieval_k = retrieval_k
        self._embedder_name = embedder_name
        self._embed_dim = embed_dim

    def align(self, judge: Judge, traces: list[Trace]) -> Judge:
        """Align judge with human feedback from traces.

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

            _logger.info(f"Starting MemAlign alignment with {len(traces)} traces")

            signature = create_dspy_signature(judge)

            examples = []
            for trace in traces:
                example = trace_to_dspy_example(trace, judge)
                if example is not None:
                    example._trace_id = trace.info.trace_id
                    examples.append(example)

            if not examples:
                raise MlflowException(
                    f"No valid feedback records found in traces. "
                    f"Ensure traces contain human assessments with name '{judge.name}'",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            _logger.info(f"Created {len(examples)} feedback records from {len(traces)} traces")

            memory_judge = MemoryAugmentedJudge(
                base_judge=judge,
                base_signature=signature,
                distillation_model=self._distillation_model,
                retrieval_k=self._retrieval_k,
                embedder_name=self._embedder_name,
                embed_dim=self._embed_dim,
                examples=examples,
            )

            _logger.info("MemAlign alignment completed successfully")
            return memory_judge

        except Exception as e:
            _logger.error(f"MemAlign alignment failed: {e}")
            raise MlflowException(
                f"Alignment optimization failed: {e!s}", error_code=INTERNAL_ERROR
            ) from e
