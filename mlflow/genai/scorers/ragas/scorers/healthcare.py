from __future__ import annotations

from typing import ClassVar

from mlflow.genai.judges.builtin import _MODEL_API_DOC
from mlflow.genai.scorers.ragas import RagasScorer
from mlflow.utils.annotations import experimental
from mlflow.utils.docstring_utils import format_docstring

@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class ClinicalAccuracy(RagasScorer):
    """Evaluate how clinically accurate a model's responses are for healthcare use cases.

    This metric assesses whether the model's answers are clinically sound, consistent
    with the provided medical context, and free of harmful or misleading guidance.
    It is intended for evaluating models used in scenarios such as triage support,
    summarization of clinical notes, or medical question answering, where incorrect
    information can have patient-safety implications.

    The scorer relies on the underlying Ragas-based implementation registered under
    the ``"ClinicalAccuracy"`` metric name.

    Args:
        model: {model_api_doc}
            The evaluation model used by the scorer to judge the quality of the
            generated responses.
        input: The input data containing prompts and expected references, typically
            passed via :func:`mlflow.evaluate` as part of an evaluation dataset.

    Examples:
        Evaluate clinical accuracy for a chat model:

        .. code-block:: python

            import mlflow

            result = mlflow.evaluate(
                model="models:/my_clinical_chat_model@prod",
                data=evaluation_data,  # e.g., prompts, references, context
                evaluators="default",
                metrics=["ClinicalAccuracy"],
            )

        Use the scorer directly:

        .. code-block:: python

            from mlflow.genai.scorers.ragas.scorers.healthcare import ClinicalAccuracy

            scorer = ClinicalAccuracy(model="gpt-4o")
            score = scorer.score(input=evaluation_data)
    """

    metric_name: ClassVar[str] = "ClinicalAccuracy"

@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class HIPAACompliance(RagasScorer):
    """Evaluate whether model responses comply with HIPAA-related privacy constraints.

    This metric focuses on detecting potential violations of healthcare privacy
    regulations in generated content, such as inappropriate disclosure of Protected
    Health Information (PHI) or mishandling of sensitive patient data.

    The scorer relies on the underlying Ragas-based implementation registered under
    the ``"HIPAACompliance"`` metric name.

    Args:
        model: {model_api_doc}
            The evaluation model used by the scorer to judge the compliance of the
            generated responses.
        input: The input data containing prompts and model outputs to be checked for
            HIPAA compliance, typically passed via :func:`mlflow.evaluate`.

    Examples:
        Evaluate HIPAA compliance for a clinical assistant:

        .. code-block:: python

            import mlflow

            result = mlflow.evaluate(
                model="models:/my_healthcare_assistant@staging",
                data=evaluation_data,  # prompts and model outputs
                evaluators="default",
                metrics=["HIPAACompliance"],
            )

        Use the scorer directly:

        .. code-block:: python

            from mlflow.genai.scorers.ragas.scorers.healthcare import HIPAACompliance

            scorer = HIPAACompliance(model="gpt-4o")
            score = scorer.score(input=evaluation_data)
    """

    metric_name: ClassVar[str] = "HIPAACompliance"

@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class SourceAttribution(RagasScorer):
    """Evaluate whether model responses properly attribute and align with medical sources.

    This metric measures how well the model grounds its answers in the provided
    clinical documents, guidelines, or other reference materials. It is useful
    when you want to ensure that generated content is traceable back to trusted
    sources, such as clinical guidelines or patient records.

    The scorer relies on the underlying Ragas-based implementation registered under
    the ``"SourceAttribution"`` metric name.

    Args:
        model: {model_api_doc}
            The evaluation model used by the scorer to judge the alignment between
            responses and reference sources.
        input: The input data including context documents, prompts, and model
            outputs, typically passed via :func:`mlflow.evaluate`.

    Examples:
        Evaluate source attribution for a retrieval-augmented clinical QA system:

        .. code-block:: python

            import mlflow

            result = mlflow.evaluate(
                model="models:/my_clinical_rag_model@prod",
                data=evaluation_data,  # context + prompts + outputs
                evaluators="default",
                metrics=["SourceAttribution"],
            )

        Use the scorer directly:

        .. code-block:: python

            from mlflow.genai.scorers.ragas.scorers.healthcare import SourceAttribution

            scorer = SourceAttribution(model="gpt-4o")
            score = scorer.score(input=evaluation_data)
    """

    metric_name: ClassVar[str] = "SourceAttribution"

@experimental(version="3.8.0")
@format_docstring(_MODEL_API_DOC)
class MedicalTerminologyConsistency(RagasScorer):
    """Evaluate the correctness and consistency of medical terminology in responses.

    This metric checks whether a model uses appropriate, precise, and consistent
    medical terminology when generating content. It is particularly useful for
    tasks such as clinical documentation, patient education materials, and
    summarization of medical records, where inaccurate terminology can cause
    confusion or clinical risk.

    The scorer relies on the underlying Ragas-based implementation registered under
    the ``"MedicalTerminologyConsistency"`` metric name.

    Args:
        model: {model_api_doc}
            The evaluation model used by the scorer to judge terminology usage in
            generated responses.
        input: The input data containing prompts and model outputs to be evaluated,
            typically passed via :func:`mlflow.evaluate`.

    Examples:
        Evaluate terminology consistency for a note-generation model:

        .. code-block:: python

            import mlflow

            result = mlflow.evaluate(
                model="models:/my_note_generation_model@prod",
                data=evaluation_data,  # prompts and outputs
                evaluators="default",
                metrics=["MedicalTerminologyConsistency"],
            )

        Use the scorer directly:

        .. code-block:: python

            from mlflow.genai.scorers.ragas.scorers.healthcare import (
                MedicalTerminologyConsistency,
            )

            scorer = MedicalTerminologyConsistency(model="gpt-4o")
            score = scorer.score(input=evaluation_data)
    """
    metric_name: ClassVar[str] = "MedicalTerminologyConsistency"
