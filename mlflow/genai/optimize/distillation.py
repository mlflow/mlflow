"""
Distillation job implementation for ICL-KD (In-Context Learning based Knowledge Distillation).

This module implements knowledge transfer from a teacher model to a student model
through prompt optimization.

**Workflow:**
1. User runs their agent/prompt over a dataset (traces are auto-captured by MLflow)
2. User calls the distillation API with their student prompt URI
3. The job automatically finds traces linked to this prompt
4. Extracts teacher responses from traces using two-stage matching
5. Creates a distillation dataset (inputs + teacher responses as expected_response)
6. Optimizes the student prompt to match teacher responses using SemanticMatch scorer

This approach is simple and natural - users just need to run their agent once,
then distill to a smaller/cheaper model. No manual dataset creation required.

Reference: ICL-KD paper on In-Context Learning based Knowledge Distillation.
"""

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.datasets import create_dataset
from mlflow.genai.optimize import optimize_prompts
from mlflow.genai.optimize.distillation_utils import (
    extract_from_span,
    extract_matching_spans,
    find_llm_spans,
)
from mlflow.genai.optimize.job import (
    _build_predict_fn,
    _create_optimizer,
    _load_scorers,
)
from mlflow.genai.prompts import load_prompt
from mlflow.genai.utils.trace_utils import extract_response_from_trace
from mlflow.server.jobs import job
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.fluent import set_experiment, start_run

_logger = logging.getLogger(__name__)

_DEFAULT_DISTILLATION_JOB_MAX_WORKERS = 2


@dataclass
class DistillationJobResult:
    """Result of a distillation job."""

    run_id: str
    student_prompt_uri: str
    optimized_prompt_uri: str | None
    distillation_dataset_id: str
    num_traces: int
    num_samples: int
    optimizer_name: str
    initial_eval_score: float | None
    final_eval_score: float | None


def _parse_prompt_uri(prompt_uri: str) -> tuple[str, str]:
    """
    Parse a prompt URI into name and version.

    Args:
        prompt_uri: Prompt URI in format "prompts:/name/version"

    Returns:
        Tuple of (prompt_name, version)

    Raises:
        MlflowException: If URI format is invalid
    """
    if not prompt_uri.startswith("prompts:/"):
        raise MlflowException(
            f"Invalid prompt URI format: {prompt_uri}. Expected 'prompts:/name/version'"
        )

    # Remove "prompts:/" prefix and split
    path = prompt_uri[9:]  # len("prompts:/") == 9
    parts = path.strip("/").split("/")

    if len(parts) != 2:
        raise MlflowException(
            f"Invalid prompt URI format: {prompt_uri}. Expected 'prompts:/name/version'"
        )

    return parts[0], parts[1]


def search_traces_by_prompt(
    experiment_id: str,
    prompt_uri: str,
    max_traces: int | None = None,
) -> list:
    """
    Search for traces linked to a specific prompt.

    Args:
        experiment_id: Experiment ID to search in.
        prompt_uri: Prompt URI (e.g., "prompts:/my_prompt/1").
        max_traces: Maximum number of traces to return. None for all.

    Returns:
        List of Trace objects linked to the prompt.
    """
    prompt_name, version = _parse_prompt_uri(prompt_uri)

    # Build filter string for prompt search
    # The filter syntax is: prompt = 'prompt_name/version'
    filter_string = f"prompt = '{prompt_name}/{version}'"

    _logger.info(f"Searching traces with filter: {filter_string}")

    # Use mlflow.search_traces to find all traces linked to this prompt
    traces = mlflow.search_traces(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        max_results=max_traces,
        return_type="list",
    )

    _logger.info(f"Found {len(traces)} traces linked to prompt {prompt_uri}")
    return traces


def extract_from_traces(
    traces: list,
    prompt_template: str | list[dict[str, Any]] | None = None,
) -> list[tuple[dict[str, Any], str]]:
    """
    Extract input variables and responses from a list of traces.

    This function is the core of the distillation workflow. It processes
    traces and extracts the data needed for training.

    Uses two-stage matching when a prompt_template is provided:
    1. Fast vague match to filter candidate spans
    2. LLM-based verification and variable extraction

    Args:
        traces: List of Trace objects to extract from.
        prompt_template: Optional prompt template for variable extraction and filtering.
            Can be either:
            - A string template for text prompts
            - A list of message dicts for chat prompts
            If provided, only LLM spans matching this template are extracted.

    Returns:
        List of (inputs, response) tuples extracted from traces.
    """
    results = []
    total = len(traces)

    for idx, trace in enumerate(traces):
        if trace is None:
            _logger.warning(f"Trace {idx} is None, skipping")
            continue

        try:
            extracted_count = 0

            if prompt_template:
                # Use two-stage matching (vague match + LLM) to find ALL matching spans
                matches = extract_matching_spans(trace, prompt_template)
                for inputs, response in matches:
                    results.append((inputs, response))
                    extracted_count += 1
                    _logger.debug(
                        f"Extracted response from trace {idx + 1}/{total}: {len(response)} chars"
                    )
            else:
                # No template - extract from first LLM span with fallbacks
                llm_spans = find_llm_spans(trace)
                inputs = None
                response = None
                if llm_spans:
                    inputs, response = extract_from_span(llm_spans[0], None)

                # Fallback for output - try root span if LLM span extraction failed
                if response is None:
                    response = extract_response_from_trace(trace)

                # Fallback for inputs - use empty dict
                if inputs is None:
                    inputs = {}

                if response:
                    results.append((inputs, response))
                    extracted_count = 1
                    _logger.debug(f"Extracted response {idx + 1}/{total}: {len(response)} chars")

            if extracted_count == 0:
                _logger.warning(f"Could not extract response from trace {idx}")
        except Exception as e:
            _logger.warning(f"Failed to extract from trace {idx}: {e}")
            continue

    _logger.info(f"Extracted {len(results)} samples from {total} traces")
    return results


def create_distillation_dataset(
    extracted_data: list[tuple[dict[str, Any], str]],
    experiment_id: str,
    prompt_name: str,
) -> str:
    """
    Create a new dataset with extracted responses as expected_response.

    Args:
        extracted_data: List of (inputs, response) tuples.
        experiment_id: Experiment to associate dataset with.
        prompt_name: Name of the source prompt (for naming the dataset).

    Returns:
        New dataset ID.
    """
    records = [
        {
            "inputs": inputs,
            "expectations": {"expected_response": response},
        }
        for inputs, response in extracted_data
    ]

    # Use prompt name and timestamp for unique dataset name
    safe_prompt_name = prompt_name.replace("/", "_")[:20]
    dataset_name = f"distillation_{safe_prompt_name}_{int(time.time())}"
    dataset = create_dataset(name=dataset_name, experiment_id=experiment_id)
    dataset.merge_records(records)

    _logger.info(f"Created distillation dataset '{dataset_name}' with {len(records)} records")
    return dataset.dataset_id


@dataclass
class DistillationDatasetResult:
    """Result of creating a distillation dataset from traces."""

    dataset_id: str
    dataset_name: str
    num_traces: int
    num_samples: int
    source_prompt_uri: str


def create_distillation_dataset_from_prompt(
    experiment_id: str,
    source_prompt_uri: str,
    prompt_template: str | list[dict[str, Any]] | None = None,
    max_traces: int | None = None,
    dataset_name: str | None = None,
) -> DistillationDatasetResult:
    """
    Search traces linked to a prompt and create a distillation dataset.

    This is a convenience function that combines:
    1. Searching for traces linked to a prompt
    2. Extracting inputs and responses from traces
    3. Creating a dataset with expected_response for distillation

    Args:
        experiment_id: Experiment ID to search traces in and create dataset.
        source_prompt_uri: URI of the prompt whose traces to use (e.g., "prompts:/my_prompt/1").
        prompt_template: Optional prompt template for variable extraction.
            If None, will load the template from source_prompt_uri.
            Can be either:
            - A string template for text prompts
            - A list of message dicts for chat prompts
        max_traces: Maximum number of traces to use. None for all available.
        dataset_name: Optional custom name for the dataset. If None, auto-generated.

    Returns:
        DistillationDatasetResult with dataset info and statistics.

    Example:
        ```python
        import mlflow
        from mlflow.genai.optimize.distillation import create_distillation_dataset_from_prompt

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("my_experiment")

        # Create distillation dataset from traces linked to a prompt
        result = create_distillation_dataset_from_prompt(
            experiment_id="123",
            source_prompt_uri="prompts:/my_teacher_prompt/1",
            max_traces=100,
        )

        print(f"Created dataset: {result.dataset_id}")
        print(f"Traces processed: {result.num_traces}")
        print(f"Samples extracted: {result.num_samples}")
        ```
    """
    # Parse prompt URI
    prompt_name, prompt_version = _parse_prompt_uri(source_prompt_uri)

    # Search for traces
    _logger.info(f"Searching for traces linked to {source_prompt_uri}...")
    traces = search_traces_by_prompt(
        experiment_id=experiment_id,
        prompt_uri=source_prompt_uri,
        max_traces=max_traces,
    )

    if not traces:
        raise MlflowException(
            f"No traces found linked to prompt {source_prompt_uri}. "
            "Please run your agent/prompt over your data with MLflow tracing enabled, "
            "and ensure the prompt is loaded using mlflow.genai.load_prompt() within "
            "the traced code."
        )

    _logger.info(f"Found {len(traces)} traces")

    import pdb

    pdb.set_trace()

    # Load prompt template if not provided
    if prompt_template is None:
        prompt = load_prompt(source_prompt_uri)
        prompt_template = prompt.template

    # Extract from traces
    _logger.info("Extracting inputs and responses from traces...")
    extracted_data = extract_from_traces(
        traces=traces,
        prompt_template=prompt_template,
    )

    import pdb

    pdb.set_trace()

    if not extracted_data:
        raise MlflowException(
            "No responses were extracted from the traces. "
            "Check that traces contain LLM spans matching the prompt template."
        )

    _logger.info(f"Extracted {len(extracted_data)} samples from {len(traces)} traces")

    # Create dataset records
    records = [
        {
            "inputs": inputs,
            "expectations": {"expected_response": response},
        }
        for inputs, response in extracted_data
    ]

    # Generate dataset name if not provided
    if dataset_name is None:
        safe_prompt_name = prompt_name.replace("/", "_")[:20]
        dataset_name = f"distillation_{safe_prompt_name}_{int(time.time())}"

    # Create and populate dataset
    dataset = create_dataset(name=dataset_name, experiment_id=experiment_id)
    dataset.merge_records(records)

    _logger.info(f"Created distillation dataset '{dataset_name}' with {len(records)} records")

    return DistillationDatasetResult(
        dataset_id=dataset.dataset_id,
        dataset_name=dataset_name,
        num_traces=len(traces),
        num_samples=len(extracted_data),
        source_prompt_uri=source_prompt_uri,
    )


def run_distillation(
    run_id: str,
    experiment_id: str,
    student_prompt_uri: str,
    source_prompt_uri: str,
    optimizer_type: str,
    optimizer_config: dict[str, Any] | None,
    max_traces: int | None = None,
) -> dict[str, Any]:
    """
    Main distillation workflow.

    This function performs the ICL-KD workflow:
    1. Search for traces linked to the source prompt
    2. Extract responses from traces using two-stage matching
    3. Create distillation dataset with extracted responses as expected outputs
    4. Optimize student prompt to match the responses using SemanticMatch scorer

    Args:
        run_id: MLflow run ID for tracking.
        experiment_id: Experiment ID for tracking.
        student_prompt_uri: URI of the student prompt to optimize.
        source_prompt_uri: URI of the prompt whose traces to use as training data.
            This is typically the teacher prompt that was used to generate the traces.
        optimizer_type: Optimizer type (e.g., "gepa", "metaprompt").
        optimizer_config: Optimizer-specific configuration.
        max_traces: Maximum number of traces to use. None for all available.

    Returns:
        Dict containing distillation results (JSON-serializable).
    """
    from mlflow.genai.datasets import get_dataset

    set_experiment(experiment_id=experiment_id)

    with start_run(run_id=run_id):
        client = MlflowClient()
        start_time = time.time()

        # Parse source prompt URI for naming
        source_prompt_name, source_prompt_version = _parse_prompt_uri(source_prompt_uri)

        # Phase 1: Search for traces linked to the source prompt
        mlflow.set_tag("distillation.status", "searching_traces")
        mlflow.log_param("source_prompt_uri", source_prompt_uri)

        _logger.info(f"Searching for traces linked to {source_prompt_uri}...")
        traces = search_traces_by_prompt(
            experiment_id=experiment_id,
            prompt_uri=source_prompt_uri,
            max_traces=max_traces,
        )

        if not traces:
            raise MlflowException(
                f"No traces found linked to prompt {source_prompt_uri}. "
                "Please run your agent/prompt over your data with MLflow tracing enabled, "
                "and ensure the prompt is loaded using mlflow.genai.load_prompt() within "
                "the traced code."
            )

        mlflow.log_metric("num_traces_found", len(traces))

        # Phase 2: Extract from traces
        mlflow.set_tag("distillation.status", "extracting_from_traces")

        # Load student prompt to get template for variable extraction
        student_prompt = load_prompt(student_prompt_uri)
        prompt_template = student_prompt.template

        _logger.info(f"Extracting responses from {len(traces)} traces...")
        extracted_data = extract_from_traces(
            traces=traces,
            prompt_template=prompt_template,
        )

        extraction_time = time.time() - start_time
        mlflow.log_metric("extraction_time_seconds", extraction_time)
        _logger.info(f"Extracted {len(extracted_data)} samples in {extraction_time:.1f}s")

        if not extracted_data:
            raise MlflowException(
                "No responses were extracted from the traces. "
                "Check that traces contain LLM spans matching the prompt template."
            )

        mlflow.log_metric("num_samples", len(extracted_data))

        # Phase 3: Create distillation dataset
        mlflow.set_tag("distillation.status", "creating_dataset")
        distillation_dataset_id = create_distillation_dataset(
            extracted_data=extracted_data,
            experiment_id=experiment_id,
            prompt_name=source_prompt_name,
        )
        mlflow.log_param("distillation_dataset_id", distillation_dataset_id)

        # Phase 4: Optimize student prompt
        mlflow.set_tag("distillation.status", "optimization_started")
        _logger.info("Starting student prompt optimization...")

        # Link student prompt to run
        client.link_prompt_version_to_run(run_id=run_id, prompt=student_prompt)

        # Build predict function for student
        student_predict_fn = _build_predict_fn(student_prompt_uri)

        # Load optimizer and scorers
        optimizer = _create_optimizer(optimizer_type, optimizer_config)
        scorers = _load_scorers(["SemanticMatch"], experiment_id)

        # Load distillation dataset
        distillation_dataset = get_dataset(dataset_id=distillation_dataset_id)

        # Run optimization
        result = optimize_prompts(
            predict_fn=student_predict_fn,
            train_data=distillation_dataset,
            prompt_uris=[student_prompt_uri],
            optimizer=optimizer,
            scorers=scorers,
            enable_tracking=True,
        )

        # Log completion
        mlflow.set_tag("distillation.status", "completed")

    # Build result
    optimized_uri = result.optimized_prompts[0].uri if result.optimized_prompts else None

    job_result = DistillationJobResult(
        run_id=run_id,
        student_prompt_uri=student_prompt_uri,
        optimized_prompt_uri=optimized_uri,
        distillation_dataset_id=distillation_dataset_id,
        num_traces=len(traces),
        num_samples=len(extracted_data),
        optimizer_name=result.optimizer_name,
        initial_eval_score=result.initial_eval_score,
        final_eval_score=result.final_eval_score,
    )

    _logger.info(
        f"Distillation completed. Optimized prompt: {optimized_uri}, "
        f"Final score: {result.final_eval_score}"
    )

    return asdict(job_result)


@job(name="distill_prompts", max_workers=_DEFAULT_DISTILLATION_JOB_MAX_WORKERS)
def distill_prompts_job(
    run_id: str,
    experiment_id: str,
    student_prompt_uri: str,
    source_prompt_uri: str,
    optimizer_type: str,
    optimizer_config: dict[str, Any] | None,
    max_traces: int | None = None,
) -> dict[str, Any]:
    """
    Async job function for prompt distillation.

    This job implements knowledge distillation to optimize a student prompt
    to match responses from traces linked to a source prompt (typically a teacher).

    **Prerequisites:**
    1. Run your agent/prompt over your dataset with MLflow tracing enabled
    2. Ensure your code uses mlflow.genai.load_prompt() to link traces to the prompt

    **Workflow:**
    1. Searches for traces linked to the source prompt
    2. Extracts responses from traces using two-stage matching
    3. Creates a distillation dataset with responses as expected outputs
    4. Optimizes the student prompt using SemanticMatch scorer

    Args:
        run_id: MLflow run ID for tracking.
        experiment_id: Experiment ID for tracking.
        student_prompt_uri: URI of the student prompt to optimize.
        source_prompt_uri: URI of the prompt whose traces to use as training data.
        optimizer_type: Optimizer type ("gepa" or "metaprompt").
        optimizer_config: Optimizer-specific configuration dict.
        max_traces: Maximum number of traces to use. None for all available.

    Returns:
        Dict containing distillation results (JSON-serializable).
    """
    return run_distillation(
        run_id=run_id,
        experiment_id=experiment_id,
        student_prompt_uri=student_prompt_uri,
        source_prompt_uri=source_prompt_uri,
        optimizer_type=optimizer_type,
        optimizer_config=optimizer_config,
        max_traces=max_traces,
    )
