from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import mlflow

if TYPE_CHECKING:
    from mlflow.genai.datasets import EvaluationDataset

from mlflow.demo.base import (
    DEMO_EXPERIMENT_NAME,
    BaseDemoGenerator,
    DemoFeature,
    DemoResult,
)
from mlflow.demo.data import EXPECTED_ANSWERS
from mlflow.demo.generators.traces import DEMO_TRACE_TYPE_TAG, DEMO_VERSION_TAG, TracesDemoGenerator
from mlflow.entities.assessment import AssessmentSource, Expectation, Feedback
from mlflow.entities.trace import Trace
from mlflow.entities.view_type import ViewType
from mlflow.genai.datasets import create_dataset, delete_dataset, search_datasets
from mlflow.genai.scorers import scorer

_logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_evaluation_output():
    """Suppress tqdm progress bars and evaluation completion messages."""
    original_tqdm_disable = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"
    try:
        # Suppress both stdout (evaluation messages) and stderr (tqdm progress bars)
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            yield
    finally:
        if original_tqdm_disable is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = original_tqdm_disable


DEMO_DATASET_TRACE_LEVEL_NAME = "demo-trace-level-dataset"
DEMO_DATASET_BASELINE_SESSION_NAME = "demo-baseline-session-dataset"
DEMO_DATASET_IMPROVED_SESSION_NAME = "demo-improved-session-dataset"


def _get_relevance_rationale(is_relevant: bool) -> str:
    if is_relevant:
        return "The response directly addresses the question with relevant information."
    return "The response is not sufficiently relevant to the question asked."


def _get_correctness_rationale(is_correct: bool) -> str:
    if is_correct:
        return "The response accurately captures the key information from the expected answer."
    return (
        "The response contains relevant information but differs "
        "significantly from the expected answer."
    )


def _get_groundedness_rationale(is_grounded: bool) -> str:
    if is_grounded:
        return "The response is well-grounded in the provided context with clear references."
    return "The response includes claims not supported by the provided context."


def _get_safety_rationale(is_safe: bool) -> str:
    if is_safe:
        return "The response contains no harmful, offensive, or inappropriate content."
    return "The response may contain potentially harmful or inappropriate content."


def _create_quality_aware_scorer(
    name: str,
    baseline_pass_rate: float,
    improved_pass_rate: float,
    rationale_fn: Callable[[bool], str],
):
    """Create a deterministic scorer that simulates quality-aware evaluation.

    The scorer detects response quality based on content characteristics:
    - Longer, more detailed responses get evaluated with higher pass rates
    - Shorter, less detailed responses get evaluated with lower pass rates

    This simulates the real-world scenario where improved model outputs
    naturally score better when evaluated by the same scorers.
    """
    quality_threshold = 400

    @scorer(name=name)
    def quality_aware_scorer(inputs, outputs, trace) -> Feedback:
        content = str(inputs) + str(outputs)
        output_str = str(outputs)

        if len(output_str) > quality_threshold:
            effective_pass_rate = improved_pass_rate
        else:
            effective_pass_rate = baseline_pass_rate

        # Use content hash for deterministic but varied results
        hash_input = f"{content}:{name}"
        hash_val = int(hashlib.md5(hash_input.encode(), usedforsecurity=False).hexdigest()[:8], 16)
        normalized = hash_val / 0xFFFFFFFF
        is_passing = normalized < effective_pass_rate

        # Use the trace timestamp so the quality overview chart shows a trend
        # across days instead of a single dot at the current time.
        trace_timestamp_ms = trace.info.timestamp_ms if trace else None

        return Feedback(
            value="yes" if is_passing else "no",
            rationale=rationale_fn(is_passing),
            source=AssessmentSource(
                source_type="LLM_JUDGE",
                source_id=f"judges/{name}",
            ),
            create_time_ms=trace_timestamp_ms,
            last_update_time_ms=trace_timestamp_ms,
        )

    return quality_aware_scorer


SCORER_PASS_RATES = {
    "relevance": {"baseline": 0.65, "improved": 0.92},
    "correctness": {"baseline": 0.58, "improved": 0.88},
    "groundedness": {"baseline": 0.52, "improved": 0.85},
    "safety": {"baseline": 0.95, "improved": 1.0},
}


class EvaluationDemoGenerator(BaseDemoGenerator):
    """Generates demo evaluation data.

    Creates:
    - Ground truth expectations on all demo traces
    - Three datasets and evaluation runs, each in a single mode:
      - trace-level-evaluation: non-session traces (v1 + v2 combined)
      - baseline-session-evaluation: v1 session traces
      - improved-session-evaluation: v2 session traces

    Assessment timestamps are spread to match trace timestamps so the
    quality overview chart shows a trend across days.
    """

    name = DemoFeature.EVALUATION
    version = 1

    def generate(self) -> DemoResult:
        traces_generator = TracesDemoGenerator()
        if not traces_generator.is_generated():
            traces_generator.generate()
            traces_generator.store_version()

        experiment = mlflow.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
        experiment_id = experiment.experiment_id

        # Fetch traces split by session vs non-session
        v1_non_session = self._fetch_demo_traces(experiment_id, "v1", session=False)
        v2_non_session = self._fetch_demo_traces(experiment_id, "v2", session=False)
        v1_session = self._fetch_demo_traces(experiment_id, "v1", session=True)
        v2_session = self._fetch_demo_traces(experiment_id, "v2", session=True)

        all_traces = v1_non_session + v2_non_session + v1_session + v2_session
        self._add_expectations_to_traces(all_traces)

        # Re-fetch to include expectations
        v1_non_session = self._fetch_demo_traces(experiment_id, "v1", session=False)
        v2_non_session = self._fetch_demo_traces(experiment_id, "v2", session=False)
        v1_session = self._fetch_demo_traces(experiment_id, "v1", session=True)
        v2_session = self._fetch_demo_traces(experiment_id, "v2", session=True)

        trace_level_traces = v1_non_session + v2_non_session

        # Create datasets
        self._create_evaluation_dataset(
            trace_level_traces, experiment_id, DEMO_DATASET_TRACE_LEVEL_NAME
        )
        self._create_evaluation_dataset(
            v1_session, experiment_id, DEMO_DATASET_BASELINE_SESSION_NAME
        )
        self._create_evaluation_dataset(
            v2_session, experiment_id, DEMO_DATASET_IMPROVED_SESSION_NAME
        )

        # Create evaluation runs
        trace_level_run_id = self._create_evaluation_run(
            traces=trace_level_traces,
            experiment_id=experiment_id,
            run_name="trace-level-evaluation",
        )

        baseline_session_run_id = self._create_evaluation_run(
            traces=v1_session,
            experiment_id=experiment_id,
            run_name="baseline-session-evaluation",
        )

        improved_session_run_id = self._create_evaluation_run(
            traces=v2_session,
            experiment_id=experiment_id,
            run_name="improved-session-evaluation",
        )

        return DemoResult(
            feature=self.name,
            entity_ids=[trace_level_run_id, baseline_session_run_id, improved_session_run_id],
            navigation_url=f"#/experiments/{experiment_id}/evaluation-runs",
        )

    def _data_exists(self) -> bool:
        experiment = mlflow.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
        if experiment is None or experiment.lifecycle_stage != "active":
            return False

        try:
            client = mlflow.MlflowClient()
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="params.demo = 'true'",
                max_results=1,
            )
            return len(runs) > 0
        except Exception:
            _logger.debug("Failed to check if evaluation demo exists", exc_info=True)
            return False

    def delete_demo(self) -> None:
        experiment = mlflow.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
        if experiment is None:
            return

        try:
            client = mlflow.MlflowClient()
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="params.demo = 'true'",
                run_view_type=ViewType.ALL,
                max_results=100,
            )
            for run in runs:
                try:
                    if run.info.lifecycle_stage == "deleted":
                        client.restore_run(run.info.run_id)
                    client.delete_run(run.info.run_id)
                except Exception:
                    _logger.debug("Failed to delete run %s", run.info.run_id, exc_info=True)
        except Exception:
            _logger.debug("Failed to delete evaluation demo runs", exc_info=True)

        for name in [
            DEMO_DATASET_TRACE_LEVEL_NAME,
            DEMO_DATASET_BASELINE_SESSION_NAME,
            DEMO_DATASET_IMPROVED_SESSION_NAME,
        ]:
            self._delete_demo_dataset(experiment.experiment_id, name)

    def _fetch_demo_traces(
        self,
        experiment_id: str,
        version: Literal["v1", "v2"],
        session: bool | None = None,
    ) -> list[Trace]:
        filter_parts = [f"metadata.`{DEMO_VERSION_TAG}` = '{version}'"]
        operator = "=" if session else "!="
        filter_parts.append(f"metadata.`{DEMO_TRACE_TYPE_TAG}` {operator} 'session'")
        return mlflow.search_traces(
            locations=[experiment_id],
            filter_string=" AND ".join(filter_parts),
            max_results=100,
            return_type="list",
        )

    def _add_expectations_to_traces(self, traces: list[Trace]) -> int:
        expectation_count = 0

        for trace in traces:
            trace_id = trace.info.trace_id
            trace_timestamp_ms = trace.info.timestamp_ms

            root_span = next((span for span in trace.data.spans if span.parent_id is None), None)
            if root_span is None:
                continue

            inputs = root_span.inputs or {}
            query = inputs.get("query") or inputs.get("message")

            if expected_answer := self._find_expected_answer(query):
                try:
                    expectation = Expectation(
                        name="expected_response",
                        value=expected_answer,
                        source=AssessmentSource(
                            source_type="HUMAN",
                            source_id="demo_annotator",
                        ),
                        metadata={"demo": "true"},
                        trace_id=trace_id,
                        create_time_ms=trace_timestamp_ms,
                        last_update_time_ms=trace_timestamp_ms,
                    )
                    mlflow.log_assessment(trace_id=trace_id, assessment=expectation)
                    expectation_count += 1
                except Exception:
                    _logger.debug("Failed to log expectation for trace %s", trace_id, exc_info=True)

        return expectation_count

    def _find_expected_answer(self, query: str) -> str | None:
        query_lower = query.lower().strip()
        if query_lower in EXPECTED_ANSWERS:
            return EXPECTED_ANSWERS[query_lower]
        for q, answer in EXPECTED_ANSWERS.items():
            if q in query_lower or query_lower in q:
                return answer
        return None

    def _create_evaluation_dataset(
        self, traces: list[Trace], experiment_id: str, dataset_name: str
    ) -> "EvaluationDataset":
        from mlflow.genai.datasets import get_dataset

        dataset = create_dataset(
            name=dataset_name,
            experiment_id=experiment_id,
            tags={"demo": "true", "description": f"Demo evaluation dataset: {dataset_name}"},
        )

        dataset.merge_records(traces)
        return get_dataset(dataset_id=dataset.dataset_id)

    def _delete_demo_dataset(self, experiment_id: str, dataset_name: str) -> None:
        datasets = search_datasets(
            experiment_ids=[experiment_id],
            filter_string=f"name = '{dataset_name}'",
            max_results=10,
        )
        for ds in datasets:
            try:
                delete_dataset(dataset_id=ds.dataset_id)
            except Exception:
                _logger.debug("Failed to delete dataset %s", ds.dataset_id, exc_info=True)

    def _create_evaluation_run(
        self,
        traces: list[Trace],
        experiment_id: str,
        run_name: str,
    ) -> str:
        demo_scorers = [
            _create_quality_aware_scorer(
                name="relevance",
                baseline_pass_rate=SCORER_PASS_RATES["relevance"]["baseline"],
                improved_pass_rate=SCORER_PASS_RATES["relevance"]["improved"],
                rationale_fn=_get_relevance_rationale,
            ),
            _create_quality_aware_scorer(
                name="correctness",
                baseline_pass_rate=SCORER_PASS_RATES["correctness"]["baseline"],
                improved_pass_rate=SCORER_PASS_RATES["correctness"]["improved"],
                rationale_fn=_get_correctness_rationale,
            ),
            _create_quality_aware_scorer(
                name="groundedness",
                baseline_pass_rate=SCORER_PASS_RATES["groundedness"]["baseline"],
                improved_pass_rate=SCORER_PASS_RATES["groundedness"]["improved"],
                rationale_fn=_get_groundedness_rationale,
            ),
            _create_quality_aware_scorer(
                name="safety",
                baseline_pass_rate=SCORER_PASS_RATES["safety"]["baseline"],
                improved_pass_rate=SCORER_PASS_RATES["safety"]["improved"],
                rationale_fn=_get_safety_rationale,
            ),
        ]

        mlflow.set_experiment(experiment_id=experiment_id)

        with _suppress_evaluation_output():
            result = mlflow.genai.evaluate(
                data=traces,
                scorers=demo_scorers,
            )

        client = mlflow.MlflowClient()
        client.set_tag(result.run_id, "mlflow.runName", run_name)
        client.log_param(result.run_id, "demo", "true")

        return result.run_id
