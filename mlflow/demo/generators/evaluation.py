from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from typing import Literal

import pandas as pd

import mlflow
from mlflow.demo.base import (
    DEMO_EXPERIMENT_NAME,
    BaseDemoGenerator,
    DemoFeature,
    DemoResult,
)
from mlflow.demo.data import EXPECTED_ANSWERS
from mlflow.demo.generators.traces import DEMO_VERSION_TAG, TracesDemoGenerator
from mlflow.entities.assessment import AssessmentSource, Feedback
from mlflow.entities.trace import Trace
from mlflow.genai.datasets import create_dataset, delete_dataset, search_datasets
from mlflow.genai.scorers import scorer

_logger = logging.getLogger(__name__)

DEMO_DATASET_V1_NAME = "demo-baseline-dataset"
DEMO_DATASET_V2_NAME = "demo-improved-dataset"


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


def _create_pass_fail_scorer(
    name: str,
    pass_rate: float,
    rationale_fn: Callable[[bool], str],
):
    """Create a deterministic scorer that returns 'yes' or 'no' Feedback.

    These scorers simulate LLM judges but use deterministic pass/fail decisions
    based on content hashes, making demo data reproducible.
    """

    @scorer(name=name)
    def pass_fail_scorer(inputs, outputs) -> Feedback:
        content = str(inputs) + str(outputs)
        hash_input = f"{content}:{name}"
        hash_val = int(hashlib.md5(hash_input.encode(), usedforsecurity=False).hexdigest()[:8], 16)
        normalized = hash_val / 0xFFFFFFFF
        is_passing = normalized < pass_rate
        return Feedback(
            value="yes" if is_passing else "no",
            rationale=rationale_fn(is_passing),
            source=AssessmentSource(
                source_type="LLM_JUDGE",
                source_id=f"demo-judge/{name}",
            ),
        )

    return pass_fail_scorer


BASELINE_PROFILE = {
    "name": "baseline-evaluation",
    "pass_rates": {
        "relevance": 0.65,
        "correctness": 0.60,
        "groundedness": 0.55,
        "safety": 0.95,
    },
}

IMPROVED_PROFILE = {
    "name": "improved-evaluation",
    "pass_rates": {
        "relevance": 0.90,
        "correctness": 0.85,
        "groundedness": 0.85,
        "safety": 1.0,
    },
}


class EvaluationDemoGenerator(BaseDemoGenerator):
    """Generates demo evaluation data comparing baseline (v1) and improved (v2) traces.

    Creates:
    - Ground truth expectations on all demo traces
    - Two datasets: baseline (v1) and improved (v2) - both include all trace types
    - Two evaluation runs comparing the same inputs with different outputs

    The baseline and improved evaluations include matching trace types:
    - 2 RAG traces
    - 2 agent traces
    - 6 prompt traces (2 per prompt type)
    - Session traces

    This allows direct comparison between v1 and v2 performance.
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

        v1_traces = self._fetch_demo_traces(experiment_id, "v1")
        v2_traces = self._fetch_demo_traces(experiment_id, "v2")

        self._add_expectations_to_traces(v1_traces)
        self._add_expectations_to_traces(v2_traces)

        v1_traces_with_expectations = self._fetch_demo_traces(experiment_id, "v1")
        v2_traces_with_expectations = self._fetch_demo_traces(experiment_id, "v2")

        self._create_evaluation_dataset(
            v1_traces_with_expectations, experiment_id, DEMO_DATASET_V1_NAME
        )
        self._create_evaluation_dataset(
            v2_traces_with_expectations, experiment_id, DEMO_DATASET_V2_NAME
        )

        v1_run_id = self._create_single_evaluation_run(
            traces=v1_traces_with_expectations,
            experiment_id=experiment_id,
            run_name=BASELINE_PROFILE["name"],
            pass_rates=BASELINE_PROFILE["pass_rates"],
        )

        v2_run_id = self._create_single_evaluation_run(
            traces=v2_traces_with_expectations,
            experiment_id=experiment_id,
            run_name=IMPROVED_PROFILE["name"],
            pass_rates=IMPROVED_PROFILE["pass_rates"],
        )

        return DemoResult(
            feature=self.name,
            entity_ids=[v1_run_id, v2_run_id],
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
                max_results=100,
            )
            for run in runs:
                try:
                    client.delete_run(run.info.run_id)
                except Exception:
                    _logger.debug("Failed to delete run %s", run.info.run_id, exc_info=True)
        except Exception:
            _logger.debug("Failed to delete evaluation demo runs", exc_info=True)

        self._delete_demo_dataset(experiment.experiment_id, DEMO_DATASET_V1_NAME)
        self._delete_demo_dataset(experiment.experiment_id, DEMO_DATASET_V2_NAME)

    def _fetch_demo_traces(self, experiment_id: str, version: Literal["v1", "v2"]) -> list[Trace]:
        return mlflow.search_traces(
            locations=[experiment_id],
            filter_string=f"metadata.`{DEMO_VERSION_TAG}` = '{version}'",
            max_results=100,
            return_type="list",
        )

    def _add_expectations_to_traces(self, traces: list[Trace]) -> int:
        expectation_count = 0

        for trace in traces:
            trace_id = trace.info.trace_id

            root_span = next((span for span in trace.data.spans if span.parent_id is None), None)
            if root_span is None:
                continue

            inputs = root_span.inputs or {}
            query = inputs.get("query") or inputs.get("message")

            if expected_answer := self._find_expected_answer(query):
                try:
                    mlflow.log_expectation(
                        trace_id=trace_id,
                        name="expected_response",
                        value=expected_answer,
                        source=AssessmentSource(
                            source_type="HUMAN",
                            source_id="demo_annotator",
                        ),
                        metadata={"demo": "true"},
                    )
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
    ) -> int:
        dataset = create_dataset(
            name=dataset_name,
            experiment_id=experiment_id,
            tags={"demo": "true", "description": f"Demo evaluation dataset: {dataset_name}"},
        )

        dataset.merge_records(traces)

        return len(traces)

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

    def _create_single_evaluation_run(
        self,
        traces: list[Trace],
        experiment_id: str,
        run_name: str,
        pass_rates: dict[str, float],
    ) -> str:
        trace_df = pd.DataFrame({"trace": traces})

        demo_scorers = [
            _create_pass_fail_scorer(
                name="relevance",
                pass_rate=pass_rates["relevance"],
                rationale_fn=_get_relevance_rationale,
            ),
            _create_pass_fail_scorer(
                name="correctness",
                pass_rate=pass_rates["correctness"],
                rationale_fn=_get_correctness_rationale,
            ),
            _create_pass_fail_scorer(
                name="groundedness",
                pass_rate=pass_rates["groundedness"],
                rationale_fn=_get_groundedness_rationale,
            ),
            _create_pass_fail_scorer(
                name="safety",
                pass_rate=pass_rates["safety"],
                rationale_fn=_get_safety_rationale,
            ),
        ]

        mlflow.set_experiment(experiment_id=experiment_id)

        result = mlflow.genai.evaluate(
            data=trace_df,
            scorers=demo_scorers,
        )

        client = mlflow.MlflowClient()
        client.set_tag(result.run_id, "mlflow.runName", run_name)
        client.log_param(result.run_id, "demo", "true")

        return result.run_id
