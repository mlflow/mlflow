from __future__ import annotations

import logging
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.demo.base import (
    DEMO_EXPERIMENT_NAME,
    BaseDemoGenerator,
    DemoFeature,
    DemoResult,
)
from mlflow.demo.data import ASSESSMENT_TO_ISSUE, ROOT_CAUSE_EXPLANATIONS
from mlflow.demo.generators.traces import DEMO_VERSION_TAG
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.issue import IssueStatus
from mlflow.store.tracking import MAX_TRACE_LINKS_PER_REQUEST
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.utils.mlflow_tags import MLFLOW_RUN_TYPE, MLFLOW_RUN_TYPE_ISSUE_DETECTION

_logger = logging.getLogger(__name__)

DEMO_ISSUE_TAG = "mlflow.demo.issue"
DEMO_ISSUE_DETECTION_RUN_NAME = "Demo Issue Detection"


class IssuesDemoGenerator(BaseDemoGenerator):
    """Generates demo issues showing the issue detection and management features.

    Creates issues based on actual failing assessments from evaluation runs.
    Issues are automatically linked to traces that failed specific quality checks
    (relevance, correctness, groundedness, safety), making the issue-trace
    relationship authentic and meaningful.
    """

    name = DemoFeature.ISSUES
    version = 3

    def generate(self) -> DemoResult:
        store = _get_store()
        experiment = store.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
        if experiment is None:
            raise ValueError(f"Demo experiment '{DEMO_EXPERIMENT_NAME}' not found")

        experiment_id = experiment.experiment_id
        traces_df = mlflow.search_traces(locations=[experiment_id], max_results=1000)
        failing_traces_by_assessment = {}

        for _, row in traces_df.iterrows():
            trace_id = row.get("trace_id")
            metadata = row.get("trace_metadata", {}) or {}
            version = metadata.get(DEMO_VERSION_TAG)

            if version != "v1":
                continue

            assessments = row.get("assessments", []) or []
            for assessment in assessments:
                if not isinstance(assessment, dict):
                    continue

                feedback_data = assessment.get("feedback")
                if not feedback_data or feedback_data.get("value") != "no":
                    continue

                source_id = assessment.get("source", {}).get("source_id", "")
                if "/" in source_id:
                    assessment_name = source_id.split("/")[-1]
                    if assessment_name in ASSESSMENT_TO_ISSUE:
                        rationale = assessment.get("rationale", "Assessment failed")
                        failing_traces_by_assessment.setdefault(assessment_name, []).append({
                            "trace_id": trace_id,
                            "rationale": rationale,
                        })

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=DEMO_ISSUE_DETECTION_RUN_NAME,
            tags={MLFLOW_RUN_TYPE: MLFLOW_RUN_TYPE_ISSUE_DETECTION},
        ) as run:
            run_id = run.info.run_id
            created_issue_ids = []
            all_linked_trace_ids = set()
            source = AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=run_id,
            )

            created_issues_info = []
            for assessment_name, failing_traces in failing_traces_by_assessment.items():
                if not failing_traces:
                    continue

                issue_config = ASSESSMENT_TO_ISSUE[assessment_name]
                issue = store.create_issue(
                    experiment_id=experiment_id,
                    name=issue_config["name"],
                    description=issue_config["description"],
                    status=IssueStatus.PENDING,
                    severity=issue_config["severity"],
                    root_causes=issue_config["root_causes"],
                    categories=issue_config["categories"],
                    created_by="demo",
                    source_run_id=run_id,
                )
                created_issue_ids.append(issue.issue_id)
                created_issues_info.append({
                    "name": issue_config["name"],
                    "description": issue_config["description"],
                    "severity": issue_config["severity"],
                    "root_causes": issue_config["root_causes"],
                    "categories": issue_config["categories"],
                })

                # Limit to 5 traces per issue for performance
                for trace_info in failing_traces[:5]:
                    mlflow.log_issue(
                        trace_id=trace_info["trace_id"],
                        issue_id=issue.issue_id,
                        issue_name=issue.name,
                        source=source,
                        run_id=run_id,
                        rationale=trace_info["rationale"],
                    )
                    all_linked_trace_ids.add(trace_info["trace_id"])

            if all_linked_trace_ids:
                client = MlflowClient()
                trace_ids_list = list(all_linked_trace_ids)
                for i in range(0, len(trace_ids_list), MAX_TRACE_LINKS_PER_REQUEST):
                    batch = trace_ids_list[i : i + MAX_TRACE_LINKS_PER_REQUEST]
                    client.link_traces_to_run(batch, run_id)

            v1_traces = [
                row
                for _, row in traces_df.iterrows()
                if ((row.get("trace_metadata") or {}).get(DEMO_VERSION_TAG) == "v1")
            ]
            summary = self._generate_issue_summary(
                total_traces_analyzed=len(v1_traces),
                created_issues=created_issues_info,
            )

            # Store result as tags so UI can display without requiring a job
            mlflow.set_tag("mlflow.issueDetection.result.issues", str(len(created_issue_ids)))
            mlflow.set_tag("mlflow.issueDetection.result.totalTracesAnalyzed", str(len(v1_traces)))
            mlflow.set_tag("mlflow.issueDetection.result.summary", summary)

        return DemoResult(
            feature=self.name,
            entity_ids=created_issue_ids,
            navigation_url=f"#/experiments/{experiment_id}/issues",
        )

    def _generate_issue_summary(
        self, total_traces_analyzed: int, created_issues: list[dict[str, Any]]
    ) -> str:
        """Generate a markdown summary of detected issues."""
        if not created_issues:
            return f"Analyzed {total_traces_analyzed} traces. No issues found."

        issue_count = len(created_issues)
        issue_plural = "issue" if issue_count == 1 else "issues"
        summary_lines = [
            f"Analyzed {total_traces_analyzed} traces. Found {issue_count} {issue_plural}:"
        ]

        for idx, issue_info in enumerate(created_issues, start=1):
            severity_str = issue_info["severity"].value.lower()
            summary_lines.append("")
            summary_lines.append(f"## {idx}. {issue_info['name']} (severity: {severity_str})")
            summary_lines.append("")
            summary_lines.append(issue_info["description"])
            summary_lines.append("")
            summary_lines.append("**Root causes:**")
            for root_cause in issue_info["root_causes"]:
                explanation = ROOT_CAUSE_EXPLANATIONS.get(
                    root_cause, root_cause.replace("_", " ").title()
                )
                summary_lines.append(f"- {explanation}")

            summary_lines.append("")
            summary_lines.append(f"**Categories:** {', '.join(issue_info['categories'])}")

        return "\n".join(summary_lines)

    def _data_exists(self) -> bool:
        try:
            store = _get_store()
            experiment = store.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
            if experiment is None:
                return False

            issues = store.search_issues(
                experiment_id=experiment.experiment_id,
                max_results=1000,
            )
            return any(i.created_by == "demo" for i in issues)
        except Exception:
            return False

    def delete_demo(self) -> None:
        store = _get_store()
        experiment = store.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
        if experiment is None:
            return

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.`{MLFLOW_RUN_TYPE}` = '{MLFLOW_RUN_TYPE_ISSUE_DETECTION}'",
            max_results=100,
        )
        for _, run in runs.iterrows():
            mlflow.delete_run(run.run_id)

        # TODO: Delete issues explicitly once delete_issue API is available
        # Note: Issues are also automatically deleted when the experiment is deleted.
