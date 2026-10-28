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
from mlflow.demo.data import (
    ASSESSMENT_TO_ISSUE,
    ISSUE_EVIDENCE_WALKTHROUGHS,
    ROOT_CAUSE_EXPLANATIONS,
)
from mlflow.demo.generators.traces import DEMO_VERSION_TAG
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.issue import IssueStatus
from mlflow.entities.issue_evidence import IssueEvidence, IssueEvidenceStep
from mlflow.entities.run_status import RunStatus
from mlflow.genai.discovery.schedule import (
    DetectionCadence,
    DetectionSchedule,
    write_detection_schedule,
)
from mlflow.store.tracking import MAX_TRACE_LINKS_PER_REQUEST
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.utils.mlflow_tags import (
    MLFLOW_ISSUE_DETECTION_ERROR,
    MLFLOW_ISSUE_DETECTION_SCHEDULE,
    MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER,
    MLFLOW_ISSUE_DETECTION_TRIGGER,
    MLFLOW_ISSUE_DETECTION_TRIGGER_SCHEDULE,
    MLFLOW_ISSUE_DETECTION_WINDOW_END,
    MLFLOW_ISSUE_DETECTION_WINDOW_START,
    MLFLOW_RUN_TYPE,
    MLFLOW_RUN_TYPE_ISSUE_DETECTION,
)
from mlflow.utils.time import get_current_time_millis

_logger = logging.getLogger(__name__)

_DEMO_CREATED_BY = "demo"

DEMO_ISSUE_DETECTION_RUN_NAME = "Demo Issue Detection"
_MAX_TRACES_PER_ISSUE = 5


def _span_type(span) -> str:
    """A span's type, upper-cased, or empty when it carries none."""
    try:
        return (span.span_type or "").upper()
    except Exception:
        return ""


def _pick_span(spans, preferred_types: list[str], used_ids: set[str], last: bool = False):
    """
    The span a walkthrough step should cite, or ``None`` for a trace with no spans.

    Preferred types first, then a span no earlier step cited, then the root span. ``last`` takes
    the newest span of the type: a ReAct trace's first LLM span is the tool-call decision
    (``content: None``) and only its last one holds the answer.
    """
    if not spans:
        return None
    order = list(reversed(spans)) if last else list(spans)
    for wanted in preferred_types:
        matching = [span for span in order if _span_type(span) == wanted]
        unused = [span for span in matching if span.span_id not in used_ids]
        if chosen := (unused or matching):
            return chosen[0]
    unused = [span for span in order if span.span_id not in used_ids]
    return (unused or order)[0]


def _build_evidence(assessment_name: str, rationale: str, spans) -> IssueEvidence | None:
    """
    The walkthrough for one trace's issue reference, or ``None`` to leave it unattached.

    The summary is the judge's own rationale, so the card still says something specific to the
    trace. ``None`` rather than raising when there is nothing to build from: the plain rationale
    is still worth linking.
    """
    steps_config = ISSUE_EVIDENCE_WALKTHROUGHS.get(assessment_name)
    if not steps_config or not spans:
        return None
    steps: list[IssueEvidenceStep] = []
    used_ids: set[str] = set()
    for step_config in steps_config:
        span = _pick_span(
            spans, step_config["span_types"], used_ids, last=step_config.get("last", False)
        )
        if span is None:
            continue
        used_ids.add(span.span_id)
        steps.append(
            IssueEvidenceStep(
                span_id=span.span_id,
                text=step_config["text"],
                highlights=list(step_config["highlights"]),
                highlight_tone=step_config["highlight_tone"],
            )
        )
    if not steps:
        return None
    # The last step names the failure itself, so its span is the one the finding belongs to.
    return IssueEvidence(summary=rationale, steps=steps, primary_span_id=steps[-1].span_id)


_DAY_MS = 24 * 60 * 60 * 1000

# The model every demo detection run reports having analyzed with.
_DEMO_DETECTION_MODEL = "openai:/gpt-4o-mini"

# The runs behind the one that filed the demo's issues, so the run history page has more than a
# single row. Between them: both triggers, three terminal statuses, a run that analyzed nothing,
# and a failure with its reason. None is left RUNNING -- the page polls while one is unsettled.
_DEMO_DETECTION_HISTORY = (
    {
        # Neither the quiet occurrence nor the failure below advanced the watermark, so this run
        # was entitled to 21 days and `STALE_WATERMARK_CAP_MS` clipped it to 8.
        "name": "Scheduled detection",
        "days_ago": 2,
        "duration_ms": 192_000,
        "status": RunStatus.FINISHED,
        "scheduled": True,
        "total_traces": 180,
        "cost_usd": "0.41",
        "window_days": 8,
    },
    {
        "name": "Scheduled detection",
        "days_ago": 9,
        "duration_ms": 61_000,
        "status": RunStatus.FAILED,
        "scheduled": True,
        "total_traces": 164,
        "cost_usd": None,
        "window_days": 7,
        "error": (
            "AuthenticationError: No API key available for provider 'openai'. Save a key in "
            "AI Gateway, then edit the schedule to use it."
        ),
    },
    {
        "name": "Issue detection",
        "days_ago": 12,
        "duration_ms": 22_000,
        "status": RunStatus.KILLED,
        "scheduled": False,
        "total_traces": 240,
        "cost_usd": None,
    },
    {
        # A window with nothing in it: finished over zero traces, not a failure.
        "name": "Scheduled detection",
        "days_ago": 16,
        "duration_ms": 4_000,
        "status": RunStatus.FINISHED,
        "scheduled": True,
        "total_traces": 0,
        "cost_usd": "0.00",
        "window_days": 7,
    },
)


class IssuesDemoGenerator(BaseDemoGenerator):
    """Generates demo issues showing the issue detection and management features.

    Creates issues based on actual failing assessments from evaluation runs.
    Issues are automatically linked to traces that failed specific quality checks
    (relevance, correctness, groundedness, safety), making the issue-trace
    relationship authentic and meaningful.
    """

    name = DemoFeature.ISSUES
    version = 5

    def generate(self) -> DemoResult:
        store = _get_store()
        experiment = store.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
        if experiment is None:
            raise ValueError(f"Demo experiment '{DEMO_EXPERIMENT_NAME}' not found")

        experiment_id = experiment.experiment_id
        traces = mlflow.search_traces(
            locations=[experiment_id], max_results=1000, return_type="list", flush=True
        )
        failing_traces_by_assessment = {}

        for trace in traces:
            trace_id = trace.info.trace_id
            metadata = trace.info.trace_metadata or {}
            version = metadata.get(DEMO_VERSION_TAG)

            if version != "v1":
                continue

            assessments = trace.info.assessments or []
            for assessment in assessments:
                feedback_data = assessment.feedback
                if not feedback_data or feedback_data.value != "no":
                    continue

                source_id = assessment.source.source_id if assessment.source else ""
                if "/" in source_id:
                    assessment_name = source_id.split("/")[-1]
                    if assessment_name in ASSESSMENT_TO_ISSUE:
                        rationale = assessment.rationale or "Assessment failed"
                        failing_traces_by_assessment.setdefault(assessment_name, []).append({
                            "trace_id": trace_id,
                            "rationale": rationale,
                            # Already returned by the search, so the walkthrough costs no reads.
                            "spans": (trace.data.spans if trace.data else None) or [],
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
                    created_by=_DEMO_CREATED_BY,
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

                for trace_info in failing_traces[:_MAX_TRACES_PER_ISSUE]:
                    mlflow.log_issue(
                        trace_id=trace_info["trace_id"],
                        issue_id=issue.issue_id,
                        issue_name=issue.name,
                        source=source,
                        run_id=run_id,
                        rationale=trace_info["rationale"],
                        # Pins the finding to the spans it happened in. `None` falls back to the
                        # bare rationale, as before.
                        evidence=_build_evidence(
                            assessment_name, trace_info["rationale"], trace_info["spans"]
                        ),
                    )
                    all_linked_trace_ids.add(trace_info["trace_id"])

            if all_linked_trace_ids:
                client = MlflowClient()
                trace_ids_list = list(all_linked_trace_ids)
                for i in range(0, len(trace_ids_list), MAX_TRACE_LINKS_PER_REQUEST):
                    batch = trace_ids_list[i : i + MAX_TRACE_LINKS_PER_REQUEST]
                    client.link_traces_to_run(batch, run_id)

            v1_traces = [
                trace
                for trace in traces
                if ((trace.info.trace_metadata or {}).get(DEMO_VERSION_TAG) == "v1")
            ]
            summary = self._generate_issue_summary(
                total_traces_analyzed=len(v1_traces),
                created_issues=created_issues_info,
            )

            # Store result as tags so UI can display without requiring a job
            mlflow.set_tags({
                "mlflow.issueDetection.result.issues": str(len(created_issue_ids)),
                "mlflow.issueDetection.result.totalTracesAnalyzed": str(len(v1_traces)),
                "mlflow.issueDetection.result.summary": summary,
                # The key a real detection run writes, which is what the run history page reads.
                "total_traces": str(len(v1_traces)),
                "model": _DEMO_DETECTION_MODEL,
            })

        # After the issue-filing run, so the history lists that one newest and these behind it.
        self._generate_detection_history(MlflowClient(), experiment_id, get_current_time_millis())
        self._write_paused_schedule(store, experiment_id)

        return DemoResult(
            feature=self.name,
            entity_ids=created_issue_ids,
            navigation_url=f"#/experiments/{experiment_id}/issues",
        )

    def _write_paused_schedule(self, store, experiment_id: str) -> None:
        """
        Store the schedule the demo's scheduled runs came from, turned off.

        Needed for the covered windows: the page formats those bounds in the zone the schedule
        stores, so without a schedule there is no zone and every window renders as a dash.
        Paused because a live one would be dispatched, firing real runs against a provider the
        demo has no key for.
        """
        try:
            write_detection_schedule(
                store,
                experiment_id,
                DetectionSchedule(cadence=DetectionCadence.WEEKLY, paused=True, fire_time="09:00"),
            )
        except Exception:
            _logger.debug("Failed to write the demo detection schedule", exc_info=True)

    def _generate_detection_history(self, client: MlflowClient, experiment_id: str, now_ms: int):
        """
        Create the past detection runs the history page lists.

        Real runs, because the page finds them with a run search. Through
        ``create_run``/``set_terminated`` rather than the fluent API, which would stamp them all
        with now. Best-effort: losing one is no reason to lose the issues.
        """
        for spec in _DEMO_DETECTION_HISTORY:
            start_time = now_ms - spec["days_ago"] * _DAY_MS
            tags = {
                MLFLOW_RUN_TYPE: MLFLOW_RUN_TYPE_ISSUE_DETECTION,
                "categories": "correctness,relevance,safety",
                "model": _DEMO_DETECTION_MODEL,
                # Stringified: `create_run` stores tag values verbatim, unlike `start_run`.
                "total_traces": str(spec["total_traces"]),
            }
            if spec["scheduled"]:
                tags[MLFLOW_ISSUE_DETECTION_TRIGGER] = MLFLOW_ISSUE_DETECTION_TRIGGER_SCHEDULE
                tags[MLFLOW_ISSUE_DETECTION_WINDOW_START] = str(
                    start_time - spec["window_days"] * _DAY_MS
                )
                tags[MLFLOW_ISSUE_DETECTION_WINDOW_END] = str(start_time)
            if spec["cost_usd"] is not None:
                tags["total_cost_usd"] = spec["cost_usd"]
            if error := spec.get("error"):
                tags[MLFLOW_ISSUE_DETECTION_ERROR] = error
            try:
                run = client.create_run(
                    experiment_id=experiment_id,
                    start_time=start_time,
                    tags=tags,
                    run_name=spec["name"],
                )
            except Exception:
                _logger.debug("Failed to create a demo detection run", exc_info=True)
                continue
            # Its own block: a run left unterminated sits RUNNING, which keeps the history page
            # polling for as long as it is open. Deleted rather than left in that state.
            try:
                client.set_terminated(
                    run.info.run_id,
                    status=RunStatus.to_string(spec["status"]),
                    end_time=start_time + spec["duration_ms"],
                )
            except Exception:
                _logger.debug("Failed to terminate a demo detection run", exc_info=True)
                try:
                    client.delete_run(run.info.run_id)
                except Exception:
                    _logger.debug("Failed to clean up an unterminated run", exc_info=True)

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
            )
            return bool(issues)
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

        # No delete_issue API exists yet. Without cleanup here, regeneration would
        # pile new PENDING issues on top of the old ones (same names, same
        # experiment) and the UI would show duplicates. Mark the old demo issues
        # as REJECTED so they're hidden from the default "active issues" view —
        # this is also semantically correct since these issues referenced traces
        # that have just been deleted as part of the demo refresh.
        try:
            issues = store.search_issues(experiment_id=experiment.experiment_id)
            for issue in issues:
                if issue.created_by == _DEMO_CREATED_BY and issue.status == IssueStatus.PENDING:
                    store.update_issue(issue_id=issue.issue_id, status=IssueStatus.REJECTED)
        except Exception:
            _logger.debug("Failed to reject old demo issues", exc_info=True)

        # Both tags `write_detection_schedule` wrote, or the trace-filter tag outlives the demo.
        for tag in (MLFLOW_ISSUE_DETECTION_SCHEDULE, MLFLOW_ISSUE_DETECTION_SCHEDULE_TRACE_FILTER):
            try:
                MlflowClient().delete_experiment_tag(experiment.experiment_id, tag)
            except Exception:
                _logger.debug("Failed to remove demo detection schedule tag %s", tag, exc_info=True)

        # Note: Issues are also automatically deleted when the experiment is deleted.
