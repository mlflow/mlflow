import json
import os
import uuid
from pathlib import Path

from mlflow.entities import Assessment, Issue, Span, Trace, TraceData, TraceInfo
from mlflow.exceptions import MlflowNotImplementedException
from mlflow.utils.workspace_context import get_request_workspace

from tests.store.tracking.sqlalchemy_store.test_trace_backend_associations import MemoryTraceBackend


class JsonTraceBackend(MemoryTraceBackend):
    """Single-writer test plugin persisting traces independently of the tracking database."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.spans = {}
        if path.exists():
            state = json.loads(path.read_text())
            self.traces = {
                trace_id: (entry["workspace"], TraceInfo.from_dict(entry["info"]))
                for trace_id, entry in state["traces"].items()
            }
            self.spans = {
                trace_id: [Span.from_dict(span) for span in spans]
                for trace_id, spans in state["spans"].items()
            }
            self.assessments = [
                Assessment.from_dictionary(assessment) for assessment in state["assessments"]
            ]
            self.issues = {
                issue_id: (entry["workspace"], Issue.from_dictionary(entry["issue"]))
                for issue_id, entry in state["issues"].items()
            }

    def _save(self):
        self.path.write_text(
            json.dumps({
                "traces": {
                    trace_id: {"workspace": workspace, "info": info.to_dict()}
                    for trace_id, (workspace, info) in self.traces.items()
                },
                "spans": {
                    trace_id: [span.to_dict() for span in spans]
                    for trace_id, spans in self.spans.items()
                },
                "assessments": [assessment.to_dictionary() for assessment in self.assessments],
                "issues": {
                    issue_id: {"workspace": workspace, "issue": issue.to_dictionary()}
                    for issue_id, (workspace, issue) in self.issues.items()
                },
            })
        )

    def start_trace(self, trace_info):
        self._tracking_store.get_experiment(trace_info.experiment_id)
        info = super().start_trace(trace_info)
        self._save()
        return info

    def get_trace_info(self, trace_id):
        info = super().get_trace_info(trace_id)
        info.assessments = [a for a in self.assessments if a.trace_id == trace_id]
        return info

    def log_spans(self, location, spans, tracking_uri=None):
        for span in spans:
            assert self.get_trace_info(span.trace_id).experiment_id == location
            self.spans.setdefault(span.trace_id, []).append(span)
        self._save()
        return spans

    def get_trace(self, trace_id, *, allow_partial=False):
        return Trace(self.get_trace_info(trace_id), TraceData(spans=self.spans[trace_id]))

    def batch_get_traces(self, trace_ids, location=None, experiment_ids=None):
        return [
            self.get_trace(info.trace_id)
            for info in self.batch_get_trace_infos(trace_ids, location, experiment_ids)
        ]

    def search_traces(
        self,
        experiment_ids=None,
        filter_string=None,
        max_results=100,
        order_by=None,
        page_token=None,
        model_id=None,
        locations=None,
    ):
        if filter_string or order_by or model_id:
            raise MlflowNotImplementedException("The test plugin only supports unfiltered search")
        allowed_experiments = locations if locations is not None else experiment_ids
        infos = [
            self.get_trace_info(trace_id)
            for trace_id, (workspace, info) in self.traces.items()
            if workspace == get_request_workspace()
            and (allowed_experiments is None or info.experiment_id in allowed_experiments)
        ]
        offset = int(page_token) if page_token else 0
        end = offset + max_results
        return infos[offset:end], str(end) if end < len(infos) else None

    def create_assessment(self, assessment):
        assessment.assessment_id = uuid.uuid4().hex
        result = super().create_assessment(assessment)
        self._save()
        return result

    def create_issue(self, experiment_id, name, description, **kwargs):
        kwargs["issue_id"] = kwargs.get("issue_id") or f"iss-{uuid.uuid4().hex}"
        issue = super().create_issue(experiment_id, name, description, **kwargs)
        self._save()
        return issue

    def update_issue(self, issue_id, **kwargs):
        issue = super().update_issue(issue_id, **kwargs)
        self._save()
        return issue

    def _delete_ids(self, trace_ids):
        super()._delete_ids(trace_ids)
        for trace_id in trace_ids:
            self.spans.pop(trace_id, None)
        self._save()

    def on_run_deleted(self, run_id):
        super().on_run_deleted(run_id)
        self._save()

    def on_run_permanently_deleted(self, run_id):
        super().on_run_permanently_deleted(run_id)
        self._save()

    def on_experiment_permanently_deleted(self, experiment_id):
        super().on_experiment_permanently_deleted(experiment_id)
        self._save()

    def on_workspace_deleted(self, workspace_name):
        super().on_workspace_deleted(workspace_name)
        self._save()


def build_backend(config):
    assert config == {}
    return JsonTraceBackend(Path(os.environ["MLFLOW_TEST_JSON_TRACE_BACKEND_PATH"]))
