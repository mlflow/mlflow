"""
Result models returned by the MLflow MCP tools.

Each tool returns one of these models. FastMCP derives the tool's output schema from the return
annotation and sends the dumped model as ``structuredContent`` alongside its JSON text.
"""

from typing import Any, Literal

from pydantic import BaseModel

from mlflow.entities import Assessment, Expectation, Experiment, Feedback, Run


class ExperimentInfo(BaseModel):
    experiment_id: str
    name: str
    artifact_location: str | None
    lifecycle_stage: str
    creation_time: int | None
    last_update_time: int | None
    tags: dict[str, str]

    @classmethod
    def from_entity(cls, experiment: Experiment) -> "ExperimentInfo":
        return cls(
            experiment_id=experiment.experiment_id,
            name=experiment.name,
            artifact_location=experiment.artifact_location,
            lifecycle_stage=experiment.lifecycle_stage,
            creation_time=experiment.creation_time,
            last_update_time=experiment.last_update_time,
            tags=dict(experiment.tags),
        )


class ExperimentPage(BaseModel):
    experiments: list[ExperimentInfo]
    next_page_token: str | None


class ExperimentRef(BaseModel):
    experiment_id: str
    name: str | None = None


class ExperimentUpdate(BaseModel):
    experiment_id: str
    changes: list[str]


class RunSummary(BaseModel):
    run_id: str
    run_name: str | None
    experiment_id: str
    status: str
    lifecycle_stage: str
    start_time: int | None
    end_time: int | None

    @classmethod
    def from_entity(cls, run: Run) -> "RunSummary":
        info = run.info
        return cls(
            run_id=info.run_id,
            run_name=info.run_name,
            experiment_id=info.experiment_id,
            status=info.status,
            lifecycle_stage=info.lifecycle_stage,
            start_time=info.start_time,
            end_time=info.end_time,
        )


class RunPage(BaseModel):
    runs: list[RunSummary]
    next_page_token: str | None


class RunDetails(BaseModel):
    run_id: str
    run_name: str | None
    experiment_id: str
    user_id: str | None
    status: str
    lifecycle_stage: str
    start_time: int | None
    end_time: int | None
    artifact_uri: str | None
    metrics: dict[str, float]
    params: dict[str, str]
    tags: dict[str, str]
    inputs: dict[str, Any] | None
    outputs: dict[str, Any] | None

    @classmethod
    def from_entity(cls, run: Run) -> "RunDetails":
        info = run.info
        return cls(
            run_id=info.run_id,
            run_name=info.run_name,
            experiment_id=info.experiment_id,
            user_id=info.user_id,
            status=info.status,
            lifecycle_stage=info.lifecycle_stage,
            start_time=info.start_time,
            end_time=info.end_time,
            artifact_uri=info.artifact_uri,
            metrics=dict(run.data.metrics),
            params=dict(run.data.params),
            tags=dict(run.data.tags),
            inputs=run.inputs.to_dictionary() if run.inputs else None,
            outputs=run.outputs.to_dictionary() if run.outputs else None,
        )


class CreatedRun(BaseModel):
    run_id: str
    experiment_id: str
    run_name: str | None
    status: str


class RunRef(BaseModel):
    run_id: str


class LinkedTraces(BaseModel):
    run_id: str
    trace_ids: list[str]


class TraceResult(BaseModel):
    # The trace in the shape of ``Trace.to_dict()``, projected to ``extract_fields`` when given.
    trace: dict[str, Any]


class TracePage(BaseModel):
    traces: list[dict[str, Any]]
    next_page_token: str | None


class DeletedTraces(BaseModel):
    experiment_id: str
    deleted_count: int


class TraceTag(BaseModel):
    trace_id: str
    key: str
    value: str | None = None


class AssessmentSourceInfo(BaseModel):
    source_type: str
    source_id: str | None


class AssessmentErrorInfo(BaseModel):
    error_code: str | None
    error_message: str | None


class AssessmentInfo(BaseModel):
    assessment_id: str | None
    trace_id: str | None
    name: str
    kind: Literal["feedback", "expectation", "other"]
    value: Any | None = None
    error: AssessmentErrorInfo | None = None
    rationale: str | None = None
    metadata: dict[str, str] | None = None
    span_id: str | None = None
    source: AssessmentSourceInfo | None = None
    create_time_ms: int | None = None
    last_update_time_ms: int | None = None
    valid: bool | None = None
    overrides: str | None = None

    @classmethod
    def from_entity(cls, assessment: Assessment) -> "AssessmentInfo":
        value = None
        error = None
        if isinstance(assessment, Feedback):
            kind = "feedback"
            value = assessment.value
            if assessment.error_code is not None or assessment.error_message is not None:
                error = AssessmentErrorInfo(
                    error_code=assessment.error_code,
                    error_message=assessment.error_message,
                )
        elif isinstance(assessment, Expectation):
            kind = "expectation"
            value = assessment.value
        else:
            kind = "other"
        source = assessment.source
        return cls(
            assessment_id=assessment.assessment_id,
            trace_id=assessment.trace_id,
            name=assessment.name,
            kind=kind,
            value=value,
            error=error,
            rationale=assessment.rationale,
            metadata=assessment.metadata,
            span_id=assessment.span_id,
            source=AssessmentSourceInfo(source_type=source.source_type, source_id=source.source_id)
            if source is not None
            else None,
            create_time_ms=assessment.create_time_ms,
            last_update_time_ms=assessment.last_update_time_ms,
            valid=assessment.valid,
            overrides=assessment.overrides,
        )


class AssessmentRef(BaseModel):
    trace_id: str
    assessment_id: str


class ScorerInfo(BaseModel):
    name: str
    description: str | None
    # Populated for built-in scorers only.
    required_columns: list[str] | None = None
    session_level: bool | None = None
    required_args: list[str] | None = None


class ScorerList(BaseModel):
    scorers: list[ScorerInfo]


class RegisteredScorer(BaseModel):
    name: str
    experiment_id: str
    # 1 for a new scorer; registering an existing name adds a version.
    version: int
