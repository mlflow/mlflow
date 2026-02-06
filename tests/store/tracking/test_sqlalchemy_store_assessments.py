import time
import uuid

import pytest

from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback, LifecycleStage
from mlflow.store.tracking.dbmodels.models import SqlAssessments
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.file_utils import TempDir


@pytest.fixture
def store():
    with TempDir() as tmp:
        db_uri = f"sqlite:///{tmp.path('test.db')}"
        artifact_uri = tmp.path("artifacts")
        store = SqlAlchemyStore(db_uri, artifact_uri)
        yield store


def test_delete_run_deletes_associated_assessments(store):
    # Create Experiment
    exp_id = store.create_experiment("test_experiment")

    # Create Run
    run = store.create_run(
        experiment_id=exp_id,
        user_id="user",
        start_time=int(time.time() * 1000),
        tags=[],
        run_name="test-run",
    )
    run_id = run.info.run_id

    # Create Trace
    from mlflow.entities.trace_info import TraceInfo
    from mlflow.entities.trace_location import TraceLocation
    from mlflow.entities.trace_state import TraceState

    trace_id = f"tr-{uuid.uuid4().hex}"
    store.start_trace(
        TraceInfo(
            trace_id=trace_id,
            trace_location=TraceLocation.from_experiment_id(exp_id),
            request_time=int(time.time() * 1000),
            execution_duration=0,
            state=TraceState.OK,
            tags={},
            trace_metadata={},
            client_request_id="req-123",
        )
    )

    # Create Assessment linked to run
    assessment = Feedback(
        name="safety",
        value=0.5,
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE),
        trace_id=trace_id,
    )
    # Ensure run_id is set
    assessment.run_id = run_id

    created_assessment = store.create_assessment(assessment)
    assessment_id = created_assessment.assessment_id

    # Verify existence
    retrieved = store.get_assessment(trace_id, assessment_id)
    assert retrieved is not None

    # Delete Run
    store.delete_run(run_id)

    # Verify Run is deleted
    deleted_run = store.get_run(run_id)
    assert deleted_run.info.lifecycle_stage == LifecycleStage.DELETED

    # Verify Assessment is deleted
    # Direct DB check to ensure it's gone
    with store.ManagedSessionMaker() as session:
        sql_ass = session.query(SqlAssessments).filter_by(assessment_id=assessment_id).first()
        assert sql_ass is None

    # Also verify get_assessment raises or returns None (depending on implementation, usually raises)
    from mlflow.exceptions import MlflowException

    with pytest.raises(MlflowException, match=r"Assessment .* not found"):
        store.get_assessment(trace_id, assessment_id)
