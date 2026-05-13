from mlflow.genai import labeling
from mlflow.genai.labeling import (
    Agent,
    LabelingSession,
    ReviewApp,
    create_labeling_session,
    delete_labeling_session,
    get_labeling_session,
    get_labeling_sessions,
    get_review_app,
)

from tests.genai.conftest import databricks_only


@databricks_only
def test_databricks_labeling_is_importable():
    assert labeling.Agent == Agent
    assert labeling.LabelingSession == LabelingSession
    assert labeling.ReviewApp == ReviewApp
    assert labeling.get_review_app == get_review_app
    assert labeling.create_labeling_session == create_labeling_session
    assert labeling.get_labeling_sessions == get_labeling_sessions
    assert labeling.get_labeling_session == get_labeling_session
    assert labeling.delete_labeling_session == delete_labeling_session
