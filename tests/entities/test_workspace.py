from mlflow.entities.workspace import Workspace


def test_workspace_to_dict_uses_nested_trace_archival_config():
    workspace = Workspace(
        name="team-a",
        description="Team A",
        default_artifact_root="s3://artifacts/team-a",
        trace_archival_location="s3://archive/team-a",
        trace_archival_retention="30d",
    )

    assert workspace.to_dict() == {
        "name": "team-a",
        "description": "Team A",
        "default_artifact_root": "s3://artifacts/team-a",
        "trace_archival_config": {
            "location": "s3://archive/team-a",
            "retention": "30d",
        },
    }


def test_workspace_from_dict_accepts_nested_trace_archival_config():
    workspace = Workspace.from_dict({
        "name": "team-b",
        "description": "Team B",
        "trace_archival_config": {
            "location": "s3://archive/team-b",
            "retention": "14d",
        },
    })

    assert workspace == Workspace(
        name="team-b",
        description="Team B",
        trace_archival_location="s3://archive/team-b",
        trace_archival_retention="14d",
    )
