import uuid

import pytest

from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking._workspace.context import WorkspaceContext, clear_workspace
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME


@pytest.fixture
def workspace_registry_store(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    db_path = tmp_path / "registry.db"
    store = SqlAlchemyStore(f"sqlite:///{db_path}")
    try:
        yield store
    finally:
        store.engine.dispose()


def _names_from_search(results):
    return {rm.name for rm in results}


def test_registered_model_operations_are_workspace_scoped(workspace_registry_store):
    with WorkspaceContext("team-a"):
        workspace_registry_store.create_registered_model("alpha")
        workspace_registry_store.set_registered_model_tag(
            "alpha", RegisteredModelTag("owner", "team-a")
        )
        rm = workspace_registry_store.get_registered_model("alpha")
        assert rm.tags == {"owner": "team-a"}

    with WorkspaceContext("team-b"):
        workspace_registry_store.create_registered_model("beta")
        with pytest.raises(
            MlflowException, match="Registered Model with name=alpha not found"
        ) as excinfo:
            workspace_registry_store.set_registered_model_tag(
                "alpha", RegisteredModelTag("owner", "team-b")
            )
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"
        with pytest.raises(MlflowException, match="Registered Model with name=alpha not found"):
            workspace_registry_store.rename_registered_model("alpha", "alpha-b")
        with pytest.raises(MlflowException, match="Registered Model with name=alpha not found"):
            workspace_registry_store.delete_registered_model("alpha")

    with WorkspaceContext("team-b"):
        names = _names_from_search(workspace_registry_store.search_registered_models())
        assert names == {"beta"}

    with WorkspaceContext("team-b"):
        with pytest.raises(
            MlflowException, match="Registered Model with name=alpha-renamed not found"
        ) as excinfo:
            workspace_registry_store.set_registered_model_tag(
                "alpha-renamed", RegisteredModelTag("owner", "team-b")
            )
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"
        with pytest.raises(
            MlflowException, match="Registered Model with name=alpha-renamed not found"
        ):
            workspace_registry_store.get_registered_model("alpha-renamed")
        # Ensure team-b model remains accessible
        beta = workspace_registry_store.get_registered_model("beta")
        assert beta.name == "beta"

    with WorkspaceContext("team-a"):
        workspace_registry_store.rename_registered_model("alpha", "alpha-renamed")
        renamed = workspace_registry_store.get_registered_model("alpha-renamed")
        assert renamed.name == "alpha-renamed"
        assert renamed.tags == {"owner": "team-a"}

    with WorkspaceContext("team-b"):
        with pytest.raises(
            MlflowException, match="Registered Model with name=alpha-renamed not found"
        ) as excinfo:
            workspace_registry_store.set_registered_model_tag(
                "alpha-renamed", RegisteredModelTag("owner", "team-b")
            )
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"
        with pytest.raises(
            MlflowException, match="Registered Model with name=alpha-renamed not found"
        ):
            workspace_registry_store.get_registered_model("alpha-renamed")

    with WorkspaceContext("team-a"):
        workspace_registry_store.delete_registered_model("alpha-renamed")
        with pytest.raises(
            MlflowException, match="Registered Model with name=alpha-renamed not found"
        ) as excinfo:
            workspace_registry_store.get_registered_model("alpha-renamed")
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_model_version_operations_are_workspace_scoped(workspace_registry_store):
    with WorkspaceContext("team-a"):
        workspace_registry_store.create_registered_model("alpha")
        mv_a = workspace_registry_store.create_model_version(
            "alpha", "s3://team-a/model", run_id=uuid.uuid4().hex
        )
        assert mv_a.version == 1
        workspace_registry_store.set_model_version_tag(
            "alpha", str(mv_a.version), ModelVersionTag("env", "prod")
        )
        workspace_registry_store.transition_model_version_stage(
            "alpha", str(mv_a.version), "Production", archive_existing_versions=False
        )
        workspace_registry_store.set_registered_model_alias(
            "alpha", "production", str(mv_a.version)
        )
        mv_detail = workspace_registry_store.get_model_version("alpha", "1")
        assert mv_detail.current_stage == "Production"
        assert mv_detail.tags == {"env": "prod"}
        aliases = workspace_registry_store.get_registered_model("alpha").aliases
        assert aliases == {"production": 1}
        download_uri = workspace_registry_store.get_model_version_download_uri("alpha", "1")
        assert download_uri == "s3://team-a/model"

    with WorkspaceContext("team-b"):
        workspace_registry_store.create_registered_model("beta")
        with pytest.raises(
            MlflowException, match="Registered Model with name=alpha not found"
        ) as excinfo:
            workspace_registry_store.create_model_version(
                "alpha", "s3://team-b/model", run_id=uuid.uuid4().hex
            )
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"

        version_scoped_calls = (
            lambda: workspace_registry_store.transition_model_version_stage(
                "alpha", "1", "Archived", archive_existing_versions=False
            ),
            lambda: workspace_registry_store.set_model_version_tag(
                "alpha", "1", ModelVersionTag("env", "stage")
            ),
            lambda: workspace_registry_store.delete_model_version_tag("alpha", "1", "env"),
            lambda: workspace_registry_store.delete_model_version("alpha", "1"),
            lambda: workspace_registry_store.get_model_version_download_uri("alpha", "1"),
        )
        for call in version_scoped_calls:
            with pytest.raises(
                MlflowException, match=r"Model Version \(name=alpha, version=1\) not found"
            ) as excinfo:
                call()
            assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"

        alias_scoped_calls = (
            lambda: workspace_registry_store.set_registered_model_alias("alpha", "shadow", "1"),
            lambda: workspace_registry_store.delete_registered_model_alias("alpha", "production"),
        )
        for call in alias_scoped_calls:
            with pytest.raises(
                MlflowException,
                match=(
                    r"(Model Version \(name=alpha, version=1\) not found|"
                    r"Registered Model with name=alpha not found)"
                ),
            ) as excinfo:
                call()
            assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"

    with WorkspaceContext("team-a"):
        workspace_registry_store.delete_model_version_tag("alpha", "1", "env")
        mv_detail = workspace_registry_store.get_model_version("alpha", "1")
        assert mv_detail.tags == {}
        workspace_registry_store.delete_registered_model_alias("alpha", "production")
        assert workspace_registry_store.get_registered_model("alpha").aliases == {}
        workspace_registry_store.delete_model_version("alpha", "1")
        with pytest.raises(
            MlflowException, match=r"Model Version \(name=alpha, version=1\) not found"
        ) as excinfo:
            workspace_registry_store.get_model_version("alpha", "1")
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_model_version_read_helpers_are_workspace_scoped(workspace_registry_store):
    with WorkspaceContext("team-a"):
        workspace_registry_store.create_registered_model("alpha")
        workspace_registry_store.create_model_version(
            "alpha", "s3://team-a/model", run_id=uuid.uuid4().hex
        )
        versions = workspace_registry_store.search_model_versions("name='alpha'")
        assert [mv.version for mv in versions] == [1]
        latest_versions = workspace_registry_store.get_latest_versions("alpha")
        assert [mv.version for mv in latest_versions] == [1]
        fetched = workspace_registry_store.get_model_version("alpha", "1")
        assert fetched.version == 1

    with WorkspaceContext("team-b"):
        assert workspace_registry_store.search_model_versions("name='alpha'") == []
        with pytest.raises(
            MlflowException, match=r"Model Version \(name=alpha, version=1\) not found"
        ) as excinfo:
            workspace_registry_store.get_model_version("alpha", "1")
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"
        with pytest.raises(
            MlflowException, match="Registered Model with name=alpha not found"
        ) as excinfo:
            workspace_registry_store.get_latest_versions("alpha")
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_same_model_name_allowed_in_different_workspaces(workspace_registry_store):
    with WorkspaceContext("team-a"):
        workspace_registry_store.create_registered_model("shared-name")
    with WorkspaceContext("team-b"):
        workspace_registry_store.create_registered_model("shared-name")
        names = _names_from_search(workspace_registry_store.search_registered_models())
        assert names == {"shared-name"}

    with WorkspaceContext("team-a"):
        names = _names_from_search(workspace_registry_store.search_registered_models())
        assert names == {"shared-name"}


def test_update_and_delete_registered_model_metadata_are_workspace_scoped(
    workspace_registry_store,
):
    with WorkspaceContext("team-a"):
        workspace_registry_store.create_registered_model("alpha")
        workspace_registry_store.set_registered_model_tag(
            "alpha", RegisteredModelTag("owner", "team-a")
        )
        updated = workspace_registry_store.update_registered_model("alpha", "updated desc")
        assert updated.description == "updated desc"
        workspace_registry_store.delete_registered_model_tag("alpha", "owner")
        assert workspace_registry_store.get_registered_model("alpha").tags == {}

    with WorkspaceContext("team-b"):
        with pytest.raises(
            MlflowException, match="Registered Model with name=alpha not found"
        ) as excinfo:
            workspace_registry_store.update_registered_model("alpha", "hijacked")
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"
        with pytest.raises(
            MlflowException, match="Registered Model with name=alpha not found"
        ) as excinfo:
            workspace_registry_store.delete_registered_model_tag("alpha", "owner")
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_webhook_operations_are_workspace_scoped(workspace_registry_store):
    event = WebhookEvent(WebhookEntity.MODEL_VERSION, WebhookAction.CREATED)

    with WorkspaceContext("team-a"):
        webhook = workspace_registry_store.create_webhook(
            name="team-a-hook",
            url="https://example.com/hook",
            events=[event],
            description="Team A hook",
        )
        assert webhook.workspace == "team-a"
        owned_hooks = workspace_registry_store.list_webhooks()
        assert len(owned_hooks) == 1
        assert owned_hooks[0].webhook_id == webhook.webhook_id
        assert owned_hooks[0].workspace == "team-a"

    with WorkspaceContext("team-b"):
        assert len(workspace_registry_store.list_webhooks()) == 0
        assert (
            len(
                workspace_registry_store.list_webhooks_by_event(
                    event, max_results=10, page_token=None
                )
            )
            == 0
        )
        with pytest.raises(
            MlflowException, match=f"Webhook with ID {webhook.webhook_id} not found"
        ) as excinfo:
            workspace_registry_store.get_webhook(webhook.webhook_id)
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"
        with pytest.raises(
            MlflowException, match=f"Webhook with ID {webhook.webhook_id} not found"
        ) as excinfo:
            workspace_registry_store.update_webhook(webhook.webhook_id, name="should-fail")
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"
        with pytest.raises(
            MlflowException, match=f"Webhook with ID {webhook.webhook_id} not found"
        ) as excinfo:
            workspace_registry_store.delete_webhook(webhook.webhook_id)
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"

    with WorkspaceContext("team-a"):
        fetched = workspace_registry_store.get_webhook(webhook.webhook_id)
        assert fetched.webhook_id == webhook.webhook_id
        assert fetched.workspace == "team-a"
        workspace_registry_store.delete_webhook(webhook.webhook_id)
        with pytest.raises(
            MlflowException, match=f"Webhook with ID {webhook.webhook_id} not found"
        ) as excinfo:
            workspace_registry_store.get_webhook(webhook.webhook_id)
        assert excinfo.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_default_workspace_behavior_when_workspaces_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv(MLFLOW_ENABLE_WORKSPACES.name, raising=False)
    clear_workspace()
    store = SqlAlchemyStore(f"sqlite:///{tmp_path / 'legacy.db'}")
    try:
        rm = store.create_registered_model("legacy-model")
        assert rm.name == "legacy-model"
        fetched = store.get_registered_model("legacy-model")
        assert fetched.name == "legacy-model"
    finally:
        store.engine.dispose()


def test_default_workspace_context_allows_operations(workspace_registry_store):
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        workspace_registry_store.create_registered_model("default-model")
        fetched = workspace_registry_store.get_registered_model("default-model")
        assert fetched.name == "default-model"


def test_single_tenant_registry_startup_rejects_non_default_workspace_models(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    db_path = tmp_path / "registry_multi_tenant.db"
    workspace_store = SqlAlchemyStore(f"sqlite:///{db_path}")

    with WorkspaceContext("team-startup"):
        workspace_store.create_registered_model("team-model")

    workspace_store.engine.dispose()

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    with pytest.raises(
        MlflowException,
        match="Cannot disable workspaces because registered models exist outside the default "
        + "workspace",
    ) as excinfo:
        SqlAlchemyStore(f"sqlite:///{db_path}")

    assert excinfo.value.error_code == "INVALID_STATE"

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    webhook_db_path = tmp_path / "registry_webhook.db"
    webhook_store = SqlAlchemyStore(f"sqlite:///{webhook_db_path}")
    webhook_event = WebhookEvent(WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED)

    with WorkspaceContext("team-webhook"):
        webhook_store.create_webhook(
            name="team-webhook",
            url="https://example.com/webhook",
            events=[webhook_event],
            description="non-default webhook",
        )

    webhook_store.engine.dispose()

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    with pytest.raises(
        MlflowException,
        match="Cannot disable workspaces because webhooks exist outside the default workspace",
    ) as excinfo:
        SqlAlchemyStore(f"sqlite:///{webhook_db_path}")

    assert excinfo.value.error_code == "INVALID_STATE"
