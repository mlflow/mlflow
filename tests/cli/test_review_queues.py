import json
from unittest import mock

import pytest
from click.testing import CliRunner

from mlflow.cli.review_queues import commands
from mlflow.exceptions import MlflowException
from mlflow.genai.review_queues import (
    ReviewItemType,
    ReviewQueue,
    ReviewQueueItem,
    ReviewQueueType,
    ReviewStatus,
)
from mlflow.store.entities.paged_list import PagedList


@pytest.fixture
def runner():
    return CliRunner()


def _queue(queue_id="rq-1", name="triage", users=None, schema_ids=None):
    return ReviewQueue(
        queue_id=queue_id,
        experiment_id="0",
        name=name,
        queue_type=ReviewQueueType.CUSTOM,
        created_by="alice",
        creation_time_ms=1,
        last_update_time_ms=1,
        users=users or [],
        schema_ids=schema_ids or [],
    )


def _item(item_id="tr-1", status=ReviewStatus.PENDING, completed_by=None):
    return ReviewQueueItem(
        queue_id="rq-1",
        item_type=ReviewItemType.TRACE,
        item_id=item_id,
        status=status,
        creation_time_ms=1,
        last_update_time_ms=1,
        completed_by=completed_by,
        completed_time_ms=2 if completed_by else None,
    )


def test_create_user_queue(runner):
    with mock.patch(
        "mlflow.cli.review_queues.create_review_queue", return_value=_queue()
    ) as mock_create:
        result = runner.invoke(
            commands,
            ["create", "--name", "alice", "--queue-type", "user", "-x", "0", "--output", "json"],
        )
    assert result.exit_code == 0
    assert json.loads(result.output)["queue_id"] == "rq-1"
    mock_create.assert_called_once_with(
        name="alice", queue_type="user", users=None, schema_ids=None, experiment_id="0"
    )


def test_create_custom_parses_csv_lists(runner):
    with mock.patch(
        "mlflow.cli.review_queues.create_review_queue", return_value=_queue()
    ) as mock_create:
        result = runner.invoke(
            commands,
            [
                "create",
                "--name",
                "triage",
                "--queue-type",
                "custom",
                "--users",
                "alice, bob",
                "--schema-ids",
                "s1,s2",
                "-x",
                "0",
            ],
        )
    assert result.exit_code == 0
    mock_create.assert_called_once_with(
        name="triage",
        queue_type="custom",
        users=["alice", "bob"],
        schema_ids=["s1", "s2"],
        experiment_id="0",
    )


def test_create_bad_queue_type(runner):
    result = runner.invoke(commands, ["create", "--name", "x", "--queue-type", "bad", "-x", "0"])
    assert result.exit_code == 2
    assert "'bad' is not one of 'user', 'custom'" in result.output


@pytest.mark.parametrize("args", [[], ["--queue-id", "rq-1", "--name", "n"]])
def test_get_xor_selector(runner, args):
    result = runner.invoke(commands, ["get", *args, "-x", "0"])
    assert result.exit_code == 2
    assert "exactly one of --queue-id or --name" in result.output


def test_list_default_max_results_and_teachline(runner):
    with mock.patch(
        "mlflow.cli.review_queues.list_review_queues",
        return_value=PagedList([_queue()], "tok-9"),
    ) as mock_list:
        result = runner.invoke(commands, ["list", "-x", "0"])
    assert result.exit_code == 0
    assert "--page-token tok-9" in result.stderr
    mock_list.assert_called_once_with(
        user=None, experiment_id="0", max_results=100, page_token=None
    )


def test_update_empty_users_clears(runner):
    with mock.patch(
        "mlflow.cli.review_queues.update_review_queue", return_value=_queue()
    ) as mock_update:
        result = runner.invoke(commands, ["update", "--queue-id", "rq-1", "--users", ""])
    assert result.exit_code == 0
    _, kwargs = mock_update.call_args
    assert kwargs["users"] == []
    assert kwargs["schema_ids"] is None


def test_delete_json_confirmation(runner):
    with mock.patch("mlflow.cli.review_queues.delete_review_queue") as mock_delete:
        result = runner.invoke(commands, ["delete", "--queue-id", "rq-1", "--output", "json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {"queue_id": "rq-1", "deleted": True}
    mock_delete.assert_called_once_with(queue_id="rq-1")


def test_get_or_create_user_queue(runner):
    with mock.patch(
        "mlflow.cli.review_queues.get_or_create_user_queue", return_value=_queue()
    ) as mock_fn:
        result = runner.invoke(commands, ["get-or-create-user-queue", "--user", "alice", "-x", "0"])
    assert result.exit_code == 0
    mock_fn.assert_called_once_with(user="alice", experiment_id="0")


def test_items_add_json(runner):
    with mock.patch(
        "mlflow.cli.review_queues.add_items_to_review_queue",
        return_value=[_item("tr-1"), _item("tr-2")],
    ) as mock_add:
        result = runner.invoke(
            commands,
            ["items", "add", "--queue-id", "rq-1", "--item-ids", "tr-1,tr-2", "--output", "json"],
        )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [i["item_id"] for i in payload["items"]] == ["tr-1", "tr-2"]
    mock_add.assert_called_once_with(queue_id="rq-1", item_ids=["tr-1", "tr-2"])


def test_items_add_requires_item_ids(runner):
    result = runner.invoke(commands, ["items", "add", "--queue-id", "rq-1", "--item-ids", ""])
    assert result.exit_code == 2
    assert "--item-ids must contain at least one value" in result.output


def test_items_remove_json_confirmation(runner):
    with mock.patch("mlflow.cli.review_queues.remove_items_from_review_queue") as mock_remove:
        result = runner.invoke(
            commands,
            ["items", "remove", "--queue-id", "rq-1", "--item-ids", "tr-1", "--output", "json"],
        )
    assert result.exit_code == 0
    assert json.loads(result.output) == {"queue_id": "rq-1", "removed_item_ids": ["tr-1"]}
    mock_remove.assert_called_once_with(queue_id="rq-1", item_ids=["tr-1"])


def test_items_list_status_filter(runner):
    with mock.patch(
        "mlflow.cli.review_queues.list_review_queue_items",
        return_value=PagedList([_item()], None),
    ) as mock_list:
        result = runner.invoke(
            commands,
            ["items", "list", "--queue-id", "rq-1", "--status", "pending", "--output", "json"],
        )
    assert result.exit_code == 0
    mock_list.assert_called_once_with(
        queue_id="rq-1", status="pending", max_results=100, page_token=None
    )


def test_items_set_status_complete(runner):
    with mock.patch(
        "mlflow.cli.review_queues.set_review_queue_item_status",
        return_value=_item(status=ReviewStatus.COMPLETE, completed_by="alice"),
    ) as mock_set:
        result = runner.invoke(
            commands,
            [
                "items",
                "set-status",
                "--queue-id",
                "rq-1",
                "--item-id",
                "tr-1",
                "--status",
                "complete",
                "--completed-by",
                "alice",
                "--output",
                "json",
            ],
        )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "COMPLETE"
    assert payload["completed_by"] == "alice"
    mock_set.assert_called_once_with(
        queue_id="rq-1", item_id="tr-1", status="complete", completed_by="alice"
    )


def test_items_set_status_bad_enum(runner):
    result = runner.invoke(
        commands,
        ["items", "set-status", "--queue-id", "rq-1", "--item-id", "tr-1", "--status", "nope"],
    )
    assert result.exit_code == 2
    assert "'nope' is not one of 'pending', 'complete', 'declined'" in result.output


def test_mlflow_exception_rendered_cleanly(runner):
    with mock.patch(
        "mlflow.cli.review_queues.set_review_queue_item_status",
        side_effect=MlflowException(
            "`completed_by` is required when setting status to `complete`."
        ),
    ):
        result = runner.invoke(
            commands,
            [
                "items",
                "set-status",
                "--queue-id",
                "rq-1",
                "--item-id",
                "tr-1",
                "--status",
                "complete",
            ],
        )
    assert result.exit_code == 1
    assert "`completed_by` is required" in result.output
    assert "Traceback" not in result.output
