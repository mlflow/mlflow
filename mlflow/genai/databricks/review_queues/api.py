"""Review queues on Databricks.

A Databricks-native client for review queues. Unlike :mod:`mlflow.genai.review_queues`
(which targets the MLflow tracking server), the functions here mirror the Databricks
review queue REST API one-to-one: questions, queues, members, and items are separate
resources nested under an experiment, and every function takes the ``experiment_id``
explicitly. They require a Databricks tracking URI.
"""

from typing import Any, Literal
from urllib.parse import quote

from mlflow.exceptions import MlflowException
from mlflow.genai.databricks.review_queues import _client
from mlflow.genai.databricks.review_queues.entities import (
    ItemKind,
    QuestionInput,
    QuestionType,
    ReviewQuestion,
    ReviewQueue,
    ReviewQueueItem,
    ReviewQueueMember,
    ReviewQueueType,
    ReviewStatus,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracing.utils import parse_trace_id_v4
from mlflow.utils.annotations import experimental


def _experiment(experiment_id: str) -> str:
    return f"experiments/{experiment_id}"


def _question(experiment_id: str, question_id: str) -> str:
    return f"{_experiment(experiment_id)}/reviewQuestions/{question_id}"


def _queue(experiment_id: str, queue_id: str) -> str:
    return f"{_experiment(experiment_id)}/reviewQueues/{queue_id}"


def _set_if_not_none(body: dict[str, Any], mask: list[str], key: str, value: Any) -> None:
    if value is not None:
        body[key] = value
        mask.append(key)


def _require_update(mask: list[str]) -> None:
    if not mask:
        raise MlflowException.invalid_parameter_value("Specify at least one field to update.")


# ---- Review questions ----


@experimental(version="3.17.1")
def create_review_question(
    experiment_id: str,
    *,
    title: str,
    type: QuestionType | Literal["FEEDBACK", "EXPECTATION"],
    input: QuestionInput,
    instruction: str | None = None,
    enable_comment: bool | None = None,
    is_default: bool | None = None,
) -> ReviewQuestion:
    """
    Create a review question in an experiment.

    Args:
        experiment_id: The experiment that owns the question.
        title: Short prompt shown to reviewers.
        type: ``"FEEDBACK"`` or ``"EXPECTATION"``.
        input: The answer input, e.g. ``InputCategorical(options=["yes", "no"])``.
        instruction: Longer guidance shown alongside the question.
        enable_comment: Whether reviewers may add a free-text comment.
        is_default: Whether this is the experiment's default question.

    Returns:
        The created :py:class:`ReviewQuestion`.
    """
    body = {
        "title": title,
        "type": QuestionType(type).value,
        input._FIELD: input.to_dict(),
    }
    for key, value in {
        "instruction": instruction,
        "enable_comment": enable_comment,
        "is_default": is_default,
    }.items():
        if value is not None:
            body[key] = value
    resp = _client.call("POST", f"{_experiment(experiment_id)}/reviewQuestions", json=body)
    return ReviewQuestion.from_dict(resp)


@experimental(version="3.17.1")
def get_review_question(experiment_id: str, question_id: str) -> ReviewQuestion:
    """
    Get a review question by ID.

    Args:
        experiment_id: The experiment that owns the question.
        question_id: The question to get.

    Returns:
        The :py:class:`ReviewQuestion`.
    """
    return ReviewQuestion.from_dict(_client.call("GET", _question(experiment_id, question_id)))


@experimental(version="3.17.1")
def list_review_questions(
    experiment_id: str,
    *,
    max_results: int | None = None,
    page_token: str | None = None,
) -> PagedList[ReviewQuestion]:
    """
    List an experiment's review questions.

    Args:
        experiment_id: The experiment to list.
        max_results: Page size.
        page_token: Continuation token from a previous call.

    Returns:
        A :py:class:`PagedList` of :py:class:`ReviewQuestion`.
    """
    resp = _client.call(
        "GET",
        f"{_experiment(experiment_id)}/reviewQuestions",
        params={"page_size": max_results, "page_token": page_token},
    )
    return PagedList(
        [ReviewQuestion.from_dict(q) for q in resp.get("review_questions", [])],
        resp.get("next_page_token"),
    )


@experimental(version="3.17.1")
def update_review_question(
    experiment_id: str,
    question_id: str,
    *,
    title: str | None = None,
    instruction: str | None = None,
    input: QuestionInput | None = None,
    enable_comment: bool | None = None,
    is_default: bool | None = None,
) -> ReviewQuestion:
    """
    Update a review question. Only the fields that are passed are changed.

    Args:
        experiment_id: The experiment that owns the question.
        question_id: The question to update.
        title: New short prompt shown to reviewers.
        instruction: New guidance shown alongside the question.
        input: New answer input. Replaces the existing input.
        enable_comment: Whether reviewers may add a free-text comment.
        is_default: Whether this is the experiment's default question.

    Returns:
        The updated :py:class:`ReviewQuestion`.
    """
    name = _question(experiment_id, question_id)
    body: dict[str, Any] = {"name": name}
    mask: list[str] = []
    _set_if_not_none(body, mask, "title", title)
    _set_if_not_none(body, mask, "instruction", instruction)
    if input is not None:
        _set_if_not_none(body, mask, input._FIELD, input.to_dict())
    _set_if_not_none(body, mask, "enable_comment", enable_comment)
    _set_if_not_none(body, mask, "is_default", is_default)
    _require_update(mask)
    resp = _client.call("PATCH", name, json=body, params={"update_mask": ",".join(mask)})
    return ReviewQuestion.from_dict(resp)


@experimental(version="3.17.1")
def delete_review_question(experiment_id: str, question_id: str) -> None:
    """
    Delete a review question.

    Args:
        experiment_id: The experiment that owns the question.
        question_id: The question to delete.
    """
    _client.call("DELETE", _question(experiment_id, question_id))


# ---- Review queues ----


@experimental(version="3.17.1")
def create_review_queue(
    experiment_id: str,
    display_name: str,
    *,
    queue_type: ReviewQueueType | Literal["USER", "CUSTOM"] = ReviewQueueType.CUSTOM,
    question_ids: list[str] | None = None,
) -> ReviewQueue:
    """
    Create a review queue in an experiment.

    Creating a ``USER`` queue is idempotent: if the user's queue already exists, it is
    returned.

    Args:
        experiment_id: The experiment that owns the queue.
        display_name: Human-readable name. For a ``USER`` queue, the reviewer.
        queue_type: ``"USER"`` or ``"CUSTOM"`` (default).
        question_ids: For a ``CUSTOM`` queue, the questions to pin. A ``USER`` queue
            uses all of the experiment's questions.

    Returns:
        The created :py:class:`ReviewQueue`.
    """
    queue_type = ReviewQueueType(queue_type)
    body: dict[str, Any] = {"display_name": display_name, "queue_type": queue_type.value}
    if queue_type == ReviewQueueType.USER:
        # The server takes a USER queue's reviewer from `owner` and ignores
        # `display_name`; without `owner` it creates the caller's own queue.
        body["owner"] = display_name
    if question_ids:
        body["question_names"] = [_question(experiment_id, q) for q in question_ids]
    resp = _client.call("POST", f"{_experiment(experiment_id)}/reviewQueues", json=body)
    return ReviewQueue.from_dict(resp)


@experimental(version="3.17.1")
def get_review_queue(experiment_id: str, queue_id: str) -> ReviewQueue:
    """
    Get a review queue by ID.

    Args:
        experiment_id: The experiment that owns the queue.
        queue_id: The queue to get.

    Returns:
        The :py:class:`ReviewQueue`.
    """
    return ReviewQueue.from_dict(_client.call("GET", _queue(experiment_id, queue_id)))


@experimental(version="3.17.1")
def list_review_queues(
    experiment_id: str,
    *,
    filter_string: str | None = None,
    max_results: int | None = None,
    page_token: str | None = None,
) -> PagedList[ReviewQueue]:
    """
    List an experiment's review queues.

    Args:
        experiment_id: The experiment to list.
        filter_string: Optional filter, e.g. ``"queue_type=USER"``.
        max_results: Page size.
        page_token: Continuation token from a previous call.

    Returns:
        A :py:class:`PagedList` of :py:class:`ReviewQueue`.
    """
    resp = _client.call(
        "GET",
        f"{_experiment(experiment_id)}/reviewQueues",
        params={"filter": filter_string, "page_size": max_results, "page_token": page_token},
    )
    return PagedList(
        [ReviewQueue.from_dict(q) for q in resp.get("review_queues", [])],
        resp.get("next_page_token"),
    )


@experimental(version="3.17.1")
def update_review_queue(
    experiment_id: str,
    queue_id: str,
    *,
    display_name: str | None = None,
    owner: str | None = None,
    question_ids: list[str] | None = None,
) -> ReviewQueue:
    """
    Update a review queue. Only the fields that are passed are changed.

    Args:
        experiment_id: The experiment that owns the queue.
        queue_id: The queue to update.
        display_name: New human-readable name.
        owner: New owner. Requires MANAGE permission on the experiment.
        question_ids: For a ``CUSTOM`` queue, the questions to pin. Replaces the
            current set.

    Returns:
        The updated :py:class:`ReviewQueue`.
    """
    name = _queue(experiment_id, queue_id)
    body: dict[str, Any] = {"name": name}
    mask: list[str] = []
    _set_if_not_none(body, mask, "display_name", display_name)
    _set_if_not_none(body, mask, "owner", owner)
    if question_ids is not None:
        _set_if_not_none(
            body,
            mask,
            "question_names",
            [_question(experiment_id, q) for q in question_ids],
        )
    _require_update(mask)
    resp = _client.call("PATCH", name, json=body, params={"update_mask": ",".join(mask)})
    return ReviewQueue.from_dict(resp)


@experimental(version="3.17.1")
def delete_review_queue(experiment_id: str, queue_id: str) -> None:
    """
    Delete a review queue. Answers already recorded on its items are unaffected.

    Args:
        experiment_id: The experiment that owns the queue.
        queue_id: The queue to delete.
    """
    _client.call("DELETE", _queue(experiment_id, queue_id))


# ---- Members ----


@experimental(version="3.17.1")
def add_review_queue_member(experiment_id: str, queue_id: str, user: str) -> ReviewQueueMember:
    """
    Assign a reviewer to a queue.

    Args:
        experiment_id: The experiment that owns the queue.
        queue_id: The queue to assign the reviewer to.
        user: The reviewer, e.g. an email address.

    Returns:
        The added :py:class:`ReviewQueueMember`.
    """
    resp = _client.call("POST", f"{_queue(experiment_id, queue_id)}/members", json={"user": user})
    return ReviewQueueMember.from_dict(resp)


@experimental(version="3.17.1")
def remove_review_queue_member(experiment_id: str, queue_id: str, user: str) -> None:
    """
    Unassign a reviewer from a queue.

    Args:
        experiment_id: The experiment that owns the queue.
        queue_id: The queue to unassign the reviewer from.
        user: The reviewer to unassign.
    """
    _client.call(
        "DELETE",
        f"{_queue(experiment_id, queue_id)}/members/{quote(user, safe='@')}",
        params={"parent": _queue(experiment_id, queue_id)},
    )


@experimental(version="3.17.1")
def list_review_queue_members(
    experiment_id: str,
    queue_id: str,
    *,
    max_results: int | None = None,
    page_token: str | None = None,
) -> PagedList[ReviewQueueMember]:
    """
    List the reviewers assigned to a queue.

    Args:
        experiment_id: The experiment that owns the queue.
        queue_id: The queue to list.
        max_results: Page size.
        page_token: Continuation token from a previous call.

    Returns:
        A :py:class:`PagedList` of :py:class:`ReviewQueueMember`.
    """
    resp = _client.call(
        "GET",
        f"{_queue(experiment_id, queue_id)}/members",
        params={"page_size": max_results, "page_token": page_token},
    )
    return PagedList(
        [ReviewQueueMember.from_dict(m) for m in resp.get("review_queue_members", [])],
        resp.get("next_page_token"),
    )


# ---- Items ----


def _trace_item(trace_id: str) -> dict[str, Any]:
    # Send only the bare trace ID: the server resolves the trace location from the
    # experiment, and including it would make the item distinct from the same trace
    # attached without one (e.g. from the UI).
    _, tid = parse_trace_id_v4(trace_id)
    return {"kind": ItemKind.V4_TRACE.value, "v4_trace": {"trace_id": tid}}


@experimental(version="3.17.1")
def add_review_queue_items(
    experiment_id: str,
    queue_id: str,
    *,
    trace_ids: list[str] | None = None,
    dataset_id: str | None = None,
    dataset_record_ids: list[str] | None = None,
) -> list[ReviewQueueItem]:
    """
    Attach traces and/or dataset records to a queue.

    Idempotent per item: re-attaching an item keeps its existing status.

    Args:
        experiment_id: The experiment that owns the queue.
        queue_id: The queue to attach to.
        trace_ids: Trace IDs to attach, either ``tr-<id>`` or, for traces stored in
            Unity Catalog, ``trace:/<location>/<id>``.
        dataset_id: Unity Catalog table ID of the dataset that owns ``dataset_record_ids``.
        dataset_record_ids: Dataset record IDs to attach. Requires ``dataset_id``.

    Returns:
        The resulting list of :py:class:`ReviewQueueItem`.
    """
    if dataset_record_ids and dataset_id is None:
        raise MlflowException.invalid_parameter_value(
            "`dataset_id` is required when `dataset_record_ids` is set."
        )
    items = [_trace_item(t) for t in trace_ids or []]
    items += [
        {
            "kind": ItemKind.DATASET_RECORD.value,
            "dataset_record": {"dataset_id": dataset_id, "dataset_record_id": r},
        }
        for r in dataset_record_ids or []
    ]
    if not items:
        raise MlflowException.invalid_parameter_value(
            "Specify `trace_ids` and/or `dataset_record_ids`."
        )
    parent = _queue(experiment_id, queue_id)
    resp = _client.call(
        "POST",
        f"{parent}/items:batchCreate",
        json={"requests": [{"parent": parent, "review_queue_item": i} for i in items]},
    )
    return [ReviewQueueItem.from_dict(i) for i in resp.get("review_queue_items", [])]


@experimental(version="3.17.1")
def remove_review_queue_items(experiment_id: str, queue_id: str, item_ids: list[str]) -> None:
    """
    Detach items from a queue. Items that are not attached are ignored.

    Args:
        experiment_id: The experiment that owns the queue.
        queue_id: The queue to detach from.
        item_ids: The item IDs (``ReviewQueueItem.item_id``) to detach.
    """
    parent = _queue(experiment_id, queue_id)
    _client.call(
        "POST",
        f"{parent}/items:batchDelete",
        json={"names": [f"{parent}/items/{i}" for i in item_ids]},
    )


@experimental(version="3.17.1")
def list_review_queue_items(
    experiment_id: str,
    queue_id: str,
    *,
    status: ReviewStatus | Literal["PENDING", "COMPLETE", "DECLINED"] | None = None,
    max_results: int | None = None,
    page_token: str | None = None,
) -> PagedList[ReviewQueueItem]:
    """
    List a queue's items.

    Args:
        experiment_id: The experiment that owns the queue.
        queue_id: The queue to list.
        status: Optional status filter.
        max_results: Page size.
        page_token: Continuation token from a previous call. Reuse the same ``status``.

    Returns:
        A :py:class:`PagedList` of :py:class:`ReviewQueueItem`.
    """
    resp = _client.call(
        "GET",
        f"{_queue(experiment_id, queue_id)}/items",
        params={
            "filter": f"status={ReviewStatus(status).value}" if status is not None else None,
            "page_size": max_results,
            "page_token": page_token,
        },
    )
    return PagedList(
        [ReviewQueueItem.from_dict(i) for i in resp.get("review_queue_items", [])],
        resp.get("next_page_token"),
    )


@experimental(version="3.17.1")
def set_review_queue_item_status(
    experiment_id: str,
    queue_id: str,
    item_id: str,
    status: ReviewStatus | Literal["PENDING", "COMPLETE", "DECLINED"],
) -> ReviewQueueItem:
    """
    Set an item's review status.

    ``completed_by`` is recorded by the server from the caller when moving to
    ``COMPLETE`` or ``DECLINED``, and cleared when moving back to ``PENDING``.

    Args:
        experiment_id: The experiment that owns the queue.
        queue_id: The queue that contains the item.
        item_id: The item to update.
        status: ``"PENDING"``, ``"COMPLETE"``, or ``"DECLINED"``.

    Returns:
        The updated :py:class:`ReviewQueueItem`.
    """
    name = f"{_queue(experiment_id, queue_id)}/items/{item_id}"
    resp = _client.call(
        "PATCH",
        name,
        json={"name": name, "status": ReviewStatus(status).value},
        params={"update_mask": "status"},
    )
    return ReviewQueueItem.from_dict(resp)


@experimental(version="3.17.1")
def resolve_effective_review_questions(
    experiment_id: str,
    queue_id: str,
    *,
    item_kind: ItemKind | Literal["V4_TRACE", "DATASET_RECORD"] | None = None,
) -> list[ReviewQuestion]:
    """
    Return the questions reviewers answer in a queue.

    A ``USER`` queue uses all of the experiment's questions; a ``CUSTOM`` queue uses its
    pinned questions.

    Args:
        experiment_id: The experiment that owns the queue.
        queue_id: The queue to resolve.
        item_kind: Optional item kind. Narrows the result to questions that apply to
            that kind of item.

    Returns:
        A list of :py:class:`ReviewQuestion`.
    """
    resp = _client.call(
        "GET",
        f"{_queue(experiment_id, queue_id)}/effectiveQuestions:resolve",
        params={"item_kind": ItemKind(item_kind).value if item_kind is not None else None},
    )
    return [ReviewQuestion.from_dict(q) for q in resp.get("review_questions", [])]
