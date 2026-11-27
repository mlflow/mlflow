"""Review queues on Databricks.

A Databricks-native client for review queues. Unlike :mod:`mlflow.genai.review_queues`
(which targets the MLflow tracking server), the functions here follow the Databricks
review queue REST API: questions, queues, members, and items are separate resources
nested under an experiment, and every function takes the ``experiment_id`` explicitly.
They require a Databricks tracking URI.

The member functions also accept a :py:class:`mlflow.genai.LabelingSession` in place of
``experiment_id`` and ``queue_id``, so code written for labeling sessions can use them too.
"""

import json
import sys
from typing import TYPE_CHECKING, Any, Iterable, Literal, overload
from urllib.parse import quote

from mlflow.entities import Trace
from mlflow.entities.trace_location import TraceLocationType
from mlflow.exceptions import MlflowException
from mlflow.genai.databricks.review_queues import _client, _grants
from mlflow.genai.databricks.review_queues.entities import (
    ItemKind,
    LabelingSessionMigrationResult,
    QuestionInput,
    QuestionType,
    ReviewQuestion,
    ReviewQueue,
    ReviewQueueItem,
    ReviewQueueMember,
    ReviewQueueType,
    ReviewStatus,
)
from mlflow.genai.labeling import stores as labeling_stores
from mlflow.genai.labeling.labeling import LabelingSession
from mlflow.protos.databricks_pb2 import INVALID_STATE, RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracing.utils import parse_trace_id_v4
from mlflow.utils.annotations import experimental
from mlflow.utils.mlflow_tags import MLFLOW_EXPERIMENT_DATABRICKS_TRACE_DESTINATION_PATH

if TYPE_CHECKING:
    import pandas as pd


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
    members: list[str] | None = None,
    grant_access: bool = True,
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
        members: Reviewers to assign once the queue is created, e.g. email addresses.
            They're added one at a time after the queue is created, so if one can't be
            added, the queue still exists and the error names it. They're also granted
            access, as described in :py:func:`add_review_queue_member`.
        grant_access: Whether to grant ``members`` the access they need to review the
            queue. Defaults to ``True``.

    Returns:
        The created :py:class:`ReviewQueue`.
    """
    member_users = _unique_users(members) if members is not None else []
    queue_type = ReviewQueueType(queue_type)
    body: dict[str, Any] = {"display_name": display_name, "queue_type": queue_type.value}
    if queue_type == ReviewQueueType.USER:
        # The server takes a USER queue's reviewer from `owner` and ignores
        # `display_name`; without `owner` it creates the caller's own queue.
        body["owner"] = display_name
    if question_ids:
        body["question_names"] = [_question(experiment_id, q) for q in question_ids]
    resp = _client.call("POST", f"{_experiment(experiment_id)}/reviewQueues", json=body)
    queue = ReviewQueue.from_dict(resp)
    if member_users:
        try:
            # A new CUSTOM queue has no items yet, so there are no dataset tables to grant. A USER
            # queue that already exists is returned as it is, and may hold dataset records.
            _add_queue_members(
                experiment_id,
                queue.queue_id,
                member_users,
                grant_access=grant_access,
                new_queue=queue_type == ReviewQueueType.CUSTOM,
            )
        except MlflowException as e:
            raise MlflowException(
                f"Created review queue {queue.queue_id!r}, but couldn't add all its members. "
                f"{e.message}",
                error_code=ErrorCode.Value(e.error_code),
            ) from e
    return queue


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
#
# Each member function also accepts a `LabelingSession` in place of `experiment_id` and
# `queue_id`. For a session, it reads and writes the session's assigned users through the
# labeling session SDK, so the session keeps behaving as it does today, including the access
# that SDK grants to assigned users. It never changes the queue a session was migrated to.
# For a queue, reviewers who are added are also granted access, as the review queue UI does,
# unless `grant_access=False`.


def _require(value: Any, name: str) -> Any:
    if value is None:
        raise MlflowException.invalid_parameter_value(f"`{name}` is required.")
    return value


def _unique_users(users: list[str]) -> list[str]:
    if isinstance(users, str):
        raise MlflowException.invalid_parameter_value(
            f"`users` must be a list of users, not a string: {users!r}."
        )
    return list(dict.fromkeys(users))


def _reject_queue_id_for_session(function: str, queue_id: Any, keyword: str | None) -> None:
    # A session has no `queue_id`, so a second positional argument is a misplaced keyword,
    # e.g. `add_review_queue_member(session, "alice@example.com")`.
    if queue_id is not None:
        hint = (
            f" Pass `{keyword}` by keyword: `{function}(session, {keyword}=...)`."
            if keyword
            else ""
        )
        raise MlflowException.invalid_parameter_value(
            f"`{function}` takes no `queue_id` when given a `LabelingSession`.{hint}"
        )


def _current_labeling_session(session: LabelingSession) -> LabelingSession:
    # The caller's object can be out of date, so read the session again before using its users.
    store = labeling_stores._get_labeling_store()
    for current in store.get_labeling_sessions(session.experiment_id):
        if current.labeling_session_id == session.labeling_session_id:
            return current
    raise MlflowException(
        f"Labeling session {session.labeling_session_id!r} was not found in experiment "
        f"{session.experiment_id!r}.",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )


def _add_session_members(session: LabelingSession, users: list[str]) -> None:
    current = _current_labeling_session(session)
    if new_users := [u for u in users if u not in current.assigned_users]:
        current.set_assigned_users(current.assigned_users + new_users)


def _reject_grant_access_for_session(function: str, grant_access: bool) -> None:
    if not grant_access:
        raise MlflowException.invalid_parameter_value(
            f"`{function}` can't take `grant_access=False` with a `LabelingSession`: the labeling "
            "session SDK always grants its assigned users access."
        )


def _grant_queue_members(
    experiment_id: str, queue_id: str, users: list[str], *, new_queue: bool = False
) -> None:
    if users:
        _grants.grant_reviewer_access(
            experiment_id, users, queue_id=None if new_queue else queue_id
        )


def _add_queue_member(experiment_id: str, queue_id: str, user: str) -> ReviewQueueMember:
    resp = _client.call("POST", f"{_queue(experiment_id, queue_id)}/members", json={"user": user})
    return ReviewQueueMember.from_dict(resp)


def _add_queue_members(
    experiment_id: str,
    queue_id: str,
    users: list[str],
    *,
    grant_access: bool,
    new_queue: bool = False,
) -> list[ReviewQueueMember]:
    added: list[ReviewQueueMember] = []
    added_users: list[str] = []
    try:
        for user in users:
            try:
                added.append(_add_queue_member(experiment_id, queue_id, user))
            except MlflowException as e:
                raise MlflowException(
                    f"Couldn't add {user!r} to review queue {queue_id!r}. Members added before "
                    f"this one: {added_users}. Error: {e.message}",
                    error_code=ErrorCode.Value(e.error_code),
                ) from e
            added_users.append(user)
    finally:
        # Reviewers who were added get access even if a later one couldn't be added.
        if grant_access:
            _grant_queue_members(experiment_id, queue_id, added_users, new_queue=new_queue)
    return added


def _remove_queue_member(experiment_id: str, queue_id: str, user: str) -> None:
    _client.call(
        "DELETE",
        f"{_queue(experiment_id, queue_id)}/members/{quote(user, safe='@')}",
        params={"parent": _queue(experiment_id, queue_id)},
    )


def _list_queue_members(
    experiment_id: str, queue_id: str, max_results: int | None, page_token: str | None
) -> PagedList[ReviewQueueMember]:
    resp = _client.call(
        "GET",
        f"{_queue(experiment_id, queue_id)}/members",
        params={"page_size": max_results, "page_token": page_token},
    )
    return PagedList(
        [ReviewQueueMember.from_dict(m) for m in resp.get("review_queue_members", [])],
        resp.get("next_page_token"),
    )


def _list_all_queue_members(experiment_id: str, queue_id: str) -> list[ReviewQueueMember]:
    members: list[ReviewQueueMember] = []
    page_token = None
    while True:
        page = _list_queue_members(experiment_id, queue_id, None, page_token)
        members.extend(page)
        if not page.token:
            return members
        page_token = page.token


@overload
def add_review_queue_member(
    experiment_id: str, queue_id: str, user: str, *, grant_access: bool = True
) -> ReviewQueueMember: ...


@overload
def add_review_queue_member(
    labeling_session: LabelingSession, /, *, user: str
) -> ReviewQueueMember: ...


@experimental(version="3.17.1")
def add_review_queue_member(
    experiment_id: str | LabelingSession,
    queue_id: str | None = None,
    user: str | None = None,
    *,
    grant_access: bool = True,
) -> ReviewQueueMember:
    """
    Assign a reviewer to a queue or a labeling session.

    Assigning a reviewer who is already assigned changes nothing.

    Unless ``grant_access=False``, the reviewer is also granted the access they need to review
    the queue, as the review queue UI does:

    - ``CAN_READ`` on the experiment when its traces are stored in Unity Catalog, otherwise
      ``CAN_EDIT``. A reviewer who already has that level or higher is left alone.
    - In Unity Catalog, ``SELECT`` on the experiment's trace spans and logs tables, ``SELECT``
      and ``MODIFY`` on its trace annotation table and on the tables of datasets whose records
      are in the queue, and ``USE CATALOG`` and ``USE SCHEMA`` on the objects above them.

    Grants are made as the caller, so they need ``CAN_MANAGE`` on the experiment and
    ``MANAGE`` on those Unity Catalog objects. A grant that fails is logged as a warning and
    never fails the call. The caller isn't granted anything, and if they can't be identified,
    nobody is.

    To assign a reviewer to a :py:class:`mlflow.genai.LabelingSession`, pass the session in
    place of ``experiment_id`` and ``queue_id``, and pass ``user`` by keyword. The reviewer is
    added to the session's assigned users with
    :py:meth:`mlflow.genai.LabelingSession.set_assigned_users`, which also grants them access
    as it does today. That reads the session's users and then writes them back, so a change
    someone else makes to them at the same moment can be lost.

    .. code-block:: python

        import mlflow
        from mlflow.genai.databricks.review_queues import add_review_queue_member

        add_review_queue_member("123", "rq-1", "alice@example.com")

        session = mlflow.genai.get_labeling_session(run_id="<run-id>")
        add_review_queue_member(session, user="alice@example.com")

    Args:
        experiment_id: The experiment that owns the queue, or a labeling session.
        queue_id: The queue to assign the reviewer to. Omit it for a labeling session.
        user: The reviewer, e.g. an email address.
        grant_access: Whether to grant the reviewer the access they need to review the queue.
            Defaults to ``True``. A labeling session always grants its assigned users access,
            so it can't be ``False`` for one.

    Returns:
        The added :py:class:`ReviewQueueMember`. For a labeling session, only ``user`` is set.
    """
    if isinstance(experiment_id, LabelingSession):
        _reject_queue_id_for_session("add_review_queue_member", queue_id, "user")
        _reject_grant_access_for_session("add_review_queue_member", grant_access)
        user = _require(user, "user")
        _add_session_members(experiment_id, [user])
        return ReviewQueueMember(user=user)
    queue_id = _require(queue_id, "queue_id")
    user = _require(user, "user")
    member = _add_queue_member(experiment_id, queue_id, user)
    if grant_access:
        _grant_queue_members(experiment_id, queue_id, [user])
    return member


@overload
def remove_review_queue_member(experiment_id: str, queue_id: str, user: str) -> None: ...


@overload
def remove_review_queue_member(labeling_session: LabelingSession, /, *, user: str) -> None: ...


@experimental(version="3.17.1")
def remove_review_queue_member(
    experiment_id: str | LabelingSession,
    queue_id: str | None = None,
    user: str | None = None,
) -> None:
    """
    Unassign a reviewer from a queue or a labeling session.

    Unassigning a reviewer who isn't assigned changes nothing.

    To unassign a reviewer from a :py:class:`mlflow.genai.LabelingSession`, pass the session
    in place of ``experiment_id`` and ``queue_id``, and pass ``user`` by keyword. The reviewer
    is removed from the session's assigned users with
    :py:meth:`mlflow.genai.LabelingSession.set_assigned_users`. That reads the session's users
    and then writes them back, so a change someone else makes to them at the same moment can
    be lost.

    Args:
        experiment_id: The experiment that owns the queue, or a labeling session.
        queue_id: The queue to unassign the reviewer from. Omit it for a labeling session.
        user: The reviewer to unassign.
    """
    if isinstance(experiment_id, LabelingSession):
        _reject_queue_id_for_session("remove_review_queue_member", queue_id, "user")
        user = _require(user, "user")
        current = _current_labeling_session(experiment_id)
        if user in current.assigned_users:
            current.set_assigned_users([u for u in current.assigned_users if u != user])
        return
    _remove_queue_member(experiment_id, _require(queue_id, "queue_id"), _require(user, "user"))


@overload
def list_review_queue_members(
    experiment_id: str,
    queue_id: str,
    *,
    max_results: int | None = None,
    page_token: str | None = None,
) -> PagedList[ReviewQueueMember]: ...


@overload
def list_review_queue_members(
    labeling_session: LabelingSession, /
) -> PagedList[ReviewQueueMember]: ...


@experimental(version="3.17.1")
def list_review_queue_members(
    experiment_id: str | LabelingSession,
    queue_id: str | None = None,
    *,
    max_results: int | None = None,
    page_token: str | None = None,
) -> PagedList[ReviewQueueMember]:
    """
    List the reviewers assigned to a queue or a labeling session.

    To list the reviewers of a :py:class:`mlflow.genai.LabelingSession`, pass the session in
    place of ``experiment_id`` and ``queue_id``. Its assigned users are read again, so the
    result is current even if the session object is not, and they all come back in one page.

    Args:
        experiment_id: The experiment that owns the queue, or a labeling session.
        queue_id: The queue to list. Omit it for a labeling session.
        max_results: Page size. Not used for a labeling session.
        page_token: Continuation token from a previous call. Not used for a labeling session.

    Returns:
        A :py:class:`PagedList` of :py:class:`ReviewQueueMember`. For a labeling session, only
        ``user`` is set.
    """
    if isinstance(experiment_id, LabelingSession):
        _reject_queue_id_for_session("list_review_queue_members", queue_id, None)
        if max_results is not None or page_token is not None:
            raise MlflowException.invalid_parameter_value(
                "`max_results` and `page_token` apply only to review queues. A labeling "
                "session's members come back in one page."
            )
        current = _current_labeling_session(experiment_id)
        return PagedList([ReviewQueueMember(user=u) for u in current.assigned_users], None)
    return _list_queue_members(
        experiment_id, _require(queue_id, "queue_id"), max_results, page_token
    )


@overload
def add_review_queue_members(
    experiment_id: str, queue_id: str, users: list[str], *, grant_access: bool = True
) -> list[ReviewQueueMember]: ...


@overload
def add_review_queue_members(
    labeling_session: LabelingSession, /, *, users: list[str]
) -> list[ReviewQueueMember]: ...


@experimental(version="3.19.0")
def add_review_queue_members(
    experiment_id: str | LabelingSession,
    queue_id: str | None = None,
    users: list[str] | None = None,
    *,
    grant_access: bool = True,
) -> list[ReviewQueueMember]:
    """
    Assign several reviewers to a queue or a labeling session.

    Reviewers who are already assigned are left as they are. On a queue, reviewers are
    added one at a time, so if one can't be added, the ones before it stay added, and the
    error names them. Unless ``grant_access=False``, the reviewers who were added are also
    granted the access they need to review the queue, as described in
    :py:func:`add_review_queue_member`.

    To assign reviewers to a :py:class:`mlflow.genai.LabelingSession`, pass the session in
    place of ``experiment_id`` and ``queue_id``, and pass ``users`` by keyword. The reviewers
    are added to the session's assigned users in one
    :py:meth:`mlflow.genai.LabelingSession.set_assigned_users` call, which also grants them
    access as it does today. That reads the session's users and then writes them back, so a
    change someone else makes to them at the same moment can be lost.

    .. code-block:: python

        import mlflow
        from mlflow.genai.databricks.review_queues import add_review_queue_members

        add_review_queue_members("123", "rq-1", ["alice@example.com", "bob@example.com"])

        session = mlflow.genai.get_labeling_session(run_id="<run-id>")
        add_review_queue_members(session, users=["alice@example.com", "bob@example.com"])

    Args:
        experiment_id: The experiment that owns the queue, or a labeling session.
        queue_id: The queue to assign the reviewers to. Omit it for a labeling session.
        users: The reviewers, e.g. email addresses.
        grant_access: Whether to grant the reviewers the access they need to review the
            queue, as described in :py:func:`add_review_queue_member`. Defaults to ``True``.
            A labeling session always grants its assigned users access, so it can't be
            ``False`` for one.

    Returns:
        One :py:class:`ReviewQueueMember` per reviewer in ``users``. For a labeling session,
        only ``user`` is set.
    """
    if isinstance(experiment_id, LabelingSession):
        _reject_queue_id_for_session("add_review_queue_members", queue_id, "users")
        _reject_grant_access_for_session("add_review_queue_members", grant_access)
        unique_users = _unique_users(_require(users, "users"))
        _add_session_members(experiment_id, unique_users)
        return [ReviewQueueMember(user=u) for u in unique_users]
    return _add_queue_members(
        experiment_id,
        _require(queue_id, "queue_id"),
        _unique_users(_require(users, "users")),
        grant_access=grant_access,
    )


@overload
def set_review_queue_members(
    experiment_id: str, queue_id: str, users: list[str], *, grant_access: bool = True
) -> list[ReviewQueueMember]: ...


@overload
def set_review_queue_members(
    labeling_session: LabelingSession, /, *, users: list[str]
) -> list[ReviewQueueMember]: ...


@experimental(version="3.19.0")
def set_review_queue_members(
    experiment_id: str | LabelingSession,
    queue_id: str | None = None,
    users: list[str] | None = None,
    *,
    grant_access: bool = True,
) -> list[ReviewQueueMember]:
    """
    Replace the reviewers assigned to a queue or a labeling session.

    On a queue, the current reviewers are listed, those not in ``users`` are unassigned, and
    then the missing ones are assigned. Unassigning first means a full replacement never goes
    over the queue's reviewer limit. The steps aren't atomic: if one fails, the earlier ones
    stay done, and the error names them. Unless ``grant_access=False``, the newly assigned
    reviewers are also granted the access they need to review the queue, as described in
    :py:func:`add_review_queue_member`. Unassigning a reviewer doesn't revoke any access.

    To set the reviewers of a :py:class:`mlflow.genai.LabelingSession`, pass the session in
    place of ``experiment_id`` and ``queue_id``, and pass ``users`` by keyword. This makes one
    :py:meth:`mlflow.genai.LabelingSession.set_assigned_users` call, which also grants the
    reviewers access as it does today.

    .. code-block:: python

        import mlflow
        from mlflow.genai.databricks.review_queues import set_review_queue_members

        set_review_queue_members("123", "rq-1", ["alice@example.com", "bob@example.com"])

        session = mlflow.genai.get_labeling_session(run_id="<run-id>")
        set_review_queue_members(session, users=["alice@example.com", "bob@example.com"])

    Args:
        experiment_id: The experiment that owns the queue, or a labeling session.
        queue_id: The queue whose reviewers to replace. Omit it for a labeling session.
        users: The reviewers the queue or session should have, e.g. email addresses.
        grant_access: Whether to grant the reviewers the access they need to review the
            queue, as described in :py:func:`add_review_queue_member`. Defaults to ``True``.
            A labeling session always grants its assigned users access, so it can't be
            ``False`` for one.

    Returns:
        The reviewers now assigned, as :py:class:`ReviewQueueMember` objects. For a labeling
        session, only ``user`` is set.
    """
    if isinstance(experiment_id, LabelingSession):
        _reject_queue_id_for_session("set_review_queue_members", queue_id, "users")
        _reject_grant_access_for_session("set_review_queue_members", grant_access)
        updated = experiment_id.set_assigned_users(_unique_users(_require(users, "users")))
        return [ReviewQueueMember(user=u) for u in updated.assigned_users]

    queue_id = _require(queue_id, "queue_id")
    wanted = _unique_users(_require(users, "users"))
    current = {m.user: m for m in _list_all_queue_members(experiment_id, queue_id)}
    removed: list[str] = []
    added: dict[str, ReviewQueueMember] = {}
    try:
        for user in [u for u in current if u not in wanted]:
            _remove_queue_member(experiment_id, queue_id, user)
            removed.append(user)
        for user in [u for u in wanted if u not in current]:
            added[user] = _add_queue_member(experiment_id, queue_id, user)
    except MlflowException as e:
        raise MlflowException(
            f"Setting the members of review queue {queue_id!r} stopped partway. Removed so far: "
            f"{removed}. Added so far: {list(added)}. Error: {e.message}",
            error_code=ErrorCode.Value(e.error_code),
        ) from e
    finally:
        # Reviewers who were added get access even if a later step failed.
        if grant_access:
            _grant_queue_members(experiment_id, queue_id, list(added))
    return [current[u] if u in current else added[u] for u in wanted]


# ---- Items ----


def _trace_item(trace_id: str) -> dict[str, Any]:
    # Send only the bare trace ID: the server resolves the trace location from the
    # experiment, and including it would make the item distinct from the same trace
    # attached without one (e.g. from the UI).
    _, tid = parse_trace_id_v4(trace_id)
    return {"kind": ItemKind.V4_TRACE.value, "v4_trace": {"trace_id": tid}}


# The server rejects a batch of more items than this.
_MAX_ITEMS_PER_REQUEST = 100


def _to_trace(trace: Any) -> Trace:
    if isinstance(trace, Trace):
        return trace
    if isinstance(trace, str):
        try:
            return Trace.from_json(trace)
        except (MlflowException, ValueError, KeyError, TypeError) as e:
            raise MlflowException.invalid_parameter_value(
                f"Couldn't read a trace from {trace[:80]!r}. `traces` takes `Trace` objects or "
                "their JSON; pass trace IDs with `trace_ids`."
            ) from e
    raise MlflowException.invalid_parameter_value(
        f"`traces` takes `Trace` objects or their JSON, got {type(trace).__name__}."
    )


def _trace_ids_from(
    experiment_id: str, traces: "Iterable[Trace] | Iterable[str] | pd.DataFrame"
) -> list[str]:
    # The inputs `LabelingSession.add_traces` takes. A caller passing a DataFrame has
    # already imported pandas, so there's no need to import it (or require it) here.
    if (pandas := sys.modules.get("pandas")) and isinstance(traces, pandas.DataFrame):
        if "trace" not in traces.columns:
            raise MlflowException.invalid_parameter_value(
                "`traces` must have a 'trace' column, like the result of `mlflow.search_traces()`."
            )
        traces = traces["trace"].to_list()
    parsed = [_to_trace(t) for t in traces]
    _check_trace_experiments(experiment_id, parsed)
    return [t.info.trace_id for t in parsed]


def _check_trace_experiments(experiment_id: str, traces: list[Trace]) -> None:
    # Items carry only the bare trace ID, which the server looks up in the queue's experiment,
    # so a trace from anywhere else would become an item with no trace behind it.
    in_uc = []
    for trace in traces:
        match trace.info.trace_location.type:
            case TraceLocationType.MLFLOW_EXPERIMENT:
                if str(trace.info.experiment_id) != str(experiment_id):
                    raise MlflowException.invalid_parameter_value(
                        f"Trace {trace.info.trace_id!r} is in experiment "
                        f"{trace.info.experiment_id!r}, but a review queue can only hold traces "
                        f"from its own experiment, {experiment_id!r}."
                    )
            case TraceLocationType.UC_SCHEMA | TraceLocationType.UC_TABLE_PREFIX:
                in_uc.append(trace)
    if not in_uc:
        return
    experiment = _client.call_api(
        "GET", "/api/2.0/mlflow/experiments/get", params={"experiment_id": experiment_id}
    ).get("experiment", {})
    tags = {tag.get("key"): tag.get("value") for tag in experiment.get("tags", [])}
    experiment_location = tags.get(MLFLOW_EXPERIMENT_DATABRICKS_TRACE_DESTINATION_PATH)
    for trace in in_uc:
        location, _ = parse_trace_id_v4(trace.info.trace_id)
        if location != experiment_location:
            stores = (
                f"which stores its traces in {experiment_location!r}"
                if experiment_location
                else "which doesn't store traces in Unity Catalog"
            )
            raise MlflowException.invalid_parameter_value(
                f"Trace {trace.info.trace_id!r} is stored in {location!r}, but a review queue "
                f"can only hold traces from its own experiment, {experiment_id!r}, {stores}."
            )


def _dataset_id_from_name(name: str) -> str:
    # A dataset's ID is the ID of its Unity Catalog table, as in `databricks-agents`.
    table = quote(name.replace("`", ""), safe=".")
    return _client.call_api("GET", f"/api/2.1/unity-catalog/tables/{table}")["table_id"]


def _dataset_record_ids(dataset_id: str) -> list[str]:
    record_ids: list[str] = []
    page_token = None
    while True:
        resp = _client.call(
            "GET",
            f"datasets/{quote(dataset_id, safe='')}/records",
            params={"page_token": page_token},
        )
        record_ids += [r["dataset_record_id"] for r in resp.get("dataset_records", [])]
        if not (page_token := resp.get("next_page_token")):
            return record_ids


@experimental(version="3.17.1")
def add_review_queue_items(
    experiment_id: str,
    queue_id: str,
    *,
    trace_ids: list[str] | None = None,
    dataset_id: str | None = None,
    dataset_record_ids: list[str] | None = None,
    traces: "Iterable[Trace] | Iterable[str] | pd.DataFrame | None" = None,
    dataset: str | None = None,
) -> list[ReviewQueueItem]:
    """
    Attach traces and/or dataset records to a queue.

    Idempotent per item: re-attaching an item keeps its existing status.

    Items are sent in batches of 100, the most the server accepts at once. If a later batch
    fails, the earlier ones stay attached and the error says how many were; attaching the
    same items again is safe. The server caps how many items a queue holds.

    .. code-block:: python

        import mlflow
        from mlflow.genai.databricks.review_queues import add_review_queue_items

        add_review_queue_items("123", "rq-1", traces=mlflow.search_traces(max_results=10))
        add_review_queue_items("123", "rq-1", dataset="catalog.schema.eval_set")

    Args:
        experiment_id: The experiment that owns the queue.
        queue_id: The queue to attach to.
        trace_ids: Trace IDs to attach, either ``tr-<id>`` or, for traces stored in
            Unity Catalog, ``trace:/<location>/<id>``.
        dataset_id: Unity Catalog table ID of the dataset that owns ``dataset_record_ids``.
        dataset_record_ids: Dataset record IDs to attach. Requires ``dataset_id`` or
            ``dataset``.
        traces: Traces to attach, as :py:class:`mlflow.entities.Trace` objects, their JSON,
            or a pandas DataFrame with a ``trace`` column such as the result of
            :py:func:`mlflow.search_traces`. The same inputs as
            :py:meth:`mlflow.genai.LabelingSession.add_traces`. Each must be from the queue's
            experiment.
        dataset: The name of a dataset, i.e. its Unity Catalog table, in place of
            ``dataset_id``. Without ``dataset_record_ids``, every record in it is attached,
            as :py:meth:`mlflow.genai.LabelingSession.add_dataset` does.

    Returns:
        The resulting list of :py:class:`ReviewQueueItem`.
    """
    if dataset is not None and dataset_id is not None:
        raise MlflowException.invalid_parameter_value(
            "Pass the dataset by `dataset_id` or by name with `dataset`, not both."
        )
    if dataset_record_ids and dataset_id is None and dataset is None:
        raise MlflowException.invalid_parameter_value(
            "`dataset_id` is required when `dataset_record_ids` is set. Pass the dataset's "
            "table ID, or its name with `dataset`."
        )
    items = [_trace_item(t) for t in trace_ids or []]
    if traces is not None:
        items += [_trace_item(t) for t in _trace_ids_from(experiment_id, traces)]
    if dataset is not None:
        dataset_id = _dataset_id_from_name(dataset)
        if dataset_record_ids is None:
            dataset_record_ids = _dataset_record_ids(dataset_id)
    items += [
        {
            "kind": ItemKind.DATASET_RECORD.value,
            "dataset_record": {"dataset_id": dataset_id, "dataset_record_id": r},
        }
        for r in dataset_record_ids or []
    ]
    # The same item twice would count twice against the queue's item cap.
    items = list({json.dumps(i, sort_keys=True): i for i in items}.values())
    if not items:
        if dataset is not None:
            raise MlflowException.invalid_parameter_value(f"Dataset {dataset!r} has no records.")
        raise MlflowException.invalid_parameter_value(
            "Specify `trace_ids` and/or `dataset_record_ids`, or `traces` or `dataset`."
        )
    parent = _queue(experiment_id, queue_id)
    added: list[ReviewQueueItem] = []
    for start in range(0, len(items), _MAX_ITEMS_PER_REQUEST):
        batch = items[start : start + _MAX_ITEMS_PER_REQUEST]
        try:
            resp = _client.call(
                "POST",
                f"{parent}/items:batchCreate",
                json={"requests": [{"parent": parent, "review_queue_item": i} for i in batch]},
            )
        except MlflowException as e:
            if start == 0:
                raise
            raise MlflowException(
                f"Attached {start} of {len(items)} items to review queue {queue_id!r} before "
                f"the server refused the rest. Attaching the same items again is safe. "
                f"Error: {e.message}",
                error_code=ErrorCode.Value(e.error_code),
            ) from e
        added += [ReviewQueueItem.from_dict(i) for i in resp.get("review_queue_items", [])]
    return added


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


# ---- Labeling session migration ----


def _call_migrate_labeling_session(
    experiment_id: str, labeling_session_id: str, *, dry_run: bool, plan_token: str | None
) -> LabelingSessionMigrationResult:
    body: dict[str, Any] = {"labeling_session_id": labeling_session_id, "dry_run": dry_run}
    if plan_token is not None:
        body["plan_token"] = plan_token
    resp = _client.call(
        "POST",
        f"{_experiment(experiment_id)}/reviewQueues:migrateLabelingSession",
        json=body,
    )
    return LabelingSessionMigrationResult.from_dict(resp)


@experimental(version="3.19.0")
def migrate_labeling_session(
    experiment_id: str,
    labeling_session_id: str,
    *,
    dry_run: bool = False,
    plan_token: str | None = None,
) -> LabelingSessionMigrationResult:
    """
    Migrate a labeling session to a ``CUSTOM`` review queue in the same experiment.

    The session's questions, assigned users and items are copied to the queue, and the
    session itself is left in place. Some parts of a session have no review queue
    equivalent, such as text-list questions or the session's agent; they are left out and
    reported as ``WARNING`` issues. Migrating the same session again targets the same queue
    and adds only what changed. The caller needs CAN_MANAGE on the experiment.

    There are three ways to call it:

    - With ``dry_run=True``: return the plan and its issues without writing anything. Unless
      an issue is ``BLOCKING``, the result includes a ``plan_token``.
    - With ``plan_token`` from a dry run: migrate, if the session still matches that plan.
      If it changed, the call raises an ``MlflowException`` with error code ``ABORTED``, and
      you can run the dry run again.
    - With neither: run a dry run, then migrate with its token. If an issue is
      ``BLOCKING``, the call raises an ``MlflowException`` with error code
      ``INVALID_STATE``, listing the issues, and writes nothing. If the session changes
      between the two steps, it raises ``ABORTED``, and you can call it again.

    This feature is rolling out gradually and may not be available in every workspace yet.
    If it isn't, the call raises an ``MlflowException`` with error code ``FEATURE_DISABLED``.

    .. code-block:: python

        from mlflow.genai.databricks.review_queues import migrate_labeling_session

        # Preview the migration, then confirm that exact preview.
        preview = migrate_labeling_session("123", "<labeling-session-id>", dry_run=True)
        for issue in preview.issues:
            print(issue.severity, issue.message)
        # There's no token when an issue blocks the migration.
        if preview.plan_token is not None:
            result = migrate_labeling_session(
                "123", "<labeling-session-id>", plan_token=preview.plan_token
            )
            print(result.review_queue.queue_id)

    To preview and migrate in one call:

    .. code-block:: python

        from mlflow.genai.databricks.review_queues import migrate_labeling_session

        result = migrate_labeling_session("123", "<labeling-session-id>")
        print(result.review_queue.queue_id)

    Args:
        experiment_id: The experiment that owns the labeling session.
        labeling_session_id: The labeling session to migrate. It's the ``labeling_session_id``
            of a :py:class:`mlflow.genai.LabelingSession`, such as one returned by
            :py:func:`mlflow.genai.get_labeling_sessions`.
        dry_run: If ``True``, return the plan without writing anything.
        plan_token: The ``plan_token`` from a dry run, to migrate that plan. Can't be combined
            with ``dry_run=True``.

    Returns:
        A :py:class:`LabelingSessionMigrationResult`. After a migration, its ``review_queue``
        is the queue the session was migrated to.
    """
    if dry_run and plan_token is not None:
        raise MlflowException.invalid_parameter_value(
            "`plan_token` confirms a dry run's plan, so it can't be combined with `dry_run=True`."
        )
    if dry_run or plan_token is not None:
        return _call_migrate_labeling_session(
            experiment_id, labeling_session_id, dry_run=dry_run, plan_token=plan_token
        )

    preview = _call_migrate_labeling_session(
        experiment_id, labeling_session_id, dry_run=True, plan_token=None
    )
    if blocking := preview.blocking_issues:
        reasons = "; ".join(i.message or i.code for i in blocking)
        raise MlflowException(
            f"Labeling session {labeling_session_id!r} can't be migrated: {reasons}",
            error_code=INVALID_STATE,
        )
    return _call_migrate_labeling_session(
        experiment_id, labeling_session_id, dry_run=False, plan_token=preview.plan_token
    )
