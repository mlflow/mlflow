from unittest import mock

import pytest
import requests

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.databricks.review_queues import (
    InputCategorical,
    InputText,
    ItemKind,
    QuestionType,
    ReviewQueueMember,
    ReviewQueueType,
    ReviewStatus,
    _client,
    _grants,
    add_review_queue_items,
    add_review_queue_member,
    add_review_queue_members,
    create_review_question,
    create_review_queue,
    delete_review_question,
    delete_review_queue,
    get_review_question,
    get_review_queue,
    list_review_questions,
    list_review_queue_items,
    list_review_queue_members,
    list_review_queues,
    migrate_labeling_session,
    remove_review_queue_items,
    remove_review_queue_member,
    resolve_effective_review_questions,
    set_review_queue_item_status,
    set_review_queue_members,
    update_review_question,
    update_review_queue,
)
from mlflow.genai.labeling import LabelingSession
from mlflow.genai.labeling.stores import DatabricksLabelingStore
from mlflow.protos.databricks_pb2 import INVALID_STATE

_CALL = "mlflow.genai.databricks.review_queues._client.call"


@pytest.fixture(autouse=True)
def grant_reviewer_access():
    with mock.patch.object(_grants, "grant_reviewer_access") as mock_grant:
        yield mock_grant


_QUESTION = {
    "name": "experiments/1/reviewQuestions/q1",
    "title": "Correct?",
    "type": "FEEDBACK",
    "categorical": {"options": ["yes", "no"], "multi_select": False},
}
_QUEUE = {
    "name": "experiments/1/reviewQueues/rq1",
    "display_name": "My queue",
    "queue_type": "CUSTOM",
}
_ITEM = {
    "name": "experiments/1/reviewQueues/rq1/items/i1",
    "kind": "V4_TRACE",
    "status": "PENDING",
    "v4_trace": {"trace_id": "tr-1"},
}


def test_create_review_question():
    with mock.patch(_CALL, return_value=_QUESTION) as mock_call:
        question = create_review_question(
            "1",
            title="Correct?",
            type="FEEDBACK",
            input=InputCategorical(options=["yes", "no"]),
            instruction="Check it",
            enable_comment=True,
        )

    mock_call.assert_called_once_with(
        "POST",
        "experiments/1/reviewQuestions",
        json={
            "title": "Correct?",
            "type": "FEEDBACK",
            "categorical": {"options": ["yes", "no"], "multi_select": False},
            "instruction": "Check it",
            "enable_comment": True,
        },
    )
    assert question.question_id == "q1"
    assert question.type == QuestionType.FEEDBACK


def test_create_review_question_rejects_invalid_type():
    with mock.patch(_CALL) as mock_call:
        with pytest.raises(ValueError, match="INVALID"):
            create_review_question("1", title="t", type="INVALID", input=InputText())
    mock_call.assert_not_called()


def test_get_review_question():
    with mock.patch(_CALL, return_value=_QUESTION) as mock_call:
        question = get_review_question("1", "q1")
    mock_call.assert_called_once_with("GET", "experiments/1/reviewQuestions/q1")
    assert question.question_id == "q1"


def test_list_review_questions():
    resp = {"review_questions": [_QUESTION], "next_page_token": "tok2"}
    with mock.patch(_CALL, return_value=resp) as mock_call:
        questions = list_review_questions("1", max_results=10, page_token="tok1")
    mock_call.assert_called_once_with(
        "GET",
        "experiments/1/reviewQuestions",
        params={"page_size": 10, "page_token": "tok1"},
    )
    assert [q.question_id for q in questions] == ["q1"]
    assert questions.token == "tok2"


def test_list_review_questions_empty():
    with mock.patch(_CALL, return_value={}) as mock_call:
        questions = list_review_questions("1")
    mock_call.assert_called_once()
    assert list(questions) == []
    assert questions.token is None


def test_update_review_question():
    with mock.patch(_CALL, return_value=_QUESTION) as mock_call:
        update_review_question("1", "q1", title="New", input=InputText(max_length=5))
    mock_call.assert_called_once_with(
        "PATCH",
        "experiments/1/reviewQuestions/q1",
        json={
            "name": "experiments/1/reviewQuestions/q1",
            "title": "New",
            "text": {"max_length": 5},
        },
        params={"update_mask": "title,text"},
    )


def test_update_review_question_requires_a_field():
    with mock.patch(_CALL) as mock_call:
        with pytest.raises(MlflowException, match="at least one field"):
            update_review_question("1", "q1")
    mock_call.assert_not_called()


def test_delete_review_question():
    with mock.patch(_CALL, return_value={}) as mock_call:
        delete_review_question("1", "q1")
    mock_call.assert_called_once_with("DELETE", "experiments/1/reviewQuestions/q1")


def test_create_review_queue():
    with mock.patch(_CALL, return_value=_QUEUE) as mock_call:
        queue = create_review_queue("1", "My queue", question_ids=["q1", "q2"])
    mock_call.assert_called_once_with(
        "POST",
        "experiments/1/reviewQueues",
        json={
            "display_name": "My queue",
            "queue_type": "CUSTOM",
            "question_names": [
                "experiments/1/reviewQuestions/q1",
                "experiments/1/reviewQuestions/q2",
            ],
        },
    )
    assert queue.queue_id == "rq1"
    assert queue.queue_type == ReviewQueueType.CUSTOM


def test_create_review_queue_user_sets_owner():
    with mock.patch(_CALL, return_value={**_QUEUE, "queue_type": "USER"}) as mock_call:
        queue = create_review_queue("1", "alice@example.com", queue_type="USER")
    mock_call.assert_called_once_with(
        "POST",
        "experiments/1/reviewQueues",
        json={
            "display_name": "alice@example.com",
            "queue_type": "USER",
            "owner": "alice@example.com",
        },
    )
    assert queue.queue_type == ReviewQueueType.USER


def test_create_review_queue_with_members():
    responses = [_QUEUE, {"user": "alice@example.com"}, {"user": "bob@example.com"}]
    with mock.patch(_CALL, side_effect=responses) as mock_call:
        queue = create_review_queue(
            "1", "My queue", members=["alice@example.com", "bob@example.com"]
        )
    members_path = "experiments/1/reviewQueues/rq1/members"
    assert mock_call.call_args_list == [
        mock.call(
            "POST",
            "experiments/1/reviewQueues",
            json={"display_name": "My queue", "queue_type": "CUSTOM"},
        ),
        mock.call("POST", members_path, json={"user": "alice@example.com"}),
        mock.call("POST", members_path, json={"user": "bob@example.com"}),
    ]
    assert queue.queue_id == "rq1"


def test_create_review_queue_names_the_queue_when_a_member_fails():
    failure = MlflowException("Not a workspace user.", INVALID_STATE)
    with mock.patch(_CALL, side_effect=[_QUEUE, failure]):
        with pytest.raises(MlflowException, match="Created review queue 'rq1'") as e:
            create_review_queue("1", "My queue", members=["nobody@example.com"])
    assert "Not a workspace user." in e.value.message
    assert e.value.error_code == "INVALID_STATE"


def test_create_review_queue_checks_members_before_creating():
    with mock.patch(_CALL) as mock_call:
        with pytest.raises(MlflowException, match="must be a list of users"):
            create_review_queue("1", "My queue", members="alice@example.com")
    mock_call.assert_not_called()


def test_create_review_queue_grants_its_members(grant_reviewer_access):
    with mock.patch(_CALL, side_effect=[_QUEUE, {"user": "alice@example.com"}]):
        create_review_queue("1", "My queue", members=["alice@example.com"])
    # A new CUSTOM queue has no items, so there are no dataset tables to look up.
    grant_reviewer_access.assert_called_once_with("1", ["alice@example.com"], queue_id=None)


def test_create_review_queue_user_grants_the_datasets_already_in_it(grant_reviewer_access):
    queue = {**_QUEUE, "queue_type": "USER"}
    with mock.patch(_CALL, side_effect=[queue, {"user": "alice@example.com"}]):
        create_review_queue(
            "1", "alice@example.com", queue_type="USER", members=["alice@example.com"]
        )
    # Creating a USER queue that already exists returns it, so it may hold dataset records.
    grant_reviewer_access.assert_called_once_with("1", ["alice@example.com"], queue_id="rq1")


@pytest.mark.parametrize(
    ("kwargs", "responses"),
    [
        ({}, [_QUEUE]),
        ({"members": ["alice@example.com"], "grant_access": False}, [_QUEUE, {"user": "a"}]),
    ],
)
def test_create_review_queue_without_grants(grant_reviewer_access, kwargs, responses):
    with mock.patch(_CALL, side_effect=responses):
        create_review_queue("1", "My queue", **kwargs)
    grant_reviewer_access.assert_not_called()


def test_get_review_queue():
    with mock.patch(_CALL, return_value=_QUEUE) as mock_call:
        queue = get_review_queue("1", "rq1")
    mock_call.assert_called_once_with("GET", "experiments/1/reviewQueues/rq1")
    assert queue.display_name == "My queue"


def test_list_review_queues():
    resp = {"review_queues": [_QUEUE], "next_page_token": "tok"}
    with mock.patch(_CALL, return_value=resp) as mock_call:
        queues = list_review_queues("1", filter_string="queue_type=USER", max_results=5)
    mock_call.assert_called_once_with(
        "GET",
        "experiments/1/reviewQueues",
        params={"filter": "queue_type=USER", "page_size": 5, "page_token": None},
    )
    assert [q.queue_id for q in queues] == ["rq1"]
    assert queues.token == "tok"


def test_update_review_queue():
    with mock.patch(_CALL, return_value=_QUEUE) as mock_call:
        update_review_queue("1", "rq1", owner="alice@example.com", question_ids=[])
    mock_call.assert_called_once_with(
        "PATCH",
        "experiments/1/reviewQueues/rq1",
        json={
            "name": "experiments/1/reviewQueues/rq1",
            "owner": "alice@example.com",
            "question_names": [],
        },
        params={"update_mask": "owner,question_names"},
    )


def test_update_review_queue_requires_a_field():
    with mock.patch(_CALL) as mock_call:
        with pytest.raises(MlflowException, match="at least one field"):
            update_review_queue("1", "rq1")
    mock_call.assert_not_called()


def test_delete_review_queue():
    with mock.patch(_CALL, return_value={}) as mock_call:
        delete_review_queue("1", "rq1")
    mock_call.assert_called_once_with("DELETE", "experiments/1/reviewQueues/rq1")


def test_add_review_queue_member():
    with mock.patch(_CALL, return_value={"user": "alice@example.com"}) as mock_call:
        member = add_review_queue_member("1", "rq1", "alice@example.com")
    mock_call.assert_called_once_with(
        "POST", "experiments/1/reviewQueues/rq1/members", json={"user": "alice@example.com"}
    )
    assert member.user == "alice@example.com"


def test_remove_review_queue_member_quotes_user():
    with mock.patch(_CALL, return_value={}) as mock_call:
        remove_review_queue_member("1", "rq1", "a+b/c@example.com")
    mock_call.assert_called_once_with(
        "DELETE",
        "experiments/1/reviewQueues/rq1/members/a%2Bb%2Fc@example.com",
        params={"parent": "experiments/1/reviewQueues/rq1"},
    )


def test_list_review_queue_members():
    resp = {"review_queue_members": [{"user": "alice@example.com"}]}
    with mock.patch(_CALL, return_value=resp) as mock_call:
        members = list_review_queue_members("1", "rq1", page_token="tok")
    mock_call.assert_called_once_with(
        "GET",
        "experiments/1/reviewQueues/rq1/members",
        params={"page_size": None, "page_token": "tok"},
    )
    assert [m.user for m in members] == ["alice@example.com"]
    assert members.token is None


_STORE = "mlflow.genai.labeling.stores._get_labeling_store"
_MEMBERS = "experiments/1/reviewQueues/rq1/members"


def _session(assigned_users, labeling_session_id="ls1"):
    return LabelingSession(
        name="My session",
        assigned_users=list(assigned_users),
        agent=None,
        label_schemas=[],
        labeling_session_id=labeling_session_id,
        mlflow_run_id="run1",
        review_app_id="app1",
        experiment_id="1",
        url="https://example.com/review",
        enable_multi_turn_chat=False,
        custom_inputs=None,
    )


@pytest.fixture
def labeling_store():
    store = mock.create_autospec(DatabricksLabelingStore, instance=True)
    store.set_session_assigned_users.side_effect = lambda session, users: _session(users)
    with mock.patch(_STORE, return_value=store):
        yield store


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (("1", "rq1", "alice@example.com"), {}),
        (("1", "rq1"), {"user": "alice@example.com"}),
        ((), {"experiment_id": "1", "queue_id": "rq1", "user": "alice@example.com"}),
    ],
)
def test_add_review_queue_member_existing_call_forms(args, kwargs):
    with mock.patch(_CALL, return_value={"user": "alice@example.com"}) as mock_call:
        member = add_review_queue_member(*args, **kwargs)
    mock_call.assert_called_once_with("POST", _MEMBERS, json={"user": "alice@example.com"})
    assert member.user == "alice@example.com"


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (("1", "rq1", "alice@example.com"), {}),
        ((), {"experiment_id": "1", "queue_id": "rq1", "user": "alice@example.com"}),
    ],
)
def test_remove_review_queue_member_existing_call_forms(args, kwargs):
    with mock.patch(_CALL, return_value={}) as mock_call:
        remove_review_queue_member(*args, **kwargs)
    mock_call.assert_called_once_with(
        "DELETE",
        f"{_MEMBERS}/alice@example.com",
        params={"parent": "experiments/1/reviewQueues/rq1"},
    )


def test_list_review_queue_members_existing_keyword_call_form():
    with mock.patch(_CALL, return_value={"review_queue_members": []}) as mock_call:
        list_review_queue_members(experiment_id="1", queue_id="rq1", max_results=5)
    mock_call.assert_called_once_with("GET", _MEMBERS, params={"page_size": 5, "page_token": None})


@pytest.mark.parametrize(
    ("func", "args", "match"),
    [
        (add_review_queue_member, ("1",), "`queue_id` is required"),
        (add_review_queue_member, ("1", "rq1"), "`user` is required"),
        (remove_review_queue_member, ("1", "rq1"), "`user` is required"),
        (list_review_queue_members, ("1",), "`queue_id` is required"),
        (add_review_queue_members, ("1", "rq1"), "`users` is required"),
        (set_review_queue_members, ("1",), "`queue_id` is required"),
    ],
)
def test_member_functions_require_queue_arguments(func, args, match):
    with mock.patch(_CALL) as mock_call:
        with pytest.raises(MlflowException, match=match):
            func(*args)
    mock_call.assert_not_called()


def test_add_review_queue_member_to_labeling_session(labeling_store):
    # The caller's object is stale: someone else has since assigned carol.
    current = _session(["alice@example.com", "carol@example.com"])
    labeling_store.get_labeling_sessions.return_value = [_session([], "ls2"), current]

    with mock.patch(_CALL) as mock_call:
        member = add_review_queue_member(_session(["alice@example.com"]), user="bob@example.com")

    labeling_store.get_labeling_sessions.assert_called_once_with("1")
    labeling_store.set_session_assigned_users.assert_called_once_with(
        current, ["alice@example.com", "carol@example.com", "bob@example.com"]
    )
    mock_call.assert_not_called()
    assert member == ReviewQueueMember(user="bob@example.com")


def test_add_review_queue_member_to_labeling_session_already_assigned(labeling_store):
    labeling_store.get_labeling_sessions.return_value = [_session(["alice@example.com"])]
    member = add_review_queue_member(_session([]), user="alice@example.com")
    labeling_store.set_session_assigned_users.assert_not_called()
    assert member.user == "alice@example.com"


def test_labeling_session_not_found(labeling_store):
    labeling_store.get_labeling_sessions.return_value = [_session([], "ls2")]
    with pytest.raises(MlflowException, match="'ls1' was not found in experiment '1'") as e:
        add_review_queue_member(_session([]), user="alice@example.com")
    assert e.value.error_code == "RESOURCE_DOES_NOT_EXIST"
    labeling_store.set_session_assigned_users.assert_not_called()


@pytest.mark.parametrize(
    ("func", "args", "match"),
    [
        (add_review_queue_member, ("alice@example.com",), "Pass `user` by keyword"),
        (remove_review_queue_member, ("alice@example.com",), "Pass `user` by keyword"),
        (add_review_queue_members, (["alice@example.com"],), "Pass `users` by keyword"),
        (set_review_queue_members, (["alice@example.com"],), "Pass `users` by keyword"),
        (list_review_queue_members, ("rq1",), "takes no `queue_id`"),
    ],
)
def test_labeling_session_rejects_positional_arguments(labeling_store, func, args, match):
    with mock.patch(_CALL) as mock_call:
        with pytest.raises(MlflowException, match=match):
            func(_session([]), *args)
    mock_call.assert_not_called()
    labeling_store.get_labeling_sessions.assert_not_called()
    labeling_store.set_session_assigned_users.assert_not_called()


def test_remove_review_queue_member_from_labeling_session(labeling_store):
    current = _session(["alice@example.com", "bob@example.com"])
    labeling_store.get_labeling_sessions.return_value = [current]

    remove_review_queue_member(_session(["alice@example.com"]), user="bob@example.com")

    labeling_store.set_session_assigned_users.assert_called_once_with(
        current, ["alice@example.com"]
    )


def test_remove_review_queue_member_from_labeling_session_not_assigned(labeling_store):
    labeling_store.get_labeling_sessions.return_value = [_session(["alice@example.com"])]
    remove_review_queue_member(_session([]), user="bob@example.com")
    labeling_store.set_session_assigned_users.assert_not_called()


def test_list_review_queue_members_of_labeling_session(labeling_store):
    labeling_store.get_labeling_sessions.return_value = [
        _session(["alice@example.com", "bob@example.com"])
    ]
    with mock.patch(_CALL) as mock_call:
        members = list_review_queue_members(_session(["alice@example.com"]))
    mock_call.assert_not_called()
    assert [m.user for m in members] == ["alice@example.com", "bob@example.com"]
    assert members.token is None


@pytest.mark.parametrize("kwargs", [{"max_results": 5}, {"page_token": "tok"}])
def test_list_review_queue_members_of_labeling_session_rejects_paging(labeling_store, kwargs):
    with pytest.raises(MlflowException, match="apply only to review queues"):
        list_review_queue_members(_session([]), **kwargs)
    labeling_store.get_labeling_sessions.assert_not_called()


def test_add_review_queue_members_to_queue():
    responses = [{"user": "alice@example.com"}, {"user": "bob@example.com"}]
    with mock.patch(_CALL, side_effect=responses) as mock_call:
        members = add_review_queue_members(
            "1", "rq1", ["alice@example.com", "bob@example.com", "alice@example.com"]
        )
    assert mock_call.call_args_list == [
        mock.call("POST", _MEMBERS, json={"user": "alice@example.com"}),
        mock.call("POST", _MEMBERS, json={"user": "bob@example.com"}),
    ]
    assert [m.user for m in members] == ["alice@example.com", "bob@example.com"]


def test_add_review_queue_members_to_queue_names_members_added_before_a_failure():
    cap = MlflowException("Queue rq1 already has the maximum of 10 reviewers.", INVALID_STATE)
    with mock.patch(_CALL, side_effect=[{"user": "alice@example.com"}, cap]):
        with pytest.raises(MlflowException, match="Couldn't add 'bob@example.com'") as e:
            add_review_queue_members("1", "rq1", ["alice@example.com", "bob@example.com"])
    assert "before this one: ['alice@example.com']" in e.value.message
    assert "maximum of 10 reviewers" in e.value.message
    assert e.value.error_code == "INVALID_STATE"


def test_add_review_queue_members_rejects_a_string():
    with mock.patch(_CALL) as mock_call:
        with pytest.raises(MlflowException, match="must be a list of users, not a string"):
            add_review_queue_members("1", "rq1", "alice@example.com")
    mock_call.assert_not_called()


def test_add_review_queue_members_to_labeling_session(labeling_store):
    current = _session(["alice@example.com"])
    labeling_store.get_labeling_sessions.return_value = [current]

    members = add_review_queue_members(
        _session([]), users=["bob@example.com", "alice@example.com", "carol@example.com"]
    )

    labeling_store.set_session_assigned_users.assert_called_once_with(
        current, ["alice@example.com", "bob@example.com", "carol@example.com"]
    )
    assert [m.user for m in members] == [
        "bob@example.com",
        "alice@example.com",
        "carol@example.com",
    ]


def test_set_review_queue_members_on_queue():
    pages = [
        {
            "review_queue_members": [
                {"user": "alice@example.com", "added_by": "owner@example.com"},
                {"user": "bob@example.com"},
            ],
            "next_page_token": "tok",
        },
        {"review_queue_members": [{"user": "carol@example.com"}]},
    ]
    with mock.patch(_CALL, side_effect=[*pages, {}, {}, {"user": "dave@example.com"}]) as mock_call:
        members = set_review_queue_members("1", "rq1", ["dave@example.com", "alice@example.com"])

    parent = {"parent": "experiments/1/reviewQueues/rq1"}
    assert mock_call.call_args_list == [
        mock.call("GET", _MEMBERS, params={"page_size": None, "page_token": None}),
        mock.call("GET", _MEMBERS, params={"page_size": None, "page_token": "tok"}),
        # Unassign before assigning, so a full replacement never hits the reviewer limit.
        mock.call("DELETE", f"{_MEMBERS}/bob@example.com", params=parent),
        mock.call("DELETE", f"{_MEMBERS}/carol@example.com", params=parent),
        mock.call("POST", _MEMBERS, json={"user": "dave@example.com"}),
    ]
    assert members == [
        ReviewQueueMember(user="dave@example.com"),
        ReviewQueueMember(user="alice@example.com", added_by="owner@example.com"),
    ]


def test_set_review_queue_members_on_queue_names_steps_done_before_a_failure():
    listing = {"review_queue_members": [{"user": "bob@example.com"}]}
    failure = MlflowException("Not allowed.", INVALID_STATE)
    with mock.patch(_CALL, side_effect=[listing, {}, failure]):
        with pytest.raises(MlflowException, match="stopped partway") as e:
            set_review_queue_members("1", "rq1", ["alice@example.com"])
    assert "Removed so far: ['bob@example.com']. Added so far: []" in e.value.message
    assert e.value.error_code == "INVALID_STATE"


def test_set_review_queue_members_on_labeling_session(labeling_store):
    session = _session(["alice@example.com"])
    with mock.patch(_CALL) as mock_call:
        members = set_review_queue_members(
            session, users=["bob@example.com", "carol@example.com", "bob@example.com"]
        )
    labeling_store.set_session_assigned_users.assert_called_once_with(
        session, ["bob@example.com", "carol@example.com"]
    )
    labeling_store.get_labeling_sessions.assert_not_called()
    mock_call.assert_not_called()
    assert [m.user for m in members] == ["bob@example.com", "carol@example.com"]


def test_add_review_queue_member_grants_the_reviewer_access(grant_reviewer_access):
    with mock.patch(_CALL, return_value={"user": "alice@example.com"}):
        add_review_queue_member("1", "rq1", "alice@example.com")
    grant_reviewer_access.assert_called_once_with("1", ["alice@example.com"], queue_id="rq1")


def test_add_review_queue_members_grants_the_reviewers_added(grant_reviewer_access):
    responses = [{"user": "alice@example.com"}, {"user": "bob@example.com"}]
    with mock.patch(_CALL, side_effect=responses):
        add_review_queue_members("1", "rq1", ["alice@example.com", "bob@example.com"])
    grant_reviewer_access.assert_called_once_with(
        "1", ["alice@example.com", "bob@example.com"], queue_id="rq1"
    )


def test_add_review_queue_members_grants_the_ones_added_before_a_failure(grant_reviewer_access):
    failure = MlflowException("Too many reviewers.", INVALID_STATE)
    with mock.patch(_CALL, side_effect=[{"user": "alice@example.com"}, failure]):
        with pytest.raises(MlflowException, match="Couldn't add 'bob@example.com'"):
            add_review_queue_members("1", "rq1", ["alice@example.com", "bob@example.com"])
    grant_reviewer_access.assert_called_once_with("1", ["alice@example.com"], queue_id="rq1")


def test_set_review_queue_members_grants_only_new_reviewers(grant_reviewer_access):
    listing = {"review_queue_members": [{"user": "alice@example.com"}, {"user": "bob@example.com"}]}
    with mock.patch(_CALL, side_effect=[listing, {}, {"user": "carol@example.com"}]):
        set_review_queue_members("1", "rq1", ["alice@example.com", "carol@example.com"])
    grant_reviewer_access.assert_called_once_with("1", ["carol@example.com"], queue_id="rq1")


def test_set_review_queue_members_grants_the_ones_added_before_a_failure(grant_reviewer_access):
    failure = MlflowException("Too many reviewers.", INVALID_STATE)
    listing = {"review_queue_members": []}
    with mock.patch(_CALL, side_effect=[listing, {"user": "alice@example.com"}, failure]):
        with pytest.raises(MlflowException, match="stopped partway"):
            set_review_queue_members("1", "rq1", ["alice@example.com", "bob@example.com"])
    grant_reviewer_access.assert_called_once_with("1", ["alice@example.com"], queue_id="rq1")


@pytest.mark.parametrize(
    ("func", "args"),
    [
        (add_review_queue_member, ("1", "rq1", "alice@example.com")),
        (add_review_queue_members, ("1", "rq1", ["alice@example.com"])),
        (set_review_queue_members, ("1", "rq1", ["alice@example.com"])),
    ],
)
def test_grant_access_false_adds_without_granting(grant_reviewer_access, func, args):
    response = {"user": "alice@example.com", "review_queue_members": []}
    with mock.patch(_CALL, return_value=response) as mock_call:
        func(*args, grant_access=False)
    mock_call.assert_any_call("POST", _MEMBERS, json={"user": "alice@example.com"})
    grant_reviewer_access.assert_not_called()


@pytest.mark.parametrize(
    ("func", "kwargs"),
    [
        (add_review_queue_member, {"user": "alice@example.com"}),
        (add_review_queue_members, {"users": ["alice@example.com"]}),
        (set_review_queue_members, {"users": ["alice@example.com"]}),
    ],
)
def test_labeling_session_rejects_grant_access_false(
    labeling_store, grant_reviewer_access, func, kwargs
):
    with pytest.raises(MlflowException, match="can't take `grant_access=False` with a"):
        func(_session([]), **kwargs, grant_access=False)
    labeling_store.get_labeling_sessions.assert_not_called()
    labeling_store.set_session_assigned_users.assert_not_called()
    grant_reviewer_access.assert_not_called()


def test_labeling_session_grants_only_through_the_labeling_sdk(
    labeling_store, grant_reviewer_access
):
    labeling_store.get_labeling_sessions.return_value = [_session([])]
    add_review_queue_member(_session([]), user="alice@example.com")
    set_review_queue_members(_session([]), users=["bob@example.com"])
    assert labeling_store.set_session_assigned_users.call_count == 2
    grant_reviewer_access.assert_not_called()


def test_add_review_queue_items():
    parent = "experiments/1/reviewQueues/rq1"
    with mock.patch(_CALL, return_value={"review_queue_items": [_ITEM]}) as mock_call:
        items = add_review_queue_items(
            "1",
            "rq1",
            trace_ids=["tr-1", "trace:/catalog.schema/tr-2"],
            dataset_id="ds1",
            dataset_record_ids=["r1"],
        )
    mock_call.assert_called_once_with(
        "POST",
        f"{parent}/items:batchCreate",
        json={
            "requests": [
                {
                    "parent": parent,
                    "review_queue_item": {"kind": "V4_TRACE", "v4_trace": {"trace_id": "tr-1"}},
                },
                {
                    "parent": parent,
                    "review_queue_item": {
                        "kind": "V4_TRACE",
                        "v4_trace": {"trace_id": "tr-2"},
                    },
                },
                {
                    "parent": parent,
                    "review_queue_item": {
                        "kind": "DATASET_RECORD",
                        "dataset_record": {"dataset_id": "ds1", "dataset_record_id": "r1"},
                    },
                },
            ]
        },
    )
    assert [i.item_id for i in items] == ["i1"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "Specify `trace_ids` and/or `dataset_record_ids`"),
        ({"trace_ids": []}, "Specify `trace_ids` and/or `dataset_record_ids`"),
        ({"dataset_record_ids": ["r1"]}, "`dataset_id` is required"),
    ],
)
def test_add_review_queue_items_validation(kwargs, match):
    with mock.patch(_CALL) as mock_call:
        with pytest.raises(MlflowException, match=match):
            add_review_queue_items("1", "rq1", **kwargs)
    mock_call.assert_not_called()


def test_remove_review_queue_items():
    with mock.patch(_CALL, return_value={}) as mock_call:
        remove_review_queue_items("1", "rq1", ["i1", "i2"])
    mock_call.assert_called_once_with(
        "POST",
        "experiments/1/reviewQueues/rq1/items:batchDelete",
        json={
            "names": [
                "experiments/1/reviewQueues/rq1/items/i1",
                "experiments/1/reviewQueues/rq1/items/i2",
            ]
        },
    )


@pytest.mark.parametrize(
    ("status", "expected_filter"),
    [
        (None, None),
        ("COMPLETE", "status=COMPLETE"),
        (ReviewStatus.PENDING, "status=PENDING"),
    ],
)
def test_list_review_queue_items(status, expected_filter):
    resp = {"review_queue_items": [_ITEM], "next_page_token": "tok"}
    with mock.patch(_CALL, return_value=resp) as mock_call:
        items = list_review_queue_items("1", "rq1", status=status, max_results=20)
    mock_call.assert_called_once_with(
        "GET",
        "experiments/1/reviewQueues/rq1/items",
        params={"filter": expected_filter, "page_size": 20, "page_token": None},
    )
    assert [i.item_id for i in items] == ["i1"]
    assert items.token == "tok"


def test_set_review_queue_item_status():
    resp = {**_ITEM, "status": "COMPLETE", "completed_by": "alice@example.com"}
    with mock.patch(_CALL, return_value=resp) as mock_call:
        item = set_review_queue_item_status("1", "rq1", "i1", "COMPLETE")
    name = "experiments/1/reviewQueues/rq1/items/i1"
    mock_call.assert_called_once_with(
        "PATCH",
        name,
        json={"name": name, "status": "COMPLETE"},
        params={"update_mask": "status"},
    )
    assert item.status == ReviewStatus.COMPLETE
    assert item.completed_by == "alice@example.com"


@pytest.mark.parametrize(
    ("item_kind", "expected"),
    [(None, None), ("V4_TRACE", "V4_TRACE"), (ItemKind.DATASET_RECORD, "DATASET_RECORD")],
)
def test_resolve_effective_review_questions(item_kind, expected):
    with mock.patch(_CALL, return_value={"review_questions": [_QUESTION]}) as mock_call:
        questions = resolve_effective_review_questions("1", "rq1", item_kind=item_kind)
    mock_call.assert_called_once_with(
        "GET",
        "experiments/1/reviewQueues/rq1/effectiveQuestions:resolve",
        params={"item_kind": expected},
    )
    assert [q.question_id for q in questions] == ["q1"]


_MIGRATE_PATH = "experiments/1/reviewQueues:migrateLabelingSession"
_MIGRATION_PLAN = {"review_queue": "experiments/1/reviewQueues/rq-ls1", "items_to_add": 2}
_WARNING = {
    "code": "TEXT_LIST_QUESTION_DROPPED",
    "severity": "WARNING",
    "count": 1,
    "message": "1 text-list question(s) are not migrated.",
}
_READY_DRY_RUN = {"plan": _MIGRATION_PLAN, "issues": [_WARNING], "plan_token": "v1.tok"}
_BLOCKED_DRY_RUN = {
    "plan": _MIGRATION_PLAN,
    "issues": [
        _WARNING,
        {
            "code": "TOO_MANY_MEMBERS",
            "severity": "BLOCKING",
            "count": 18,
            "limit": 16,
            "message": "The review queue would have 18 members, more than a review queue allows.",
        },
    ],
}
_MIGRATED = {"plan": _MIGRATION_PLAN, "issues": [_WARNING], "review_queue": _QUEUE}


def test_migrate_labeling_session_dry_run():
    with mock.patch(_CALL, return_value=_READY_DRY_RUN) as mock_call:
        result = migrate_labeling_session("1", "ls1", dry_run=True)

    mock_call.assert_called_once_with(
        "POST", _MIGRATE_PATH, json={"labeling_session_id": "ls1", "dry_run": True}
    )
    assert result.plan.review_queue_id == "rq-ls1"
    assert [w.code for w in result.warnings] == ["TEXT_LIST_QUESTION_DROPPED"]
    assert result.plan_token == "v1.tok"
    assert result.review_queue is None


def test_migrate_labeling_session_with_plan_token():
    with mock.patch(_CALL, return_value=_MIGRATED) as mock_call:
        result = migrate_labeling_session("1", "ls1", plan_token="v1.tok")

    mock_call.assert_called_once_with(
        "POST",
        _MIGRATE_PATH,
        json={"labeling_session_id": "ls1", "dry_run": False, "plan_token": "v1.tok"},
    )
    assert result.review_queue.queue_id == "rq1"


def test_migrate_labeling_session_previews_then_migrates():
    with mock.patch(_CALL, side_effect=[_READY_DRY_RUN, _MIGRATED]) as mock_call:
        result = migrate_labeling_session("1", "ls1")

    assert mock_call.call_args_list == [
        mock.call("POST", _MIGRATE_PATH, json={"labeling_session_id": "ls1", "dry_run": True}),
        mock.call(
            "POST",
            _MIGRATE_PATH,
            json={"labeling_session_id": "ls1", "dry_run": False, "plan_token": "v1.tok"},
        ),
    ]
    assert result.review_queue.queue_id == "rq1"
    assert [w.code for w in result.warnings] == ["TEXT_LIST_QUESTION_DROPPED"]


def test_migrate_labeling_session_raises_when_blocked():
    with mock.patch(_CALL, return_value=_BLOCKED_DRY_RUN) as mock_call:
        with pytest.raises(
            MlflowException, match="18 members, more than a review queue allows"
        ) as e:
            migrate_labeling_session("1", "ls1")

    mock_call.assert_called_once_with(
        "POST", _MIGRATE_PATH, json={"labeling_session_id": "ls1", "dry_run": True}
    )
    assert e.value.error_code == "INVALID_STATE"
    # Only blocking issues explain the failure.
    assert "text-list" not in e.value.message


def test_migrate_labeling_session_rejects_dry_run_with_plan_token():
    with mock.patch(_CALL) as mock_call:
        with pytest.raises(MlflowException, match="can't be combined with `dry_run=True`"):
            migrate_labeling_session("1", "ls1", dry_run=True, plan_token="v1.tok")
    mock_call.assert_not_called()


@pytest.mark.parametrize(
    ("status_code", "error_code"),
    [(403, "FEATURE_DISABLED"), (409, "ABORTED"), (400, "INVALID_STATE")],
)
def test_migrate_labeling_session_surfaces_server_error_codes(status_code, error_code):
    mlflow.set_tracking_uri("databricks")
    response = requests.Response()
    response.status_code = status_code
    response._content = f'{{"error_code": "{error_code}", "message": "Rejected."}}'.encode()
    with (
        mock.patch.object(_client, "get_databricks_host_creds") as mock_creds,
        mock.patch.object(_client, "http_request", return_value=response) as mock_request,
    ):
        with pytest.raises(MlflowException, match="Rejected.") as exc_info:
            migrate_labeling_session("1", "ls1", plan_token="v1.tok")

    mock_creds.assert_called_once_with("databricks")
    mock_request.assert_called_once()
    assert exc_info.value.error_code == error_code
