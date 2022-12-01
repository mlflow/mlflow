import pytest

from mlflow.utils.autologging_utils import _get_new_training_session_class

_TrainingSession = _get_new_training_session_class()


class Parent:
    pass


class Child:
    pass


class Grandchild:
    pass


parent = Parent()
child = Child()
grand_child = Grandchild()


@pytest.fixture(autouse=True)
def clear_session_stack():
    yield
    _TrainingSession._session_stack.clear()


def assert_session_stack(estimators):
    assert len(_TrainingSession._session_stack) == len(estimators)

    for idx, (sess, (parent_estimator, estimator)) in enumerate(
        zip(_TrainingSession._session_stack, estimators)
    ):
        assert sess.estimator is estimator
        if idx == 0:
            assert sess._parent is None
        else:
            assert sess._parent.estimator is parent_estimator


@pytest.fixture(params=[True, False])
def allow_children(request):
    return request.param


def test_should_log_always_returns_true_in_root_session(allow_children):
    with _TrainingSession(parent, allow_children=allow_children) as p:
        assert_session_stack([(None, parent)])
        assert p.should_enable_patch()

    assert_session_stack([])


def test_nested_sessions(allow_children):
    with _TrainingSession(parent, allow_children=allow_children) as p:
        assert_session_stack([(None, parent)])

        with _TrainingSession(child, allow_children=True) as c:
            assert_session_stack([(None, parent), (parent, child)])
            assert p.should_enable_patch()
            assert c.should_enable_patch() == allow_children

        assert_session_stack([(None, parent)])
    assert_session_stack([])


def test_session_is_active():
    assert not _TrainingSession.is_active()
    with _TrainingSession(parent, allow_children=True):
        assert _TrainingSession.is_active()

        with _TrainingSession(child, allow_children=False):
            assert _TrainingSession.is_active()

        assert _TrainingSession.is_active()

    assert_session_stack([])
    assert not _TrainingSession.is_active()


def test_parent_session_overrides_child_session():
    with _TrainingSession(parent, allow_children=False) as p:
        assert_session_stack([(None, parent)])

        with _TrainingSession(child, allow_children=True) as c:
            assert_session_stack([(None, parent), (parent, child)])

            with _TrainingSession(grand_child, allow_children=True) as g:
                assert_session_stack([(None, parent), (parent, child), (child, grand_child)])

                assert p.should_enable_patch()
                assert not c.should_enable_patch()
                assert not g.should_enable_patch()

            assert_session_stack([(None, parent), (parent, child)])
        assert_session_stack([(None, parent)])
    assert_session_stack([])


def test_should_log_returns_false_when_parrent_session_has_the_same_estimator():
    # This test case corresponds to when Pipeline.fit() calls Transformer.fit_transform()
    # which calls Transformer.fit()
    with _TrainingSession(parent, allow_children=True) as p:
        assert_session_stack([(None, parent)])

        with _TrainingSession(child, allow_children=True) as c1:
            assert_session_stack([(None, parent), (parent, child)])

            with _TrainingSession(child, allow_children=True) as c2:
                assert_session_stack([(None, parent), (parent, child), (child, child)])

                assert p.should_enable_patch()
                assert p.should_log()
                assert c1.should_enable_patch()
                assert c1.should_log()
                assert not c2.should_enable_patch()
                assert c2.should_log()

            assert_session_stack([(None, parent), (parent, child)])
        assert_session_stack([(None, parent)])
    assert_session_stack([])
