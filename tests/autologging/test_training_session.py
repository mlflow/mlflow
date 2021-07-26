import pytest

from mlflow.utils.autologging_utils import _get_new_training_session_class

_TrainingSession = _get_new_training_session_class()


class Parent:
    pass


class Child:
    pass


class Grandchild:
    pass


@pytest.fixture(autouse=True)
def clear_session_stack():
    yield
    _TrainingSession._session_stack.clear()


def assert_session_stack(classes):
    assert len(_TrainingSession._session_stack) == len(classes)

    for idx, (sess, (parent_clazz, clazz)) in enumerate(
        zip(_TrainingSession._session_stack, classes)
    ):
        assert sess.clazz == clazz
        if idx == 0:
            assert sess._parent is None
        else:
            assert sess._parent.clazz == parent_clazz


@pytest.fixture(params=[True, False])
def allow_children(request):
    return request.param


def test_should_log_always_returns_true_in_root_session(allow_children):
    with _TrainingSession(Parent, allow_children=allow_children) as p:
        assert_session_stack([(None, Parent)])
        assert p.should_log()

    assert_session_stack([])


def test_nested_sessions(allow_children):
    with _TrainingSession(Parent, allow_children=allow_children) as p:
        assert_session_stack([(None, Parent)])

        with _TrainingSession(Child, allow_children=True) as c:
            assert_session_stack([(None, Parent), (Parent, Child)])
            assert p.should_log()
            assert c.should_log() == allow_children

        assert_session_stack([(None, Parent)])
    assert_session_stack([])


def test_session_is_active():
    assert not _TrainingSession.is_active()
    with _TrainingSession(Parent, allow_children=True):
        assert _TrainingSession.is_active()

        with _TrainingSession(Child, allow_children=False):
            assert _TrainingSession.is_active()

        assert _TrainingSession.is_active()

    assert_session_stack([])
    assert not _TrainingSession.is_active()


def test_parent_session_overrides_child_session():
    with _TrainingSession(Parent, allow_children=False) as p:
        assert_session_stack([(None, Parent)])

        with _TrainingSession(Child, allow_children=True) as c:
            assert_session_stack([(None, Parent), (Parent, Child)])

            with _TrainingSession(Grandchild, allow_children=True) as g:
                assert_session_stack([(None, Parent), (Parent, Child), (Child, Grandchild)])

                assert p.should_log()
                assert not c.should_log()
                assert not g.should_log()

            assert_session_stack([(None, Parent), (Parent, Child)])
        assert_session_stack([(None, Parent)])
    assert_session_stack([])


def test_should_log_returns_false_when_parrent_session_has_the_same_class():
    # This test case corresponds to when Pipeline.fit() calls Transformer.fit_transform()
    # which calls Transformer.fit()
    with _TrainingSession(Parent, allow_children=True) as p:
        assert_session_stack([(None, Parent)])

        with _TrainingSession(Child, allow_children=True) as c1:
            assert_session_stack([(None, Parent), (Parent, Child)])

            with _TrainingSession(Child, allow_children=True) as c2:
                assert_session_stack([(None, Parent), (Parent, Child), (Child, Child)])

                assert p.should_log()
                assert c1.should_log()
                assert not c2.should_log()

            assert_session_stack([(None, Parent), (Parent, Child)])
        assert_session_stack([(None, Parent)])
    assert_session_stack([])
