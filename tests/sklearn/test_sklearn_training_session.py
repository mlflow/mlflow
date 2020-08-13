import pytest

from mlflow.sklearn import _SklearnTrainingSession


class Parent:
    pass


class Child:
    pass


def assert_session_stack(classes):
    assert len(_SklearnTrainingSession._session_stack) == len(classes)

    for idx, (sess, (clazz, parent_clazz)) in enumerate(
        zip(_SklearnTrainingSession._session_stack, classes)
    ):
        assert sess.clazz == clazz
        if idx == 0:
            assert sess._parent is None
        else:
            assert sess._parent.clazz == parent_clazz


@pytest.fixture(params=[True, False])
def allow_children(request):
    return request.param


def test_only_root_session(allow_children):
    with _SklearnTrainingSession(Parent, allow_children=allow_children) as p:
        assert p.should_log()

        assert_session_stack([(Parent, None)])
    assert_session_stack([])


def test_nested_once(allow_children):
    with _SklearnTrainingSession(Parent, allow_children=allow_children) as p:
        assert_session_stack([(Parent, None)])

        with _SklearnTrainingSession(Child) as c:
            assert p.should_log()
            assert c.should_log() == allow_children

            assert_session_stack([(Parent, None), (Child, Parent)])
        assert_session_stack([(Parent, None)])
    assert_session_stack([])
