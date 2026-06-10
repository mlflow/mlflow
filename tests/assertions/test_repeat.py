from __future__ import annotations

import pytest

import mlflow
from mlflow._assertions import pytest_plugin as plugin
from mlflow._assertions import session
from mlflow._assertions.decorator import (
    MLFLOW_TEST_PASS_THRESHOLD_ATTR,
    MLFLOW_TEST_REPEAT_ATTR,
)
from mlflow._assertions.runner import AssertionResult
# --------------------------------------------------------------------------- #
# Decorator: attribute recording + validation                                 #
# --------------------------------------------------------------------------- #


def test_default_threshold_is_strict_majority():
    @mlflow.test(repeat=3)
    def three():
        pass

    @mlflow.test(repeat=5)
    def five():
        pass

    assert getattr(three, MLFLOW_TEST_REPEAT_ATTR) == 3
    assert getattr(three, MLFLOW_TEST_PASS_THRESHOLD_ATTR) == 2
    assert getattr(five, MLFLOW_TEST_PASS_THRESHOLD_ATTR) == 3


def test_bare_marker_is_single_shot():
    @mlflow.test
    def bare():
        pass

    @mlflow.test()
    def called():
        pass

    for fn in (bare, called):
        assert getattr(fn, MLFLOW_TEST_REPEAT_ATTR) == 1
        assert getattr(fn, MLFLOW_TEST_PASS_THRESHOLD_ATTR) == 1


@pytest.mark.parametrize("repeat", [0, -1])
def test_invalid_repeat_raises(repeat):
    with pytest.raises(ValueError, match="must be >= 1"):

        @mlflow.test(repeat=repeat)
        def t():
            pass


@pytest.mark.parametrize("threshold", [0, 4])
def test_threshold_out_of_range_raises(threshold):
    with pytest.raises(ValueError, match="between 1 and repeat"):

        @mlflow.test(repeat=3, pass_threshold=threshold)
        def t():
            pass


# --------------------------------------------------------------------------- #
# _run_repeated: early-exit, threshold, fail synthesis                        #
# --------------------------------------------------------------------------- #


class _FakeItem:
    """Minimal stand-in: ``_run_repeated`` only calls ``item.obj(**args)``."""

    def __init__(self, body):
        self.obj = body


def test_run_repeated_clear_pass_early_exits():
    calls: list[int] = []

    def body():
        calls.append(1)

    err = plugin._run_repeated(_FakeItem(body), {}, "case_clear_pass", None, 3, 2)
    assert err is None
    # Two passes already meet the threshold -> third run is skipped.
    assert len(calls) == 2


def test_run_repeated_clear_fail_early_exits():
    calls: list[int] = []

    def body():
        calls.append(1)
        raise AssertionError("always fails")

    err = plugin._run_repeated(_FakeItem(body), {}, "case_clear_fail", None, 3, 2)
    assert isinstance(err, AssertionError)
    assert "0/2 runs passed (need 2 of 3)" in str(err)
    # Two fails make the threshold unreachable -> third run is skipped.
    assert len(calls) == 2
    # The last underlying failure is chained for debugging context.
    assert isinstance(err.__cause__, AssertionError)
    assert "always fails" in str(err.__cause__)


def test_run_repeated_passes_on_majority_uses_all_runs():
    calls: list[int] = []

    def body():
        n = len(calls)
        calls.append(1)
        if n == 0:
            raise AssertionError("flaky first run")

    err = plugin._run_repeated(_FakeItem(body), {}, "case_flaky", None, 3, 2)
    assert err is None
    # fail, pass, pass -> threshold reached only on the third run.
    assert len(calls) == 3


def test_run_repeated_clears_thread_local_after_each_run():
    def body():
        pass

    plugin._run_repeated(_FakeItem(body), {}, "case_cleanup", None, 2, 1)
    assert session.repeat_index() is None
    assert session.current_test() == (None, None)


# --------------------------------------------------------------------------- #
# session: trace tags + per-scorer suppression for repeated runs              #
# --------------------------------------------------------------------------- #


def test_build_trace_tags_includes_repeat_index():
    tags = session.build_trace_tags("t", None, 2)
    assert tags[session.TAG_REPEAT_INDEX] == "2"
    # repeat_index 0 is still tagged (a falsy-but-present index).
    assert session.build_trace_tags("t", None, 0)[session.TAG_REPEAT_INDEX] == "0"
    assert session.TAG_REPEAT_INDEX not in session.build_trace_tags("t", None, None)


def test_record_suppressed_inside_a_repeated_run():
    result = AssertionResult(scorer_name="s", value=True, rationale=None, passed=True)

    session.set_current_test("t", None, 0)
    try:
        before = len(session.snapshot())
        session.record("t", [result])
        # Inside a repeated run the per-scorer rollup is skipped; the case-level
        # summary owns the verdict instead.
        assert len(session.snapshot()) == before
    finally:
        session.set_current_test(None, None)

    before = len(session.snapshot())
    session.record("t", [result])
    assert len(session.snapshot()) == before + 1
