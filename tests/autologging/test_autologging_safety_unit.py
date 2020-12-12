# pylint: disable=unused-argument

import copy
import inspect
import os
import pytest
from unittest import mock

import mlflow
import mlflow.utils.autologging_utils as autologging_utils
from mlflow.entities import RunStatus
from mlflow.tracking.client import MlflowClient
from mlflow.utils.autologging_utils import (
    safe_patch,
    autologging_integration,
    exception_safe_function,
    ExceptionSafeClass,
    PatchFunction,
    with_managed_run,
    _validate_args,
    _is_testing,
)


pytestmark = pytest.mark.large


@pytest.fixture
def test_mode_on():
    with mock.patch("mlflow.utils.autologging_utils._is_testing") as testing_mock:
        testing_mock.return_value = True
        assert autologging_utils._is_testing()
        yield


PATCH_DESTINATION_FN_DEFAULT_RESULT = "original_result"


@pytest.fixture
def patch_destination():
    class PatchObj:
        def __init__(self):
            self.fn_call_count = 0

        def fn(self, *args, **kwargs):
            self.fn_call_count += 1
            return PATCH_DESTINATION_FN_DEFAULT_RESULT

    return PatchObj()


@pytest.fixture
def test_autologging_integration():
    integration_name = "test_integration"

    @autologging_integration(integration_name)
    def autolog(disable=False):
        pass

    autolog()

    return integration_name


def test_is_testing_respects_environment_variable():
    try:
        prev_env_var_value = os.environ.pop("MLFLOW_AUTOLOGGING_TESTING", None)
        assert not _is_testing()

        os.environ["MLFLOW_AUTOLOGGING_TESTING"] = "false"
        assert not _is_testing()

        os.environ["MLFLOW_AUTOLOGGING_TESTING"] = "true"
        assert _is_testing()
    finally:
        if prev_env_var_value:
            os.environ["MLFLOW_AUTOLOGGING_TESTING"] = prev_env_var_value
        else:
            del os.environ["MLFLOW_AUTOLOGGING_TESTING"]


def test_safe_patch_forwards_expected_arguments_to_function_based_patch_implementation(
    patch_destination, test_autologging_integration
):

    foo_val = None
    bar_val = None

    def patch_impl(original, foo, bar=10):
        nonlocal foo_val
        nonlocal bar_val
        foo_val = foo
        bar_val = bar

    safe_patch(test_autologging_integration, patch_destination, "fn", patch_impl)
    patch_destination.fn(foo=7, bar=11)
    assert foo_val == 7
    assert bar_val == 11


def test_safe_patch_forwards_expected_arguments_to_class_based_patch(
    patch_destination, test_autologging_integration
):

    foo_val = None
    bar_val = None

    class TestPatch(PatchFunction):
        def _patch_implementation(self, original, foo, bar=10):  # pylint: disable=arguments-differ
            nonlocal foo_val
            nonlocal bar_val
            foo_val = foo
            bar_val = bar

        def _on_exception(self, exception):
            pass

    safe_patch(test_autologging_integration, patch_destination, "fn", TestPatch)
    with mock.patch(
        "mlflow.utils.autologging_utils.PatchFunction.call", wraps=TestPatch.call
    ) as call_mock:
        patch_destination.fn(foo=7, bar=11)
        assert call_mock.call_count == 1
        assert foo_val == 7
        assert bar_val == 11


def test_safe_patch_provides_expected_original_function(
    patch_destination, test_autologging_integration
):
    def original_fn(foo, bar=10):
        return {
            "foo": foo,
            "bar": bar,
        }

    patch_destination.fn = original_fn

    def patch_impl(original, foo, bar):
        return original(foo + 1, bar + 2)

    safe_patch(test_autologging_integration, patch_destination, "fn", patch_impl)
    assert patch_destination.fn(1, 2) == {"foo": 2, "bar": 4}


def test_safe_patch_provides_expected_original_function_to_class_based_patch(
    patch_destination, test_autologging_integration
):
    def original_fn(foo, bar=10):
        return {
            "foo": foo,
            "bar": bar,
        }

    patch_destination.fn = original_fn

    class TestPatch(PatchFunction):
        def _patch_implementation(self, original, foo, bar=10):  # pylint: disable=arguments-differ
            return original(foo + 1, bar + 2)

        def _on_exception(self, exception):
            pass

    safe_patch(test_autologging_integration, patch_destination, "fn", TestPatch)
    with mock.patch(
        "mlflow.utils.autologging_utils.PatchFunction.call", wraps=TestPatch.call
    ) as call_mock:
        assert patch_destination.fn(1, 2) == {"foo": 2, "bar": 4}
        assert call_mock.call_count == 1


def test_safe_patch_propagates_exceptions_raised_from_original_function(
    patch_destination, test_autologging_integration
):

    exc_to_throw = Exception("Bad original function")

    def original(*args, **kwargs):
        raise exc_to_throw

    patch_destination.fn = original

    patch_impl_called = False

    def patch_impl(original, *args, **kwargs):
        nonlocal patch_impl_called
        patch_impl_called = True
        return original(*args, **kwargs)

    safe_patch(test_autologging_integration, patch_destination, "fn", patch_impl)

    with pytest.raises(Exception) as exc:
        patch_destination.fn()

    assert exc.value == exc_to_throw
    assert patch_impl_called


def test_safe_patch_logs_exceptions_raised_outside_of_original_function_as_warnings(
    patch_destination, test_autologging_integration
):

    exc_to_throw = Exception("Bad patch implementation")

    def patch_impl(original, *args, **kwargs):
        raise exc_to_throw

    safe_patch(test_autologging_integration, patch_destination, "fn", patch_impl)
    with mock.patch("mlflow.utils.autologging_utils._logger.warning") as logger_mock:
        assert patch_destination.fn() == PATCH_DESTINATION_FN_DEFAULT_RESULT
        assert logger_mock.call_count == 1
        message, formatting_arg1, formatting_arg2 = logger_mock.call_args[0]
        assert "Encountered unexpected error" in message
        assert formatting_arg1 == test_autologging_integration
        assert formatting_arg2 == exc_to_throw


@pytest.mark.usefixtures(test_mode_on.__name__)
def test_safe_patch_propagates_exceptions_raised_outside_of_original_function_in_test_mode(
    patch_destination, test_autologging_integration
):

    exc_to_throw = Exception("Bad patch implementation")

    def patch_impl(original, *args, **kwargs):
        raise exc_to_throw

    safe_patch(test_autologging_integration, patch_destination, "fn", patch_impl)
    with pytest.raises(Exception) as exc:
        patch_destination.fn()

    assert exc.value == exc_to_throw


def test_safe_patch_calls_original_function_when_patch_preamble_throws(
    patch_destination, test_autologging_integration
):

    patch_impl_called = False

    def patch_impl(original, *args, **kwargs):
        nonlocal patch_impl_called
        patch_impl_called = True
        raise Exception("Bad patch preamble")

    safe_patch(test_autologging_integration, patch_destination, "fn", patch_impl)
    assert patch_destination.fn() == PATCH_DESTINATION_FN_DEFAULT_RESULT
    assert patch_destination.fn_call_count == 1
    assert patch_impl_called


def test_safe_patch_returns_original_result_without_second_call_when_patch_postamble_throws(
    patch_destination, test_autologging_integration
):

    patch_impl_called = False

    def patch_impl(original, *args, **kwargs):
        nonlocal patch_impl_called
        patch_impl_called = True
        original(*args, **kwargs)
        raise Exception("Bad patch postamble")

    safe_patch(test_autologging_integration, patch_destination, "fn", patch_impl)
    assert patch_destination.fn() == PATCH_DESTINATION_FN_DEFAULT_RESULT
    assert patch_destination.fn_call_count == 1
    assert patch_impl_called


def test_safe_patch_respects_disable_flag(patch_destination):

    patch_impl_call_count = 0

    @autologging_integration("test_respects_disable")
    def autolog(disable=False):
        def patch_impl(original, *args, **kwargs):
            nonlocal patch_impl_call_count
            patch_impl_call_count += 1
            return original(*args, **kwargs)

        safe_patch("test_respects_disable", patch_destination, "fn", patch_impl)

    autolog(disable=False)
    patch_destination.fn()
    assert patch_impl_call_count == 1

    autolog(disable=True)
    patch_destination.fn()
    assert patch_impl_call_count == 1


def test_safe_patch_returns_original_result_and_ignores_patch_return_value(
    patch_destination, test_autologging_integration
):

    patch_impl_called = False

    def patch_impl(original, *args, **kwargs):
        nonlocal patch_impl_called
        patch_impl_called = True
        return 10

    safe_patch(test_autologging_integration, patch_destination, "fn", patch_impl)
    assert patch_destination.fn() == PATCH_DESTINATION_FN_DEFAULT_RESULT
    assert patch_destination.fn_call_count == 1
    assert patch_impl_called


@pytest.mark.usefixtures(test_mode_on.__name__)
def test_safe_patch_validates_arguments_to_original_function_in_test_mode(
    patch_destination, test_autologging_integration
):
    def patch_impl(original, *args, **kwargs):
        return original("1", "2", "3")

    safe_patch(test_autologging_integration, patch_destination, "fn", patch_impl)

    with pytest.raises(Exception, match="does not match expected input"), mock.patch(
        "mlflow.utils.autologging_utils._validate_args", wraps=autologging_utils._validate_args
    ) as validate_mock:
        patch_destination.fn("a", "b", "c")

    assert validate_mock.call_count == 1


def test_safe_patch_manages_run_if_specified(patch_destination, test_autologging_integration):

    active_run = None

    def patch_impl(original, *args, **kwargs):
        nonlocal active_run
        active_run = mlflow.active_run()
        return original(*args, **kwargs)

    with mock.patch(
        "mlflow.utils.autologging_utils.with_managed_run", wraps=with_managed_run
    ) as managed_run_mock:
        safe_patch(
            test_autologging_integration, patch_destination, "fn", patch_impl, manage_run=True
        )
        patch_destination.fn()
        assert managed_run_mock.call_count == 1
        assert active_run is not None
        assert active_run.info.run_id is not None


def test_safe_patch_does_not_manage_run_if_unspecified(
    patch_destination, test_autologging_integration
):

    active_run = None

    def patch_impl(original, *args, **kwargs):
        nonlocal active_run
        active_run = mlflow.active_run()
        return original(*args, **kwargs)

    with mock.patch(
        "mlflow.utils.autologging_utils.with_managed_run", wraps=with_managed_run
    ) as managed_run_mock:
        safe_patch(
            test_autologging_integration, patch_destination, "fn", patch_impl, manage_run=False
        )
        patch_destination.fn()
        assert managed_run_mock.call_count == 0
        assert active_run is None


def test_safe_patch_preserves_signature_of_patched_function(
    patch_destination, test_autologging_integration
):
    def original(a, b, c=10, *, d=11):
        return 10

    patch_destination.fn = original

    patch_impl_called = False

    def patch_impl(original, *args, **kwargs):
        nonlocal patch_impl_called
        patch_impl_called = True
        return original(*args, **kwargs)

    safe_patch(test_autologging_integration, patch_destination, "fn", patch_impl)
    patch_destination.fn(1, 2)
    assert patch_impl_called
    assert inspect.signature(patch_destination.fn) == inspect.signature(original)


def test_safe_patch_provides_original_function_with_expected_signature(
    patch_destination, test_autologging_integration
):
    def original(a, b, c=10, *, d=11):
        return 10

    patch_destination.fn = original

    original_signature = False

    def patch_impl(original, *args, **kwargs):
        nonlocal original_signature
        original_signature = inspect.signature(original)
        return original(*args, **kwargs)

    safe_patch(test_autologging_integration, patch_destination, "fn", patch_impl)
    patch_destination.fn(1, 2)
    assert original_signature == inspect.signature(original)


def test_exception_safe_function_exhibits_expected_behavior_in_standard_mode():
    assert not autologging_utils._is_testing()

    @exception_safe_function
    def non_throwing_function():
        return 10

    assert non_throwing_function() == 10

    exc_to_throw = Exception("bad implementation")

    @exception_safe_function
    def throwing_function():
        raise exc_to_throw

    with mock.patch("mlflow.utils.autologging_utils._logger.warning") as logger_mock:
        throwing_function()
        assert logger_mock.call_count == 1
        message, formatting_arg = logger_mock.call_args[0]
        assert "unexpected error during autologging" in message
        assert formatting_arg == exc_to_throw


@pytest.mark.usefixtures(test_mode_on.__name__)
def test_exception_safe_function_exhibits_expected_behavior_in_test_mode():
    assert autologging_utils._is_testing()

    @exception_safe_function
    def non_throwing_function():
        return 10

    assert non_throwing_function() == 10

    exc_to_throw = Exception("function error")

    @exception_safe_function
    def throwing_function():
        raise exc_to_throw

    with pytest.raises(Exception) as exc:
        throwing_function()

    assert exc.value == exc_to_throw


def test_exception_safe_class_exhibits_expected_behavior_in_standard_mode():
    assert not autologging_utils._is_testing()

    class NonThrowingClass(metaclass=ExceptionSafeClass):
        def function(self):
            return 10

    assert NonThrowingClass().function() == 10

    exc_to_throw = Exception("function error")

    class ThrowingClass(metaclass=ExceptionSafeClass):
        def function(self):
            raise exc_to_throw

    with mock.patch("mlflow.utils.autologging_utils._logger.warning") as logger_mock:
        ThrowingClass().function()

        assert logger_mock.call_count == 1

        message, formatting_arg = logger_mock.call_args[0]
        assert "unexpected error during autologging" in message
        assert formatting_arg == exc_to_throw


@pytest.mark.usefixtures(test_mode_on.__name__)
def test_exception_safe_class_exhibits_expected_behavior_in_test_mode():
    assert autologging_utils._is_testing()

    class NonThrowingClass(metaclass=ExceptionSafeClass):
        def function(self):
            return 10

    assert NonThrowingClass().function() == 10

    exc_to_throw = Exception("function error")

    class ThrowingClass(metaclass=ExceptionSafeClass):
        def function(self):
            raise exc_to_throw

    with pytest.raises(Exception) as exc:
        ThrowingClass().function()

    assert exc.value == exc_to_throw


def test_patch_function_class_call_invokes_implementation_and_returns_result():
    class TestPatchFunction(PatchFunction):
        def _patch_implementation(self, original, *args, **kwargs):
            return 10

        def _on_exception(self, exception):
            pass

    assert TestPatchFunction.call("foo", lambda: "foo") == 10


def test_patch_function_class_call_handles_exceptions_properly():

    called_on_exception = False

    class TestPatchFunction(PatchFunction):
        def _patch_implementation(self, original, *args, **kwargs):
            raise Exception("implementation exception")

        def _on_exception(self, exception):
            nonlocal called_on_exception
            called_on_exception = True
            raise Exception("on_exception exception")

    # Even if an exception is thrown from `_on_exception`, we expect the original
    # exception from the implementation to be surfaced to the caller
    with pytest.raises(Exception, match="implementation exception"):
        TestPatchFunction.call("foo", lambda: "foo")

    assert called_on_exception


def test_with_managed_runs_yields_functions_and_classes_as_expected():
    def patch_function(original, *args, **kwargs):
        pass

    class TestPatch(PatchFunction):
        def _patch_implementation(self, original, *args, **kwargs):
            pass

        def _on_exception(self, exception):
            pass

    assert callable(with_managed_run(patch_function))
    assert inspect.isclass(with_managed_run(TestPatch))


def test_with_managed_run_with_non_throwing_function_exhibits_expected_behavior():
    client = MlflowClient()

    @with_managed_run
    def patch_function(original, *args, **kwargs):
        return mlflow.active_run()

    run1 = patch_function(lambda: "foo")
    run1_status = client.get_run(run1.info.run_id).info.status
    assert RunStatus.from_string(run1_status) == RunStatus.FINISHED

    with mlflow.start_run() as active_run:
        run2 = patch_function(lambda: "foo")

    assert run2 == active_run
    run2_status = client.get_run(run2.info.run_id).info.status
    assert RunStatus.from_string(run2_status) == RunStatus.FINISHED


def test_with_managed_run_with_throwing_function_exhibits_expected_behavior():
    client = MlflowClient()
    patch_function_active_run = None

    @with_managed_run
    def patch_function(original, *args, **kwargs):
        nonlocal patch_function_active_run
        patch_function_active_run = mlflow.active_run()
        raise Exception("bad implementation")

    with pytest.raises(Exception):
        patch_function(lambda: "foo")

    assert patch_function_active_run is not None
    status1 = client.get_run(patch_function_active_run.info.run_id).info.status
    assert RunStatus.from_string(status1) == RunStatus.FAILED

    with mlflow.start_run() as active_run, pytest.raises(Exception):
        patch_function(lambda: "foo")
        assert patch_function_active_run == active_run
        # `with_managed_run` should not terminate a preexisting MLflow run,
        # even if the patch function throws
        status2 = client.get_run(active_run.info.run_id).info.status
        assert RunStatus.from_string(status2) == RunStatus.FINISHED


def test_with_managed_run_with_non_throwing_class_exhibits_expected_behavior():
    client = MlflowClient()

    @with_managed_run
    class TestPatch(PatchFunction):
        def _patch_implementation(self, original, *args, **kwargs):
            return mlflow.active_run()

        def _on_exception(self, exception):
            pass

    run1 = TestPatch.call(lambda: "foo")
    run1_status = client.get_run(run1.info.run_id).info.status
    assert RunStatus.from_string(run1_status) == RunStatus.FINISHED

    with mlflow.start_run() as active_run:
        run2 = TestPatch.call(lambda: "foo")

    assert run2 == active_run
    run2_status = client.get_run(run2.info.run_id).info.status
    assert RunStatus.from_string(run2_status) == RunStatus.FINISHED


def test_with_managed_run_with_throwing_class_exhibits_expected_behavior():
    client = MlflowClient()
    patch_function_active_run = None

    @with_managed_run
    class TestPatch(PatchFunction):
        def _patch_implementation(self, original, *args, **kwargs):
            nonlocal patch_function_active_run
            patch_function_active_run = mlflow.active_run()
            raise Exception("bad implementation")

        def _on_exception(self, exception):
            pass

    with pytest.raises(Exception):
        TestPatch.call(lambda: "foo")

    assert patch_function_active_run is not None
    status1 = client.get_run(patch_function_active_run.info.run_id).info.status
    assert RunStatus.from_string(status1) == RunStatus.FAILED

    with mlflow.start_run() as active_run, pytest.raises(Exception):
        TestPatch.call(lambda: "foo")
        assert patch_function_active_run == active_run
        # `with_managed_run` should not terminate a preexisting MLflow run,
        # even if the patch function throws
        status2 = client.get_run(active_run.info.run_id).info.status
        assert RunStatus.from_string(status2) == RunStatus.FINISHED


@pytest.mark.usefixtures(test_mode_on.__name__)
def test_validate_args_succeeds_when_arg_sets_are_equivalent_or_identical():
    args = (1, "b", ["c"])
    kwargs = {
        "foo": ["bar"],
        "biz": {"baz": 5},
    }

    _validate_args(args, kwargs, args, kwargs)
    _validate_args(args, None, args, None)
    _validate_args(None, kwargs, None, kwargs)

    args_copy = copy.deepcopy(args)
    kwargs_copy = copy.deepcopy(kwargs)

    _validate_args(args, kwargs, args_copy, kwargs_copy)
    _validate_args(args, None, args_copy, None)
    _validate_args(None, kwargs, None, kwargs_copy)


@pytest.mark.usefixtures(test_mode_on.__name__)
def test_validate_args_throws_when_extra_args_are_not_functions_classes_or_lists():
    user_call_args = (1, "b", ["c"])
    user_call_kwargs = {
        "foo": ["bar"],
        "biz": {"baz": 5},
    }

    invalid_type_autologging_call_args = copy.deepcopy(user_call_args)
    invalid_type_autologging_call_args[2].append(10)
    invalid_type_autologging_call_kwargs = copy.deepcopy(user_call_kwargs)
    invalid_type_autologging_call_kwargs["new"] = {}

    with pytest.raises(Exception, match="Invalid new input"):
        _validate_args(
            user_call_args, user_call_kwargs, invalid_type_autologging_call_args, user_call_kwargs
        )

    with pytest.raises(Exception, match="Invalid new input"):
        _validate_args(
            user_call_args, user_call_kwargs, user_call_args, invalid_type_autologging_call_kwargs
        )


@pytest.mark.usefixtures(test_mode_on.__name__)
def test_validate_args_throws_when_extra_args_are_not_exception_safe():
    user_call_args = (1, "b", ["c"])
    user_call_kwargs = {
        "foo": ["bar"],
        "biz": {"baz": 5},
    }

    class Unsafe:
        pass

    unsafe_autologging_call_args = copy.deepcopy(user_call_args)
    unsafe_autologging_call_args += (lambda: "foo",)
    unsafe_autologging_call_kwargs1 = copy.deepcopy(user_call_kwargs)
    unsafe_autologging_call_kwargs1["foo"].append(Unsafe())

    with pytest.raises(Exception, match="not exception-safe"):
        _validate_args(
            user_call_args, user_call_kwargs, unsafe_autologging_call_args, user_call_kwargs
        )

    with pytest.raises(Exception, match="Invalid new input"):
        _validate_args(
            user_call_args, user_call_kwargs, user_call_args, unsafe_autologging_call_kwargs1
        )

    unsafe_autologging_call_kwargs2 = copy.deepcopy(user_call_kwargs)
    unsafe_autologging_call_kwargs2["biz"]["new"] = Unsafe()

    with pytest.raises(Exception, match="Invalid new input"):
        _validate_args(
            user_call_args, user_call_kwargs, user_call_args, unsafe_autologging_call_kwargs2
        )


@pytest.mark.usefixtures(test_mode_on.__name__)
def test_validate_args_succeeds_when_extra_args_are_exception_safe_functions_or_classes():
    user_call_args = (1, "b", ["c"])
    user_call_kwargs = {
        "foo": ["bar"],
    }

    class Safe(metaclass=ExceptionSafeClass):
        pass

    autologging_call_args = copy.deepcopy(user_call_args)
    autologging_call_args[2].append(Safe())
    autologging_call_args += (exception_safe_function(lambda: "foo"),)

    autologging_call_kwargs = copy.deepcopy(user_call_kwargs)
    autologging_call_kwargs["foo"].append(exception_safe_function(lambda: "foo"))
    autologging_call_kwargs["new"] = Safe()

    _validate_args(user_call_args, user_call_kwargs, autologging_call_args, autologging_call_kwargs)


@pytest.mark.usefixtures(test_mode_on.__name__)
def test_validate_args_throws_when_args_are_omitted():
    user_call_args = (1, "b", ["c"], {"d": "e"})
    user_call_kwargs = {
        "foo": ["bar"],
        "biz": {"baz": 4, "fuzz": 5},
    }

    invalid_autologging_call_args_1 = copy.deepcopy(user_call_args)
    invalid_autologging_call_args_1[2].pop()
    invalid_autologging_call_kwargs_1 = copy.deepcopy(user_call_kwargs)
    invalid_autologging_call_kwargs_1["foo"].pop()

    with pytest.raises(Exception, match="missing from the call"):
        _validate_args(
            user_call_args, user_call_kwargs, invalid_autologging_call_args_1, user_call_kwargs
        )

    with pytest.raises(Exception, match="missing from the call"):
        _validate_args(
            user_call_args, user_call_kwargs, user_call_args, invalid_autologging_call_kwargs_1
        )

    invalid_autologging_call_args_2 = copy.deepcopy(user_call_args)[1:]
    invalid_autologging_call_kwargs_2 = copy.deepcopy(user_call_kwargs)
    invalid_autologging_call_kwargs_2.pop("foo")

    with pytest.raises(Exception, match="missing from the call"):
        _validate_args(
            user_call_args, user_call_kwargs, invalid_autologging_call_args_2, user_call_kwargs
        )

    with pytest.raises(Exception, match="omit one or more expected keys"):
        _validate_args(
            user_call_args, user_call_kwargs, user_call_args, invalid_autologging_call_kwargs_2
        )

    invalid_autologging_call_args_3 = copy.deepcopy(user_call_args)
    invalid_autologging_call_args_3[3].pop("d")
    invalid_autologging_call_kwargs_3 = copy.deepcopy(user_call_kwargs)
    invalid_autologging_call_kwargs_3["biz"].pop("baz")

    with pytest.raises(Exception, match="omit one or more expected keys"):
        _validate_args(
            user_call_args, user_call_kwargs, invalid_autologging_call_args_3, user_call_kwargs
        )

    with pytest.raises(Exception, match="omit one or more expected keys"):
        _validate_args(
            user_call_args, user_call_kwargs, user_call_args, invalid_autologging_call_kwargs_3
        )


@pytest.mark.usefixtures(test_mode_on.__name__)
def test_validate_args_throws_when_arg_types_or_values_are_changed():
    user_call_args = (1, "b", ["c"])
    user_call_kwargs = {
        "foo": ["bar"],
    }

    invalid_autologging_call_args_1 = copy.deepcopy(user_call_args)
    invalid_autologging_call_args_1 = (2,) + invalid_autologging_call_args_1[1:]
    invalid_autologging_call_kwargs_1 = copy.deepcopy(user_call_kwargs)
    invalid_autologging_call_kwargs_1["foo"] = ["biz"]

    with pytest.raises(Exception, match="does not match expected input"):
        _validate_args(
            user_call_args, user_call_kwargs, invalid_autologging_call_args_1, user_call_kwargs
        )

    with pytest.raises(Exception, match="does not match expected input"):
        _validate_args(
            user_call_args, user_call_kwargs, user_call_args, invalid_autologging_call_kwargs_1
        )

    call_arg_1, call_arg_2, _ = copy.deepcopy(user_call_args)
    invalid_autologging_call_args_2 = ({"7": 1}, call_arg_1, call_arg_2)
    invalid_autologging_call_kwargs_2 = copy.deepcopy(user_call_kwargs)
    invalid_autologging_call_kwargs_2["foo"] = 8

    with pytest.raises(Exception, match="does not match expected type"):
        _validate_args(
            user_call_args, user_call_kwargs, invalid_autologging_call_args_2, user_call_kwargs
        )

    with pytest.raises(Exception, match="does not match expected type"):
        _validate_args(
            user_call_args, user_call_kwargs, user_call_args, invalid_autologging_call_kwargs_2
        )
