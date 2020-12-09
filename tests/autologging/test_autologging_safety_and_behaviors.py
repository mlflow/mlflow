"""
Test suite intended to test the following:

- Correctness conditions for autologging integrations:

    - All autologging functions are decorated with the `autologging_integration` decorator
      and can be disabled via the `disable=True` flag
    - Autologging patch functions are applied using `safe_patch`

- Correctness conditions for autologging safety utilities

    - `autologging_integration` stores configuration attributes as expected

    - `safe_patch` catches exceptions raised by patch code outside of test mode
    - `safe_patch` invokes the underlying / original function when patch code terminates without
      doing so (due to an exception in patch code or due to omission of an original function call)
    - `safe_patch` does not invoke the underlying / original function again when a patch code
      failure occurs during or after the underlying function call
    - `safe_patch` propagates exceptions raised by original function calls
    - `safe_patch` does not perform argument consistency / exception safety validation outside
      of test mode
    - `safe_patch` ends runs created by patch code when exceptions are encountered
    - `safe_patch` preserves the documentation and signature of the patched method
    - `safe_patch` preserves the documentation and signature of the `original` function argument
    - `safe_patch` invokes the underlying / original function directly if the associated autologging
      integration is disabled

    - `safe_patch` propagates exceptions raised by patch code in test mode
    - `safe_patch` performs argument consistency / exception safety validation in test mode
    - `safe_patch`, `exception_safe_function`, and `ExceptionSafeClass` do not operate in test mode
      unless test mode is enabled via the associated environment variable

    - `exception_safe_function` catches exceptions raised outside of test mode
    - `exception_safe_function` propagates exceptions in test mode
    - `exception_safe_function` preserves the documentation and signature of the wrapped function

    - Methods on an `ExceptionSafeClass` catch exceptions raised outside of test mode
    - Methods on an `ExceptionSafeClass` propagate exceptions in test mode
"""

import copy
import mock
import os
import pytest

import mlflow
import mlflow.utils.autologging_utils as autologging_utils
from mlflow.entities import RunStatus
from mlflow.tracking.client import MlflowClient
from mlflow.utils.autologging_utils import (
    autologging_integration, exception_safe_function, ExceptionSafeClass, PatchFunction,
    with_managed_run, _validate_args,
)
from mlflow.utils.autologging_utils import AUTOLOGGING_INTEGRATIONS

@pytest.fixture
def test_mode():
    with mock.patch("mlflow.utils.autologging_utils._is_testing") as testing_mock:
        testing_mock.return_value = True
        assert autologging_utils._is_testing()
        yield


def test_is_testing_respects_environment_variable():
    try:
        prev_env_var_value = os.environ.pop("MLFLOW_AUTOLOGGING_TESTING", None)
        assert not autologging_utils._is_testing()

        os.environ["MLFLOW_AUTOLOGGING_TESTING"] = "false"
        assert not autologging_utils._is_testing()

        os.environ["MLFLOW_AUTOLOGGING_TESTING"] = "true"
        assert autologging_utils._is_testing()
    finally:
        if prev_env_var_value:
            os.environ["MLFLOW_AUTOLOGGING_TESTING"] = prev_env_var_value
        else:
            del os.environ["MLFLOW_AUTOLOGGING_TESTING"]


def test_autologging_integration_calls_underlying_function_correctly():

    @autologging_integration("test_integration")
    def autolog(foo=7, disable=False):
        return foo

    assert autolog(foo=10) == 10


def test_autologging_integration_stores_and_updates_config():

    @autologging_integration("test_integration")
    def autolog(foo=7, bar=10, disable=False):
        return foo

    autolog()
    assert AUTOLOGGING_INTEGRATIONS["test_integration"] == {"foo": 7, "bar": 10, "disable": False}
    autolog(bar=11)
    assert AUTOLOGGING_INTEGRATIONS["test_integration"] == {"foo": 7, "bar": 11, "disable": False}
    autolog(foo=6, disable=True)
    assert AUTOLOGGING_INTEGRATIONS["test_integration"] == {"foo": 6, "bar": 10, "disable": True}


def test_autologging_integration_validates_structure_of_autolog_function():

    def fn_missing_disable_conf():
        pass

    def fn_bad_disable_conf_1(disable=True):
        pass

    # Try to use a falsy value that isn't "false"
    def fn_bad_disable_conf_2(disable=0):
        pass

    for fn in [fn_missing_disable_conf, fn_bad_disable_conf_1, fn_bad_disable_conf_2]:
        with pytest.raises(Exception) as exc:
            autologging_integration("test")(fn)
        assert "must specify a 'disable' argument" in str(exc)

    def fn_positional_args(positional, disable=False):
        pass

    with pytest.raises(Exception) as exc:
        autologging_integration("test")(fn_positional_args)
        assert "Positional arguments are not allowed" in str(exc)

    # Failure to apply the @autologging_integration decorator should not create a
    # placeholder for configuration state
    assert "test" not in AUTOLOGGING_INTEGRATIONS


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
        message, formatting_args = logger_mock.call_args[0]
        assert "unexpected error during autologging" in message
        assert formatting_args == exc_to_throw


def test_exception_safe_function_exhibits_expected_behavior_in_test_mode(test_mode):  # pylint: disable=unused-argument
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

        message, formatting_args = logger_mock.call_args[0]
        assert "unexpected error during autologging" in message
        assert formatting_args == exc_to_throw


def test_exception_safe_class_exhibits_expected_behavior_in_test_mode(test_mode):  # pylint: disable=unused-argument
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

        def _patch_implementation(self, original, *args, **kwargs):  # pylint: disable=unused-argument
            return 10
        def _on_exception(self, exception):  # pylint: disable=unused-argument
            pass

    assert TestPatchFunction.call("foo", lambda: "foo") == 10

def test_patch_function_class_call_handles_exceptions_properly():

    called_on_exception = False

    class TestPatchFunction(PatchFunction):

        def _patch_implementation(self, original, *args, **kwargs):  # pylint: disable=unused-argument
            raise Exception("implementation exception")

        def _on_exception(self, exception):  # pylint: disable=unused-argument
            nonlocal called_on_exception
            called_on_exception = True
            raise Exception("on_exception exception")

    with pytest.raises(Exception) as exc:
        TestPatchFunction.call("foo", lambda: "foo")

    assert called_on_exception == True
    # Even if an exception is thrown from `_on_exception`, we expect the original
    # exception from the implementation to be surfaced to the caller
    assert "implementation exception" in str(exc)


def test_with_managed_runs_yields_functions_and_classes_as_expected():
    def patch_function(original, *args, **kwargs):  # pylint: disable=unused-argument
        pass

    class TestPatch(PatchFunction):
        def _patch_implementation(self, original, *args, **kwargs):  # pylint: disable=unused-argument
            pass
        def _on_exception(self, exception):  # pylint: disable=unused-argument
            pass

    assert callable(with_managed_run(patch_function))
    import inspect
    assert inspect.isclass(with_managed_run(TestPatch))


def test_with_managed_run_with_non_throwing_function_exhibits_expected_behavior():
    client = MlflowClient()

    @with_managed_run
    def patch_function(original, *args, **kwargs):  # pylint: disable=unused-argument
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
    def patch_function(original, *args, **kwargs):  # pylint: disable=unused-argument
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
        def _patch_implementation(self, original, *args, **kwargs):  # pylint: disable=unused-argument
            return mlflow.active_run()
        def _on_exception(self, exception):  # pylint: disable=unused-argument
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
        def _patch_implementation(self, original, *args, **kwargs):  # pylint: disable=unused-argument
            nonlocal patch_function_active_run
            patch_function_active_run = mlflow.active_run()
            raise Exception("bad implementation")
        def _on_exception(self, exception):  # pylint: disable=unused-argument
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


def test_validate_args_succeeds_when_arg_sets_are_equivalent_or_identical(test_mode):  # pylint: disable=unused-argument
    args = [1, "b", ["c"]]
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


def test_validate_args_throws_when_extra_args_are_not_functions_classes_or_lists(test_mode):  # pylint: disable=unused-argument
    user_call_args = [1, "b", ["c"]]
    user_call_kwargs = {
        "foo": ["bar"],
        "biz": {"baz": 5},
    }

    invalid_type_autologging_call_args = copy.deepcopy(user_call_args)
    invalid_type_autologging_call_args[2].append(10)
    invalid_type_autologging_call_kwargs = copy.deepcopy(user_call_kwargs)
    invalid_type_autologging_call_kwargs["new"] = {}

    with pytest.raises(Exception) as exc:
        _validate_args(user_call_args, user_call_kwargs, invalid_type_autologging_call_args, user_call_kwargs)
    assert "Invalid new input" in str(exc)

    with pytest.raises(Exception) as exc:
        _validate_args(user_call_args, user_call_kwargs, user_call_args, invalid_type_autologging_call_kwargs)
    assert "Invalid new input" in str(exc)


def test_validate_args_throws_when_extra_args_are_not_exception_safe(test_mode):  # pylint: disable=unused-argument
    user_call_args = [1, "b", ["c"]]
    user_call_kwargs = {
        "foo": ["bar"],
        "biz": {"baz": 5},
    }

    class Unsafe:
        pass

    unsafe_autologging_call_args = copy.deepcopy(user_call_args)
    unsafe_autologging_call_args.append(lambda: "foo")
    unsafe_autologging_call_kwargs1 = copy.deepcopy(user_call_kwargs)
    unsafe_autologging_call_kwargs1["foo"].append(Unsafe())

    with pytest.raises(Exception) as exc:
        _validate_args(user_call_args, user_call_kwargs, unsafe_autologging_call_args, user_call_kwargs)
    assert "not exception-safe" in str(exc)

    with pytest.raises(Exception) as exc:
        _validate_args(user_call_args, user_call_kwargs, user_call_args, unsafe_autologging_call_kwargs1)
    assert "Invalid new input" in str(exc)

    unsafe_autologging_call_kwargs2 = copy.deepcopy(user_call_kwargs)
    unsafe_autologging_call_kwargs2["biz"]["new"] = Unsafe()

    with pytest.raises(Exception) as exc:
        _validate_args(user_call_args, user_call_kwargs, user_call_args, unsafe_autologging_call_kwargs2)
    assert "Invalid new input" in str(exc)


def test_validate_args_succeeds_when_extra_args_are_exception_safe_functions_or_classes(test_mode):  # pylint: disable=unused-argument
    user_call_args = [1, "b", ["c"]]
    user_call_kwargs = {
        "foo": ["bar"],
    }

    class Safe(metaclass=ExceptionSafeClass):
        pass

    autologging_call_args = copy.deepcopy(user_call_args)
    autologging_call_args[2].append(Safe())
    autologging_call_args.append(exception_safe_function(lambda: "foo"))

    autologging_call_kwargs = copy.deepcopy(user_call_kwargs)
    autologging_call_kwargs["foo"].append(exception_safe_function(lambda: "foo"))
    autologging_call_kwargs["new"] = Safe()

    _validate_args(user_call_args, user_call_kwargs, autologging_call_args, autologging_call_kwargs)


def test_validate_args_throws_when_args_are_omitted(test_mode):  # pylint: disable=unused-argument
    user_call_args = [1, "b", ["c"], {"d": "e"}]
    user_call_kwargs = {
        "foo": ["bar"],
        "biz": {"baz": 4, "fuzz": 5},
    }

    invalid_autologging_call_args_1 = copy.deepcopy(user_call_args)
    invalid_autologging_call_args_1[2].pop()
    invalid_autologging_call_kwargs_1 = copy.deepcopy(user_call_kwargs)
    invalid_autologging_call_kwargs_1["foo"].pop()

    with pytest.raises(Exception) as exc:
        _validate_args(user_call_args, user_call_kwargs, invalid_autologging_call_args_1, user_call_kwargs)
    assert "missing from the call" in str(exc)

    with pytest.raises(Exception) as exc:
        _validate_args(user_call_args, user_call_kwargs, user_call_args, invalid_autologging_call_kwargs_1)
    assert "missing from the call" in str(exc)

    invalid_autologging_call_args_2 = copy.deepcopy(user_call_args)[1:]
    invalid_autologging_call_kwargs_2 = copy.deepcopy(user_call_kwargs)
    invalid_autologging_call_kwargs_2.pop("foo")

    with pytest.raises(Exception) as exc:
        _validate_args(user_call_args, user_call_kwargs, invalid_autologging_call_args_2, user_call_kwargs)
    assert "missing from the call" in str(exc)

    with pytest.raises(Exception) as exc:
        _validate_args(user_call_args, user_call_kwargs, user_call_args, invalid_autologging_call_kwargs_2)
    assert "omit one or more expected keys" in str(exc)


    invalid_autologging_call_args_3 = copy.deepcopy(user_call_args)
    invalid_autologging_call_args_3[3].pop("d")
    invalid_autologging_call_kwargs_3 = copy.deepcopy(user_call_kwargs)
    invalid_autologging_call_kwargs_3["biz"].pop("baz")

    with pytest.raises(Exception) as exc:
        _validate_args(user_call_args, user_call_kwargs, invalid_autologging_call_args_3, user_call_kwargs)
    assert "omit one or more expected keys" in str(exc)

    with pytest.raises(Exception) as exc:
        _validate_args(user_call_args, user_call_kwargs, user_call_args, invalid_autologging_call_kwargs_3)
    assert "omit one or more expected keys" in str(exc)


def test_validate_args_throws_when_arg_types_or_values_are_changed(test_mode):  # pylint: disable=unused-argument
    user_call_args = [1, "b", ["c"]]
    user_call_kwargs = {
        "foo": ["bar"],
    }

    invalid_autologging_call_args_1 = copy.deepcopy(user_call_args)
    invalid_autologging_call_args_1[0] = 2
    invalid_autologging_call_kwargs_1 = copy.deepcopy(user_call_kwargs)
    invalid_autologging_call_kwargs_1["foo"] = ["biz"]

    with pytest.raises(Exception) as exc:
        _validate_args(user_call_args, user_call_kwargs, invalid_autologging_call_args_1, user_call_kwargs)
    assert "does not match expected input" in str(exc)

    with pytest.raises(Exception) as exc:
        _validate_args(user_call_args, user_call_kwargs, user_call_args, invalid_autologging_call_kwargs_1)
    assert "does not match expected input" in str(exc)

    invalid_autologging_call_args_2 = copy.deepcopy(user_call_args)
    invalid_autologging_call_args_2[1] = {"7": 1}
    invalid_autologging_call_kwargs_2 = copy.deepcopy(user_call_kwargs)
    invalid_autologging_call_kwargs_2["foo"] = 8

    with pytest.raises(Exception) as exc:
        _validate_args(user_call_args, user_call_kwargs, invalid_autologging_call_args_2, user_call_kwargs)
    assert "does not match expected type" in str(exc)

    with pytest.raises(Exception) as exc:
        _validate_args(user_call_args, user_call_kwargs, user_call_args, invalid_autologging_call_kwargs_2)
    assert "does not match expected type" in str(exc)
