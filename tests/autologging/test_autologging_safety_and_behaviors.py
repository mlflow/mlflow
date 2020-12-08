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
    - `safe_patch` does not invoke the underlying / original function again when a patch code failure
      occurs during or after the underlying function call
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
