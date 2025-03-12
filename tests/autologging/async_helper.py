import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor

from mlflow.utils.autologging_utils.safety import update_wrapper_extended


def asyncify(is_async):
    """
    Decorator that converts a function to an async function if `is_async` is True. This is useful
    for testing purposes, where we want to test both synchronous and asynchronous code paths.
    """

    def decorator(fn):
        if is_async:

            async def async_fn(*args, **kwargs):
                if args and inspect.iscoroutinefunction(args[0]):
                    original = args[0]

                    def wrapped_original(*og_args, **og_kwargs):
                        # Run the original async function in a separate thread. This is a workaround
                        # for the fact that we cannot use asyncio.run here because an event loop is
                        # already running in the main thread.
                        with ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, original(*og_args, **og_kwargs))
                            return future.result()

                    args = (update_wrapper_extended(wrapped_original, original), *args[1:])
                return fn(*args, **kwargs)

            return update_wrapper_extended(async_fn, fn)
        else:
            return fn

    return decorator


def run_sync_or_async(fn, *args, **kwargs):
    """Convenience function that runs a function synchronously regardless of whether it is async."""
    if inspect.iscoroutinefunction(fn):
        return asyncio.run(fn(*args, **kwargs))
    else:
        return fn(*args, **kwargs)
