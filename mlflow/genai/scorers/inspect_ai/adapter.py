"""Adapter utilities for resolving and invoking Inspect AI scorer tasks.

This module is focused on two responsibilities:
- Locate a callable for a given Inspect AI `metric_name` (task/scorer).
- Invoke the callable in a synchronous manner (awaiting async results when needed).

The implementation mirrors the defensive import and discovery strategy used
in the wrapper, but centralizes the callable resolution and invocation so
other modules (e.g., `wrapper.py`, `registry.py`) can reuse it.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable, Dict, Optional

from mlflow.exceptions import MlflowException

_logger = logging.getLogger(__name__)

INSPECTAI_NOT_INSTALLED_ERROR_MESSAGE = (
    "Inspect AI scorers require the 'inspectai' package. "
    "Please install it with: pip install inspectai"
)


def _import_inspectai_module():
    try:
        import inspectai as ia  

        return ia
    except Exception:
        pass

    try:
        import inspect_ai as ia  

        return ia
    except Exception:
        raise MlflowException(INSPECTAI_NOT_INSTALLED_ERROR_MESSAGE)


def find_task_callable(metric_name: str) -> Callable[..., Any]:
    """Resolve a callable for `metric_name` inside the installed Inspect AI package.

    The function searches common locations and naming conventions so the
    registry/adapter can support a variety of Inspect AI package shapes.
    """
    ia = _import_inspectai_module()

    try:
        from inspect_ai._util.registry import registry_lookup
        for registry_type in ("task", "scorer"):
            task = registry_lookup(registry_type, metric_name)
            if callable(task):
                return task
    except Exception:
        pass

    get_task = getattr(ia, "get_task", None)
    if callable(get_task):
        try:
            task = get_task(metric_name)
            if callable(task):
                return task
        except Exception:
            _logger.debug("inspectai.get_task failed for %s", metric_name, exc_info=True)

    attr = getattr(ia, metric_name, None)
    if callable(attr):
        return attr

    for sub in ("tasks", "scorers", "client"):
        submod = getattr(ia, sub, None)
        if submod is None:
            continue
        
        try:
            candidate = getattr(submod, "get", None)
            if callable(candidate):
                task = candidate(metric_name)
                if callable(task):
                    return task
        except Exception:
            _logger.debug("Failed to get %s from %s", metric_name, sub, exc_info=True)

        
        candidate = getattr(submod, metric_name, None)
        if callable(candidate):
            return candidate

    raise MlflowException(f"Unable to locate Inspect AI callable for metric '{metric_name}'")


def invoke_task_callable(
    fn: Callable[..., Any], *, metric_name: str, payload: Dict[str, Any], call_kwargs: Optional[Dict[str, Any]] = None
) -> Any:
    """Invoke the resolved Inspect AI callable and return its result.

    This function attempts to call the callable with a common keyword-style
    signature (`metric_name`, `payload`, **call_kwargs) and falls back to a
    positional call if necessary. If the callable returns an awaitable, it is
    awaited via `asyncio.run` to provide a synchronous interface to callers.
    """
    call_kwargs = dict(call_kwargs or {})

    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__

    try:
        sig = inspect.signature(fn)
        has_payload_param = "payload" in sig.parameters
    except Exception:
        has_payload_param = True

    if has_payload_param:
        try:
            result = fn(metric_name=metric_name, payload=payload, **call_kwargs)
        except TypeError:
            try:
                result = fn(metric_name, payload, **call_kwargs)
            except Exception as e:
                try:
                    merged = {**payload, **call_kwargs}
                    result = fn(**merged)
                except Exception as inner_exc:
                    _logger.error("Failed to invoke Inspect AI callable '%s': %s", fn, inner_exc)
                    raise MlflowException(f"Failed to invoke Inspect AI callable: {inner_exc}") from inner_exc
        except Exception as e:
            _logger.error("Inspect AI callable raised an exception: %s", e)
            raise
    else:
        try:
            merged = {**payload, **call_kwargs}
            result = fn(**merged)
        except Exception as e:
            _logger.error("Failed to invoke Inspect AI callable '%s': %s", fn, e)
            raise MlflowException(f"Failed to invoke Inspect AI callable: {e}") from e

    if inspect.isawaitable(result) or asyncio.iscoroutine(result):
        try:
            return asyncio.run(result)
        except asyncio.TimeoutError as exc:
            raise MlflowException(
                f"Inspect AI scorer timed out during evaluation."
            ) from exc
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(result)
            except asyncio.TimeoutError as exc:
                raise MlflowException(
                    f"Inspect AI scorer timed out during evaluation."
                ) from exc
            finally:
                loop.close()

    return result