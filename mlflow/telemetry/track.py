import functools
import inspect
import time
from typing import Any, Callable, Optional, ParamSpec, TypeVar

from mlflow.telemetry.client import get_telemetry_client
from mlflow.telemetry.params import PARAMS_MAPPING
from mlflow.telemetry.schemas import APIRecord, APIStatus
from mlflow.telemetry.utils import (
    is_telemetry_disabled,
)

P = ParamSpec("P")
R = TypeVar("R")


def track_api_usage(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if is_telemetry_disabled() or _is_telemetry_disabled_for_func(func):
            return func(*args, **kwargs)

        success = True
        start_time = time.time()
        try:
            return func(*args, **kwargs)
        except Exception:
            success = False
            raise
        finally:
            try:
                duration_ms = int((time.time() - start_time) * 1000)
                client = get_telemetry_client()
                if client and (
                    record := _generate_telemetry_record(func, args, kwargs, success, duration_ms)
                ):
                    client.add_record(record)
            except Exception:
                pass

    return wrapper


def _generate_telemetry_record(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    success: bool,
    duration_ms: int,
) -> Optional[APIRecord]:
    try:
        signature = inspect.signature(func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        arguments = dict(bound_args.arguments)
        if "self" in arguments:
            del arguments["self"]

        params = list(arguments.keys())
        if params and params[0] == "cls" and isinstance(arguments["cls"], type):
            del arguments["cls"]

        params_class = PARAMS_MAPPING.get(func.__name__)
        record_params = params_class.parse(arguments) if params_class else None
        return APIRecord(
            api_module=func.__module__,
            api_name=func.__qualname__,
            timestamp_ns=time.time_ns(),
            params=record_params,
            status=APIStatus.SUCCESS if success else APIStatus.FAILURE,
            duration_ms=duration_ms,
        )
    except Exception:
        return


def _is_telemetry_disabled_for_func(func: Callable[..., Any]) -> bool:
    try:
        if client := get_telemetry_client():
            if client.config:
                return func.__qualname__ in client.config.disable_api_map.get(func.__module__, [])
            # config is not fetched yet, so we assume telemetry is enabled
            else:
                return False
        # telemetry is disabled
        else:
            return True
    except Exception:
        return True
