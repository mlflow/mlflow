import functools
import inspect
import logging
import time
from typing import Any, Callable, Optional, ParamSpec, TypeVar

from mlflow.telemetry.client import get_telemetry_client
from mlflow.telemetry.parser import API_PARSER_MAPPING
from mlflow.telemetry.schemas import APIRecord, APIStatus
from mlflow.telemetry.utils import (
    _disable_telemetry,
    invoked_from_internal_api,
    is_telemetry_disabled,
)

_logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


P = ParamSpec("P")
R = TypeVar("R")


def track_api_usage(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if is_telemetry_disabled() or invoked_from_internal_api(func):
            return func(*args, **kwargs)

        success = True
        start_time = time.time()
        try:
            # disable telemetry for nested API calls
            with _disable_telemetry():
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
            except Exception as e:
                _logger.debug(f"Failed to record telemetry for function {func.__name__}: {e}")

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

        parser = API_PARSER_MAPPING.get(func.__name__)
        record_params = parser.extract_params(func, arguments) if parser else None
        return APIRecord(
            api_module=func.__module__,
            api_name=func.__qualname__,
            params=record_params,
            status=APIStatus.SUCCESS.value if success else APIStatus.FAILURE.value,
            duration_ms=duration_ms,
        )
    except Exception:
        _logger.debug(
            f"Failed to generate telemetry record for function {func.__name__}",
            exc_info=True,
        )
