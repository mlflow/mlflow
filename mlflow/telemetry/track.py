import functools
import inspect
import logging
import time
from asyncio import CancelledError as AsyncCancelledError
from concurrent.futures import CancelledError as FutureCancelledError
from typing import Any, Callable, ParamSpec, TypeVar

from mlflow.environment_variables import MLFLOW_EXPERIMENT_ID
from mlflow.telemetry.client import get_telemetry_client
from mlflow.telemetry.events import Event
from mlflow.telemetry.schemas import Record, Status
from mlflow.telemetry.utils import _log_error, is_telemetry_disabled

P = ParamSpec("P")
R = TypeVar("R")

_logger = logging.getLogger(__name__)


def record_usage_event(
    event: type[Event], *, enabled: Callable[[], bool] | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if is_telemetry_disabled() or _is_telemetry_disabled_for_event(event, enabled=enabled):
                return func(*args, **kwargs)

            status = Status.SUCCESS
            result = None
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            except (KeyboardInterrupt, AsyncCancelledError, FutureCancelledError):
                status = Status.CANCELED
                raise
            except Exception:
                status = Status.FAILURE
                raise
            finally:
                try:
                    duration_ms = int((time.time() - start_time) * 1000)
                    _add_telemetry_record(func, args, kwargs, status, duration_ms, event, result)
                except Exception as e:
                    _log_error(f"Failed to record telemetry event {event.name}: {e}")

        return wrapper

    return decorator


def _add_telemetry_record(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    status: Status,
    duration_ms: int,
    event: type[Event],
    result: Any,
) -> None:
    try:
        if client := get_telemetry_client():
            signature = inspect.signature(func)
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            arguments = dict(bound_args.arguments)
            record_params = event.parse(arguments) or {}
            if hasattr(event, "parse_result"):
                record_params.update(event.parse_result(result))
            if experiment_id := MLFLOW_EXPERIMENT_ID.get():
                record_params["mlflow_experiment_id"] = experiment_id
            record = Record(
                event_name=event.name,
                timestamp_ns=time.time_ns(),
                params=record_params or None,
                status=status,
                duration_ms=duration_ms,
            )
            client.add_record(record)
    except Exception as e:
        _log_error(f"Failed to generate telemetry record for event {event.name}: {e}")


def _record_event(
    event: type[Event],
    params: dict[str, Any],
    *,
    success: bool = True,
    duration_ms: int = 0,
    status: Status | None = None,
) -> None:
    try:
        if _is_telemetry_disabled_for_event(event):
            return

        if client := get_telemetry_client():
            record_params = params.copy()
            if experiment_id := MLFLOW_EXPERIMENT_ID.get():
                record_params["mlflow_experiment_id"] = experiment_id
            client.add_record(
                Record(
                    event_name=event.name,
                    params=record_params or None,
                    timestamp_ns=time.time_ns(),
                    status=status
                    if status is not None
                    else (Status.SUCCESS if success else Status.FAILURE),
                    duration_ms=duration_ms,
                )
            )
    except Exception as e:
        _log_error(f"Failed to record telemetry event {event.name}: {e}")


def _is_telemetry_disabled_for_event(
    event: type[Event], *, enabled: Callable[[], bool] | None = None
) -> bool:
    try:
        if enabled is not None and not enabled():
            return True
        if client := get_telemetry_client():
            if client.config:
                return event.name in client.config.disable_events
            # when config is not fetched yet, we assume telemetry is enabled and
            # append records. After fetching the config, we check the telemetry
            # status and drop the records if disabled.
            else:
                return False
        # telemetry is disabled
        else:
            return True
    except Exception as e:
        _log_error(f"Failed to check telemetry status for event {event.name}: {e}")
        return True
