import functools
import inspect
import logging
import threading
from typing import Any, Callable

from mlflow.telemetry.client import get_telemetry_client
from mlflow.telemetry.parser import API_PARSER_MAPPING
from mlflow.telemetry.schemas import APIStatus, Record
from mlflow.telemetry.utils import (
    invoked_from_internal_api,
    is_telemetry_disabled,
)

_logger = logging.getLogger(__name__)


def track_api_usage(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_telemetry_disabled() or invoked_from_internal_api():
            return func(*args, **kwargs)

        success = True
        try:
            return func(*args, **kwargs)
        except Exception:
            success = False
            raise
        finally:
            try:
                # TODO: move this inside TelemetryClient and use ThreadPoolExecutor
                thread = threading.Thread(
                    target=_add_telemetry_record,
                    name="add_telemetry_record",
                    args=(func, args, kwargs, success),
                    daemon=True,
                )
                thread.start()
            except Exception as e:
                _logger.debug(f"Failed to record telemetry for function {func.__name__}: {e}")

    return wrapper


# TODO: catch exception
def _add_telemetry_record(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    success: bool,
):
    signature = inspect.signature(func)
    bound_args = signature.bind(*args, **kwargs)
    bound_args.apply_defaults()

    arguments = dict(bound_args.arguments)
    if arguments.get("self") is not None:
        del arguments["self"]

    params = list(arguments.keys())
    if params and params[0] == "cls" and isinstance(arguments["cls"], type):
        del arguments["cls"]

    full_func_name = f"{func.__module__}.{func.__qualname__}"
    parser = API_PARSER_MAPPING.get(full_func_name) or API_PARSER_MAPPING.get(func.__name__)

    record_params = parser.extract_params(full_func_name, arguments) if parser else None
    record = Record(
        api_name=full_func_name,
        params=record_params,
        status=APIStatus.SUCCESS.value if success else APIStatus.FAILURE.value,
    )
    get_telemetry_client().add_record(record)
