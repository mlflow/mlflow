import functools
import inspect
import logging
import threading
from typing import Callable

from mlflow.telemetry.client import get_telemetry_client
from mlflow.telemetry.schemas import APIStatus, Record
from mlflow.telemetry.utils import (
    API_RECORD_PARAMS_MAPPING,
    is_telemetry_disabled,
    temporarily_disable_telemetry,
)

_logger = logging.getLogger(__name__)


def track_api_usage(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_telemetry_disabled():
            return func(*args, **kwargs)

        success = True
        try:
            # Temporarily disable telemetry for the function call to avoid recording
            # telemetry for the subsequent API calls.
            with temporarily_disable_telemetry():
                return func(*args, **kwargs)
        except Exception:
            success = False
            raise
        finally:
            try:
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


def _add_telemetry_record(func, args, kwargs, success):
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
    record_params_func = API_RECORD_PARAMS_MAPPING.get(
        full_func_name
    ) or API_RECORD_PARAMS_MAPPING.get(func.__name__)

    record_params = record_params_func(full_func_name, arguments) if record_params_func else None
    record = Record(
        api_name=full_func_name,
        params=record_params,
        status=APIStatus.SUCCESS if success else APIStatus.FAILURE,
    )
    get_telemetry_client().add_record(record)
