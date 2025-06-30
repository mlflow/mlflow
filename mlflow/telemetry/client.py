import logging

from mlflow.telemetry.schemas import Record, TelemetryInfo
from mlflow.telemetry.utils import is_telemetry_disabled
from mlflow.tracking._tracking_service.utils import _get_store

_logger = logging.getLogger(__name__)


class TelemetryClient:
    def __init__(self):
        self.info = TelemetryInfo()
        self.records: list[Record] = []

    # temp helper function to test
    # TODO: update this in next PR
    def add_record(self, record: Record):
        """
        Add a record to the queue to be sent to the telemetry server.
        """
        self.records.append(record)

    def _update_backend_store(self):
        """
        Backend store might be changed after mlflow is imported, we should use this
        method to update the backend store info at sending telemetry step.
        """
        tracking_store = _get_store()
        self.info.backend_store = tracking_store.__class__.__name__


_MLFLOW_TELEMETRY_CLIENT = None


def set_telemetry_client():
    global _MLFLOW_TELEMETRY_CLIENT

    if is_telemetry_disabled():
        _logger.debug("MLflow Telemetry is disabled")
        # set to None again so this function can be used to
        # re-initialize the telemetry client
        _MLFLOW_TELEMETRY_CLIENT = None
    else:
        _MLFLOW_TELEMETRY_CLIENT = TelemetryClient()


def get_telemetry_client() -> TelemetryClient | None:
    return _MLFLOW_TELEMETRY_CLIENT
