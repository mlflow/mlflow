from mlflow.telemetry.client import TelemetryClient, get_telemetry_client


def wait_for_telemetry_threads(client: TelemetryClient = None):
    """Wait for telemetry threads to finish to avoid race conditions in tests."""
    telemetry_client = client or get_telemetry_client()
    if telemetry_client is None:
        return

    # Flush the telemetry client to ensure all pending records are processed
    telemetry_client.flush()
