import json
from unittest.mock import Mock, patch

import pytest

from mlflow.telemetry.client import TelemetryClient
from mlflow.telemetry.schemas import Record, Status


def _make_record(**kwargs):
    defaults = {
        "event_name": "genai_evaluate",
        "timestamp_ns": 1700000000000000000,
        "status": Status.SUCCESS,
        "duration_ms": 1234,
        "params": {"predict_fn_provided": True},
    }
    defaults.update(kwargs)
    return Record(**defaults)


@pytest.fixture
def databricks_client():
    with TelemetryClient() as client:
        client.info["tracking_uri_scheme"] = "databricks"
        client.info["is_localhost"] = False
        yield client


class TestForwardToDatabricks:
    def test_successful_forwarding(self, databricks_client):
        mock_response = Mock(status_code=200)
        mock_creds = Mock()
        records = [_make_record()]

        with (
            patch(
                "mlflow.telemetry.client.http_request", return_value=mock_response
            ) as mock_http,
            patch(
                "mlflow.utils.databricks_utils.get_databricks_host_creds",
                return_value=mock_creds,
            ),
            patch(
                "mlflow.tracking._tracking_service.utils.get_tracking_uri",
                return_value="databricks",
            ),
        ):
            result = databricks_client._forward_to_databricks(records)

        assert result is True
        mock_http.assert_called_once()
        call_kwargs = mock_http.call_args
        payload = call_kwargs.kwargs["json"]
        assert len(payload["events"]) == 1
        assert payload["events"][0]["event_name"] == "genai_evaluate"

    def test_params_json_serialization(self, databricks_client):
        mock_response = Mock(status_code=200)
        records = [_make_record(params={"predict_fn_provided": True})]

        with (
            patch(
                "mlflow.telemetry.client.http_request", return_value=mock_response
            ) as mock_http,
            patch(
                "mlflow.utils.databricks_utils.get_databricks_host_creds",
                return_value=Mock(),
            ),
            patch(
                "mlflow.tracking._tracking_service.utils.get_tracking_uri",
                return_value="databricks",
            ),
        ):
            databricks_client._forward_to_databricks(records)

        event = mock_http.call_args.kwargs["json"]["events"][0]
        assert "params" not in event
        assert "params_json" in event
        assert json.loads(event["params_json"]) == {"predict_fn_provided": True}

    def test_auth_failure_non_fatal(self, databricks_client):
        mock_response = Mock(status_code=403)
        records = [_make_record()]

        with (
            patch(
                "mlflow.telemetry.client.http_request", return_value=mock_response
            ),
            patch(
                "mlflow.utils.databricks_utils.get_databricks_host_creds",
                return_value=Mock(),
            ),
            patch(
                "mlflow.tracking._tracking_service.utils.get_tracking_uri",
                return_value="databricks",
            ),
        ):
            result = databricks_client._forward_to_databricks(records)

        assert result is False
        assert not databricks_client._is_stopped

    def test_credential_fetch_failure_non_fatal(self, databricks_client):
        records = [_make_record()]

        with (
            patch(
                "mlflow.utils.databricks_utils.get_databricks_host_creds",
                side_effect=Exception("no creds"),
            ),
            patch(
                "mlflow.tracking._tracking_service.utils.get_tracking_uri",
                return_value="databricks",
            ),
        ):
            result = databricks_client._forward_to_databricks(records)

        assert result is False
        assert not databricks_client._is_stopped

    def test_retry_on_500(self, databricks_client):
        mock_responses = [Mock(status_code=500), Mock(status_code=500), Mock(status_code=200)]
        records = [_make_record()]

        with (
            patch(
                "mlflow.telemetry.client.http_request", side_effect=mock_responses
            ) as mock_http,
            patch(
                "mlflow.utils.databricks_utils.get_databricks_host_creds",
                return_value=Mock(),
            ),
            patch(
                "mlflow.tracking._tracking_service.utils.get_tracking_uri",
                return_value="databricks",
            ),
            patch("mlflow.telemetry.client.time.sleep"),
        ):
            result = databricks_client._forward_to_databricks(records)

        assert result is True
        assert mock_http.call_count == 3

    @pytest.mark.parametrize("scheme", ["databricks", "databricks-uc", "uc"])
    def test_all_databricks_schemes_forward(self, scheme):
        with TelemetryClient() as client:
            client.info["tracking_uri_scheme"] = scheme
            mock_response = Mock(status_code=200)
            records = [_make_record()]

            with (
                patch(
                    "mlflow.telemetry.client.http_request", return_value=mock_response
                ) as mock_http,
                patch(
                    "mlflow.utils.databricks_utils.get_databricks_host_creds",
                    return_value=Mock(),
                ),
                patch(
                    "mlflow.tracking._tracking_service.utils.get_tracking_uri",
                    return_value=scheme,
                ),
            ):
                result = client._forward_to_databricks(records)

            assert result is True
            mock_http.assert_called_once()
