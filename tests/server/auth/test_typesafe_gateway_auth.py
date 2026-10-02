from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.server import auth


@pytest.mark.parametrize("allowed", [True, False])
@pytest.mark.asyncio
async def test_typesafe_gateway_requires_endpoint_use_permission(allowed):
    body = {"model": "jev-evaluator", "state": "Example", "questions": {}}
    request = MagicMock()
    request.json = AsyncMock(return_value=body)
    validator = auth._find_fastapi_validator("/gateway/typesafe/v1/systemone", "POST")

    with patch(
        "mlflow.server.auth._validate_gateway_use_permission", return_value=allowed
    ) as check:
        assert await validator("alice", request) is allowed

    check.assert_called_once_with("jev-evaluator", "alice")
    assert request.state.cached_body == body


@pytest.mark.asyncio
async def test_typesafe_gateway_rejects_missing_endpoint():
    request = MagicMock()
    request.json = AsyncMock(return_value={"state": "Example", "questions": {}})
    validator = auth._find_fastapi_validator("/gateway/typesafe/v1/systemone", "POST")

    with patch("mlflow.server.auth._validate_gateway_use_permission") as check:
        with pytest.raises(MlflowException, match="No endpoint name found"):
            await validator("alice", request)
    check.assert_not_called()
