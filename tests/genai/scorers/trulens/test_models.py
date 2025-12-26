from unittest.mock import MagicMock, patch

import pytest

from mlflow.exceptions import MlflowException


def test_create_trulens_provider_databricks():
    mock_provider = MagicMock()

    with (
        patch("mlflow.genai.scorers.trulens.models._check_trulens_installed"),
        patch(
            "mlflow.genai.scorers.trulens.models._create_databricks_managed_judge_provider",
            return_value=mock_provider,
        ) as mock_create,
    ):
        from mlflow.genai.scorers.trulens.models import create_trulens_provider

        result = create_trulens_provider("databricks")
        assert result == mock_provider
        mock_create.assert_called_once()


def test_create_trulens_provider_databricks_endpoint():
    mock_provider = MagicMock()

    with (
        patch("mlflow.genai.scorers.trulens.models._check_trulens_installed"),
        patch(
            "mlflow.genai.scorers.trulens.models._create_databricks_serving_endpoint_provider",
            return_value=mock_provider,
        ) as mock_create,
    ):
        from mlflow.genai.scorers.trulens.models import create_trulens_provider

        result = create_trulens_provider("databricks:/my-endpoint")
        assert result == mock_provider
        mock_create.assert_called_once_with("my-endpoint")


def test_create_trulens_provider_openai():
    mock_openai = MagicMock()
    mock_openai_class = MagicMock(return_value=mock_openai)

    with (
        patch("mlflow.genai.scorers.trulens.models._check_trulens_installed"),
        patch.dict(
            "sys.modules",
            {
                "trulens": MagicMock(),
                "trulens.providers": MagicMock(),
                "trulens.providers.openai": MagicMock(OpenAI=mock_openai_class),
            },
        ),
    ):
        from mlflow.genai.scorers.trulens.models import create_trulens_provider

        result = create_trulens_provider("openai:/gpt-4")
        mock_openai_class.assert_called_once_with(model_engine="gpt-4")
        assert result == mock_openai


def test_create_trulens_provider_litellm():
    mock_litellm = MagicMock()
    mock_litellm_class = MagicMock(return_value=mock_litellm)

    with (
        patch("mlflow.genai.scorers.trulens.models._check_trulens_installed"),
        patch.dict(
            "sys.modules",
            {
                "trulens": MagicMock(),
                "trulens.providers": MagicMock(),
                "trulens.providers.litellm": MagicMock(LiteLLM=mock_litellm_class),
            },
        ),
    ):
        from mlflow.genai.scorers.trulens.models import create_trulens_provider

        result = create_trulens_provider("litellm:/claude-3")
        mock_litellm_class.assert_called_once_with(model_engine="claude-3")
        assert result == mock_litellm


def test_create_trulens_provider_invalid_format():
    with patch("mlflow.genai.scorers.trulens.models._check_trulens_installed"):
        from mlflow.genai.scorers.trulens.models import create_trulens_provider

        with pytest.raises(MlflowException, match="Invalid model_uri format"):
            create_trulens_provider("gpt-4")


def test_check_trulens_installed_raises_without_trulens():
    with patch.dict("sys.modules", {"trulens": None}):
        # Clear cached imports
        import sys

        for mod in list(sys.modules.keys()):
            if "mlflow.genai.scorers.trulens" in mod:
                del sys.modules[mod]

        from mlflow.genai.scorers.trulens.models import _check_trulens_installed

        with pytest.raises(MlflowException, match="trulens"):
            _check_trulens_installed()
