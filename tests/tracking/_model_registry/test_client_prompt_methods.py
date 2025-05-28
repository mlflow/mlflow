import pytest
from unittest.mock import Mock

from mlflow.entities.model_registry.prompt import Prompt
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracking._model_registry.client import ModelRegistryClient


@pytest.fixture
def mock_store():
    """Mock store for testing client methods."""
    return Mock()


@pytest.fixture
def client(mock_store):
    """ModelRegistryClient with mocked store."""
    client = ModelRegistryClient()
    client.store = mock_store
    return client


class TestPromptMethods:
    """Test the prompt-related methods in ModelRegistryClient."""

    def test_create_prompt(self, client, mock_store):
        """Test create_prompt method."""
        expected_prompt = Prompt(name="test_prompt", version=1, template="Hello {{name}}!")
        mock_store.create_prompt.return_value = expected_prompt

        result = client.create_prompt(
            name="test_prompt",
            template="Hello {{name}}!",
            description="Test prompt",
            tags={"env": "test"}
        )

        mock_store.create_prompt.assert_called_once_with(
            "test_prompt", "Hello {{name}}!", "Test prompt", {"env": "test"}
        )
        assert result == expected_prompt

    def test_get_prompt(self, client, mock_store):
        """Test get_prompt method."""
        expected_prompt = Prompt(name="test_prompt", version=1, template="Hello {{name}}!")
        mock_store.get_prompt.return_value = expected_prompt

        result = client.get_prompt("test_prompt", version="1")

        mock_store.get_prompt.assert_called_once_with("test_prompt", "1")
        assert result == expected_prompt

    def test_get_prompt_no_version(self, client, mock_store):
        """Test get_prompt method without version."""
        expected_prompt = Prompt(name="test_prompt", version=2, template="Hello {{name}}!")
        mock_store.get_prompt.return_value = expected_prompt

        result = client.get_prompt("test_prompt")

        mock_store.get_prompt.assert_called_once_with("test_prompt", None)
        assert result == expected_prompt

    def test_search_prompts(self, client, mock_store):
        """Test search_prompts method."""
        prompts = [
            Prompt(name="prompt1", version=1, template="Template 1"),
            Prompt(name="prompt2", version=1, template="Template 2"),
        ]
        expected_result = PagedList(prompts, "next_token")
        mock_store.search_prompts.return_value = expected_result

        result = client.search_prompts(
            filter_string="name='test'",
            max_results=10,
            order_by=["name"],
            page_token="token"
        )

        mock_store.search_prompts.assert_called_once_with(
            "name='test'", 10, ["name"], "token"
        )
        assert result == expected_result

    def test_delete_prompt(self, client, mock_store):
        """Test delete_prompt method."""
        client.delete_prompt("test_prompt")

        mock_store.delete_prompt.assert_called_once_with("test_prompt")

    def test_create_prompt_version(self, client, mock_store):
        """Test create_prompt_version method."""
        expected_prompt = Prompt(name="test_prompt", version=2, template="Hello {{name}}!")
        mock_store.create_prompt_version.return_value = expected_prompt

        result = client.create_prompt_version(
            name="test_prompt",
            template="Hello {{name}}!",
            description="Version 2",
            tags={"version": "2"}
        )

        mock_store.create_prompt_version.assert_called_once_with(
            "test_prompt", "Hello {{name}}!", "Version 2", {"version": "2"}
        )
        assert result == expected_prompt

    def test_get_prompt_version(self, client, mock_store):
        """Test get_prompt_version method."""
        expected_prompt = Prompt(name="test_prompt", version=1, template="Hello {{name}}!")
        mock_store.get_prompt_version.return_value = expected_prompt

        result = client.get_prompt_version("test_prompt", 1)

        mock_store.get_prompt_version.assert_called_once_with("test_prompt", 1)
        assert result == expected_prompt

    def test_delete_prompt_version(self, client, mock_store):
        """Test delete_prompt_version method."""
        client.delete_prompt_version("test_prompt", 1)

        mock_store.delete_prompt_version.assert_called_once_with("test_prompt", 1)

    def test_set_prompt_tag(self, client, mock_store):
        """Test set_prompt_tag method."""
        client.set_prompt_tag("test_prompt", "env", "prod")

        mock_store.set_prompt_tag.assert_called_once_with("test_prompt", "env", "prod")

    def test_delete_prompt_tag(self, client, mock_store):
        """Test delete_prompt_tag method."""
        client.delete_prompt_tag("test_prompt", "env")

        mock_store.delete_prompt_tag.assert_called_once_with("test_prompt", "env")

    def test_set_prompt_version_tag(self, client, mock_store):
        """Test set_prompt_version_tag method."""
        client.set_prompt_version_tag("test_prompt", 1, "env", "prod")

        mock_store.set_prompt_version_tag.assert_called_once_with("test_prompt", 1, "env", "prod")

    def test_delete_prompt_version_tag(self, client, mock_store):
        """Test delete_prompt_version_tag method."""
        client.delete_prompt_version_tag("test_prompt", 1, "env")

        mock_store.delete_prompt_version_tag.assert_called_once_with("test_prompt", 1, "env")

    def test_set_prompt_alias(self, client, mock_store):
        """Test set_prompt_alias method."""
        client.set_prompt_alias("test_prompt", "production", 1)

        mock_store.set_prompt_alias.assert_called_once_with("test_prompt", "production", 1)

    def test_delete_prompt_alias(self, client, mock_store):
        """Test delete_prompt_alias method."""
        client.delete_prompt_alias("test_prompt", "production")

        mock_store.delete_prompt_alias.assert_called_once_with("test_prompt", "production")

    def test_get_prompt_version_by_alias(self, client, mock_store):
        """Test get_prompt_version_by_alias method."""
        expected_prompt = Prompt(name="test_prompt", version=1, template="Hello {{name}}!")
        mock_store.get_prompt_version_by_alias.return_value = expected_prompt

        result = client.get_prompt_version_by_alias("test_prompt", "production")

        mock_store.get_prompt_version_by_alias.assert_called_once_with("test_prompt", "production")
        assert result == expected_prompt 