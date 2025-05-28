import pytest
from unittest.mock import patch

from mlflow.entities.model_registry.registered_model_tag import RegisteredModelTag
from mlflow.exceptions import MlflowException
from mlflow.prompt.registry_utils import (
    is_prompt_supported_registry,
    require_prompt_registry,
    translate_prompt_exception,
    validate_prompt_name,
    handle_resource_already_exist_error,
    has_prompt_tag,
)
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_ALREADY_EXISTS


class TestIsPromptSupportedRegistry:
    """Test the is_prompt_supported_registry function."""

    def test_oss_mlflow_registry(self):
        """Test that OSS MLflow registry supports prompts."""
        assert is_prompt_supported_registry("sqlite:///registry.db") is True
        assert is_prompt_supported_registry("postgresql://user:pass@host/db") is True
        assert is_prompt_supported_registry("mysql://user:pass@host/db") is True

    def test_unity_catalog_registry(self):
        """Test that Unity Catalog registries support prompts."""
        assert is_prompt_supported_registry("databricks-uc") is True
        assert is_prompt_supported_registry("databricks-uc://host") is True

    def test_legacy_databricks_registry(self):
        """Test that legacy Databricks workspace registry doesn't support prompts."""
        assert is_prompt_supported_registry("databricks") is False
        assert is_prompt_supported_registry("databricks://host") is False

    @patch("mlflow.get_registry_uri")
    def test_default_registry_uri(self, mock_get_registry_uri):
        """Test using default registry URI."""
        mock_get_registry_uri.return_value = "sqlite:///registry.db"
        assert is_prompt_supported_registry() is True

        mock_get_registry_uri.return_value = "databricks"
        assert is_prompt_supported_registry() is False


class TestRequirePromptRegistry:
    """Test the require_prompt_registry decorator."""

    @require_prompt_registry
    def dummy_function(self, arg1, arg2=None):
        """A dummy function for testing the decorator."""
        return f"result: {arg1}, {arg2}"

    @patch("mlflow.get_registry_uri")
    def test_supported_registry(self, mock_get_registry_uri):
        """Test decorator allows execution with supported registry."""
        mock_get_registry_uri.return_value = "sqlite:///registry.db"
        result = self.dummy_function("test")
        assert result == "result: test, None"

    @patch("mlflow.get_registry_uri")
    def test_unsupported_registry(self, mock_get_registry_uri):
        """Test decorator raises exception with unsupported registry."""
        mock_get_registry_uri.return_value = "databricks"
        
        with pytest.raises(MlflowException) as exc_info:
            self.dummy_function("test")
        
        assert "not supported with the current registry" in str(exc_info.value)
        assert exc_info.value.error_code == INVALID_PARAMETER_VALUE

    def test_docstring_updated(self):
        """Test that decorator adds note to docstring."""
        assert "This API is supported in OSS MLflow" in self.dummy_function.__doc__


class TestTranslatePromptException:
    """Test the translate_prompt_exception decorator."""

    @translate_prompt_exception
    def dummy_function_that_raises(self, message):
        """A dummy function that raises MlflowException."""
        raise MlflowException(message)

    def test_translates_registered_model_to_prompt(self):
        """Test that 'registered model' is translated to 'prompt'."""
        with pytest.raises(MlflowException) as exc_info:
            self.dummy_function_that_raises("Registered model not found")
        
        assert "Prompt not found" in str(exc_info.value)

    def test_translates_model_version_to_prompt(self):
        """Test that 'model version' is translated to 'prompt'."""
        with pytest.raises(MlflowException) as exc_info:
            self.dummy_function_that_raises("Model version already exists")
        
        assert "prompt already exists" in str(exc_info.value)

    def test_preserves_case(self):
        """Test that case is preserved in translation."""
        with pytest.raises(MlflowException) as exc_info:
            self.dummy_function_that_raises("Model Version not found")
        
        assert "Prompt not found" in str(exc_info.value)

    def test_no_translation_needed(self):
        """Test that messages without model/version are unchanged."""
        with pytest.raises(MlflowException) as exc_info:
            self.dummy_function_that_raises("Some other error")
        
        assert "Some other error" in str(exc_info.value)

    def test_non_mlflow_exception_passthrough(self):
        """Test that non-MlflowException errors pass through unchanged."""
        @translate_prompt_exception
        def raise_value_error():
            raise ValueError("test error")
        
        with pytest.raises(ValueError) as exc_info:
            raise_value_error()
        
        assert "test error" in str(exc_info.value)


class TestValidatePromptName:
    """Test the validate_prompt_name function."""

    def test_valid_names(self):
        """Test that valid prompt names don't raise exceptions."""
        validate_prompt_name("valid_name")
        validate_prompt_name("valid-name")
        validate_prompt_name("valid.name")
        validate_prompt_name("valid123")
        validate_prompt_name("valid_name_123")

    def test_invalid_name_non_string(self):
        """Test that non-string names raise exceptions."""
        with pytest.raises(MlflowException):
            validate_prompt_name(123)
        
        with pytest.raises(MlflowException):
            validate_prompt_name(None)

    def test_invalid_name_empty_string(self):
        """Test that empty string raises exception."""
        with pytest.raises(MlflowException):
            validate_prompt_name("")

    def test_invalid_name_special_characters(self):
        """Test that names with invalid special characters raise exceptions."""
        with pytest.raises(MlflowException):
            validate_prompt_name("invalid@name")
        
        with pytest.raises(MlflowException):
            validate_prompt_name("invalid name")  # space
        
        with pytest.raises(MlflowException):
            validate_prompt_name("invalid/name")


class TestHandleResourceAlreadyExistError:
    """Test the handle_resource_already_exist_error function."""

    def test_model_to_model_conflict(self):
        """Test model to model conflict error."""
        with pytest.raises(MlflowException) as exc_info:
            handle_resource_already_exist_error("test", False, False)
        
        assert "Registered Model (name=test) already exists" in str(exc_info.value)
        assert exc_info.value.error_code == RESOURCE_ALREADY_EXISTS

    def test_prompt_to_prompt_conflict(self):
        """Test prompt to prompt conflict error."""
        with pytest.raises(MlflowException) as exc_info:
            handle_resource_already_exist_error("test", True, True)
        
        assert "Prompt (name=test) already exists" in str(exc_info.value)
        assert exc_info.value.error_code == RESOURCE_ALREADY_EXISTS

    def test_model_to_prompt_conflict(self):
        """Test model to prompt conflict error."""
        with pytest.raises(MlflowException) as exc_info:
            handle_resource_already_exist_error("test", False, True)
        
        assert "already taken by a registered model" in str(exc_info.value)
        assert "MLflow does not allow creating a model and a prompt with the same name" in str(exc_info.value)
        assert exc_info.value.error_code == RESOURCE_ALREADY_EXISTS

    def test_prompt_to_model_conflict(self):
        """Test prompt to model conflict error."""
        with pytest.raises(MlflowException) as exc_info:
            handle_resource_already_exist_error("test", True, False)
        
        assert "already taken by a prompt" in str(exc_info.value)
        assert "MLflow does not allow creating a model and a prompt with the same name" in str(exc_info.value)
        assert exc_info.value.error_code == RESOURCE_ALREADY_EXISTS


class TestHasPromptTag:
    """Test the has_prompt_tag function."""

    def test_dict_tags_with_prompt_tag(self):
        """Test dict tags containing prompt tag."""
        tags = {"mlflow.prompt.is_prompt": "true", "other": "value"}
        assert has_prompt_tag(tags) is True

    def test_dict_tags_without_prompt_tag(self):
        """Test dict tags not containing prompt tag."""
        tags = {"other": "value"}
        assert has_prompt_tag(tags) is False

    def test_empty_dict_tags(self):
        """Test empty dict tags."""
        assert has_prompt_tag({}) is False

    def test_none_dict_tags(self):
        """Test None dict tags."""
        assert has_prompt_tag(None) is None

    def test_list_tags_with_prompt_tag(self):
        """Test list tags containing prompt tag."""
        tags = [
            RegisteredModelTag(key="mlflow.prompt.is_prompt", value="true"),
            RegisteredModelTag(key="other", value="value"),
        ]
        assert has_prompt_tag(tags) is True

    def test_list_tags_without_prompt_tag(self):
        """Test list tags not containing prompt tag."""
        tags = [RegisteredModelTag(key="other", value="value")]
        assert has_prompt_tag(tags) is False

    def test_empty_list_tags(self):
        """Test empty list tags."""
        assert has_prompt_tag([]) is None

    def test_none_list_tags(self):
        """Test None list tags."""
        assert has_prompt_tag(None) is None 