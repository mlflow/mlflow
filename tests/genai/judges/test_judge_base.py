"""
Unit tests for the Judge base class and factory functions.
"""

import pytest
from unittest.mock import MagicMock, patch

from mlflow.genai.judges import (
    Judge,
    make_judge,
    register_judge,
    load_judge,
    set_judge_alias,
    delete_judge_alias,
    list_judges,
    list_judge_versions,
    get_judge_aliases,
)
from mlflow.genai.judges.registry import _JUDGE_REGISTRY, _JUDGE_ALIASES, _JUDGE_VERSION_COUNTER
from mlflow.genai.scorers.base import ScorerKind


class TestJudgeBase:
    """Tests for the Judge base class."""
    
    def test_judge_initialization(self):
        """Test that a Judge can be initialized with required parameters."""
        judge = Judge(
            name="test_judge",
            instructions="Check if output is valid",
            model="openai/gpt-4o-mini",
            version=1,
        )
        
        assert judge.name == "test_judge"
        assert judge.instructions == "Check if output is valid"
        assert judge.model == "openai/gpt-4o-mini"
        assert judge.version == 1
        assert judge._examples == []
        assert judge.description == "Check if output is valid"
    
    def test_judge_with_examples(self):
        """Test that a Judge can be initialized with examples."""
        examples = [
            {"inputs": {"q": "test"}, "outputs": "answer", "assessment": True},
            {"inputs": {"q": "test2"}, "outputs": "answer2", "assessment": False},
        ]
        
        judge = Judge(
            name="test_judge",
            instructions="Check validity",
            model="openai/gpt-4",
            examples=examples,
        )
        
        assert judge._examples == examples
    
    def test_judge_kind_is_builtin(self):
        """Test that Judge reports itself as a builtin scorer."""
        judge = Judge(
            name="test_judge",
            instructions="Test instructions",
            model="openai/gpt-4",
        )
        
        assert judge.kind == ScorerKind.BUILTIN
    
    def test_judge_align_not_implemented(self):
        """Test that align() raises NotImplementedError."""
        judge = Judge(
            name="test_judge",
            instructions="Test instructions",
            model="openai/gpt-4",
        )
        
        with pytest.raises(NotImplementedError, match="alignment is not yet implemented"):
            judge.align([])
    
    @patch("mlflow.genai.judges.utils.invoke_judge_model")
    def test_judge_call_invokes_model(self, mock_invoke):
        """Test that calling a judge invokes the LLM model."""
        from mlflow.entities.assessment import Feedback
        mock_invoke.return_value = Feedback(name="test_judge", value="yes")
        
        judge = Judge(
            name="test_judge",
            instructions="Check if formal",
            model="openai/gpt-4",
        )
        
        result = judge(
            inputs={"question": "How to reset?"},
            outputs="Just click reset button.",
        )
        
        # Check that invoke_judge_model was called
        mock_invoke.assert_called_once()
        call_args = mock_invoke.call_args
        assert call_args[1]["model"] == "openai/gpt-4"
        assert call_args[1]["name"] == "test_judge"
        assert "Check if formal" in call_args[1]["prompt"]
        
        # Check result
        assert result.name == "test_judge"
        assert result.value == "yes"
    
    def test_judge_prepare_prompt(self):
        """Test prompt preparation with different inputs."""
        judge = Judge(
            name="test_judge",
            instructions="Check formality",
            model="openai/gpt-4",
            examples=[
                {"inputs": {"q": "test"}, "outputs": "Hey!", "assessment": False},
            ]
        )
        
        prompt = judge._prepare_prompt(
            inputs={"question": "How to reset?"},
            outputs="Please navigate to settings.",
            expectations={"formal": True},
        )
        
        # Check prompt contains expected parts
        assert "Instructions: Check formality" in prompt
        assert "Examples:" in prompt
        assert "Example 1:" in prompt
        assert "Current evaluation:" in prompt
        assert "Inputs: {'question': 'How to reset?'}" in prompt
        assert "Outputs: Please navigate to settings." in prompt
        assert "Expectations: {'formal': True}" in prompt


class TestJudgeFactory:
    """Tests for judge factory functions."""
    
    @patch("mlflow.genai.judges.factory.get_default_model")
    def test_make_judge_with_model(self, mock_get_default):
        """Test make_judge with explicit model."""
        judge = make_judge(
            name="formality_judge",
            instructions="Check if formal",
            model="openai/gpt-4",
        )
        
        assert judge.name == "formality_judge"
        assert judge.instructions == "Check if formal"
        assert judge.model == "openai/gpt-4"
        assert judge.version == 1
        
        # Default model should not be called
        mock_get_default.assert_not_called()
    
    @patch("mlflow.genai.judges.factory.get_default_model")
    def test_make_judge_without_model(self, mock_get_default):
        """Test make_judge uses default model when not specified."""
        mock_get_default.return_value = "databricks/default-judge"
        
        judge = make_judge(
            name="test_judge",
            instructions="Test instructions",
        )
        
        assert judge.model == "databricks/default-judge"
        mock_get_default.assert_called_once()
    
    def test_make_judge_from_dspy_not_implemented(self):
        """Test that make_judge_from_dspy raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            from mlflow.genai.judges.factory import make_judge_from_dspy
            make_judge_from_dspy("test", None)


class TestJudgeRegistry:
    """Tests for judge registry functions."""
    
    def setup_method(self):
        """Clear the registry before each test."""
        _JUDGE_REGISTRY.clear()
        _JUDGE_ALIASES.clear()
        _JUDGE_VERSION_COUNTER.clear()
    
    @patch("mlflow.genai.judges.utils.get_default_model")
    def test_register_judge_creates_version(self, mock_get_default):
        """Test registering a new judge creates version 1."""
        mock_get_default.return_value = "databricks/default"
        
        version = register_judge(
            name="test_judge",
            instructions="Test instructions",
        )
        
        assert version.name == "test_judge"
        assert version.version == 1
        assert version.judge.name == "test_judge"
        assert version.judge.version == 1
    
    def test_register_judge_increments_version(self):
        """Test registering same judge name increments version."""
        v1 = register_judge(
            name="test_judge",
            instructions="Instructions v1",
            model="openai/gpt-4",
        )
        
        v2 = register_judge(
            name="test_judge",
            instructions="Instructions v2",
            model="openai/gpt-4",
            commit_message="Updated instructions",
        )
        
        assert v1.version == 1
        assert v2.version == 2
        assert v2.commit_message == "Updated instructions"
    
    def test_load_judge_by_name_and_version(self):
        """Test loading a judge by name and version."""
        register_judge("test_judge", "Instructions", model="openai/gpt-4")
        register_judge("test_judge", "Instructions v2", model="openai/gpt-4")
        
        judge_v1 = load_judge("test_judge", version=1)
        judge_v2 = load_judge("test_judge", version=2)
        
        assert judge_v1.version == 1
        assert judge_v1.instructions == "Instructions"
        assert judge_v2.version == 2
        assert judge_v2.instructions == "Instructions v2"
    
    def test_load_judge_latest_version(self):
        """Test loading latest version when no version specified."""
        register_judge("test_judge", "v1", model="openai/gpt-4")
        register_judge("test_judge", "v2", model="openai/gpt-4")
        register_judge("test_judge", "v3", model="openai/gpt-4")
        
        judge = load_judge("test_judge")
        
        assert judge.version == 3
        assert judge.instructions == "v3"
    
    def test_load_judge_by_uri(self):
        """Test loading judge by URI format."""
        register_judge("test_judge", "Instructions", model="openai/gpt-4")
        register_judge("test_judge", "Instructions v2", model="openai/gpt-4")
        
        judge = load_judge("judges:/test_judge/2")
        
        assert judge.version == 2
        assert judge.instructions == "Instructions v2"
    
    def test_judge_aliases(self):
        """Test setting and using judge aliases."""
        v1 = register_judge("test_judge", "v1", model="openai/gpt-4")
        v2 = register_judge("test_judge", "v2", model="openai/gpt-4")
        
        # Set aliases
        set_judge_alias("test_judge", "production", 1)
        set_judge_alias("test_judge", "staging", 2)
        
        # Load by alias
        prod_judge = load_judge("judges:/test_judge@production")
        staging_judge = load_judge("test_judge", version="staging")
        
        assert prod_judge.version == 1
        assert staging_judge.version == 2
        
        # Get aliases
        aliases = get_judge_aliases("test_judge")
        assert aliases == {"production": 1, "staging": 2}
        
        # Delete alias
        delete_judge_alias("test_judge", "staging")
        aliases = get_judge_aliases("test_judge")
        assert aliases == {"production": 1}
    
    def test_list_judges(self):
        """Test listing all registered judges."""
        register_judge("judge1", "Instructions", model="openai/gpt-4")
        register_judge("judge2", "Instructions", model="openai/gpt-4")
        register_judge("judge3", "Instructions", model="openai/gpt-4")
        
        judges = list_judges()
        assert set(judges) == {"judge1", "judge2", "judge3"}
    
    def test_list_judge_versions(self):
        """Test listing versions of a judge."""
        register_judge("test_judge", "v1", model="openai/gpt-4")
        register_judge("test_judge", "v2", model="openai/gpt-4")
        register_judge("test_judge", "v3", model="openai/gpt-4")
        
        versions = list_judge_versions("test_judge")
        assert versions == [1, 2, 3]
    
    def test_registry_errors(self):
        """Test error conditions in registry functions."""
        # Try to load non-existent judge
        with pytest.raises(ValueError, match="not found in registry"):
            load_judge("non_existent")
        
        # Try to set alias for non-existent judge
        with pytest.raises(ValueError, match="not found in registry"):
            set_judge_alias("non_existent", "prod", 1)
        
        # Try to delete non-existent alias
        register_judge("test_judge", "Instructions", model="openai/gpt-4")
        with pytest.raises(ValueError, match="Alias .* not found"):
            delete_judge_alias("test_judge", "non_existent")


class TestJudgeSerialization:
    """Tests for Judge serialization and deserialization."""
    
    def test_judge_serialization(self):
        """Test that a Judge can be serialized to dictionary."""
        judge = Judge(
            name="test_judge",
            instructions="Check validity",
            model="openai/gpt-4",
            version=2,
            examples=[{"test": "example"}],
        )
        
        serialized = judge.model_dump()
        
        # Check structure
        assert serialized["name"] == "test_judge"
        assert serialized["builtin_scorer_class"] == "mlflow.genai.judges.base.Judge"
        assert serialized["builtin_scorer_pydantic_data"]["instructions"] == "Check validity"
        assert serialized["builtin_scorer_pydantic_data"]["model"] == "openai/gpt-4"
        assert serialized["builtin_scorer_pydantic_data"]["version"] == 2
        assert serialized["builtin_scorer_pydantic_data"]["examples"] == [{"test": "example"}]
    
    def test_judge_deserialization(self):
        """Test that a Judge can be deserialized from dictionary."""
        judge = Judge(
            name="test_judge",
            instructions="Check validity",
            model="openai/gpt-4",
            version=3,
            examples=[{"input": "test", "output": "result"}],
        )
        
        serialized = judge.model_dump()
        deserialized = Judge.model_validate(serialized)
        
        assert deserialized.name == "test_judge"
        assert deserialized.instructions == "Check validity"
        assert deserialized.model == "openai/gpt-4"
        assert deserialized.version == 3
        assert deserialized._examples == [{"input": "test", "output": "result"}]
    
    def test_judge_roundtrip_serialization(self):
        """Test that serialization and deserialization preserve all data."""
        original = Judge(
            name="complex_judge",
            instructions="Complex evaluation instructions with multiple criteria",
            model="anthropic/claude-3",
            version=5,
            examples=[
                {"inputs": {"q": "test1"}, "outputs": "answer1", "assessment": True},
                {"inputs": {"q": "test2"}, "outputs": "answer2", "assessment": False},
            ],
            aggregations=["mean", "max"],
        )
        
        # Serialize and deserialize
        serialized = original.model_dump()
        restored = Judge.model_validate(serialized)
        
        # Check all fields preserved
        assert restored.name == original.name
        assert restored.instructions == original.instructions
        assert restored.model == original.model
        assert restored.version == original.version
        assert restored._examples == original._examples
        assert restored.aggregations == original.aggregations