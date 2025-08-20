import pytest
from unittest.mock import MagicMock, patch

from mlflow.entities.assessment import Feedback
from mlflow.genai.judges import (
    Judge,
    make_judge,
    register_judge,
    load_judge,
    list_judge_versions,
)
from mlflow.genai.judges.factory import make_judge_from_dspy
from mlflow.genai.scorers.base import ScorerKind


def test_judge_initialization():
    judge = Judge(
        name="test_judge",
        instructions="Check if output is valid",
        model="openai/gpt-4o-mini",
    )
    
    assert judge.name == "test_judge"
    assert judge.instructions == "Check if output is valid"
    assert judge.model == "openai/gpt-4o-mini"
    assert judge._examples == []
    assert judge.description == "Check if output is valid"


def test_judge_with_examples():
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


def test_judge_kind_is_builtin():
    judge = Judge(
        name="test_judge",
        instructions="Test instructions",
        model="openai/gpt-4",
    )
    
    assert judge.kind == ScorerKind.BUILTIN


def test_judge_align_not_implemented():
    judge = Judge(
        name="test_judge",
        instructions="Test instructions",
        model="openai/gpt-4",
    )
    
    with pytest.raises(NotImplementedError, match="alignment is not yet implemented"):
        judge.align([])


def test_judge_call_invokes_model():
    with patch("mlflow.genai.judges.utils.invoke_judge_model") as mock_invoke:
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
        
        mock_invoke.assert_called_once()
        call_args = mock_invoke.call_args
        assert call_args[1]["model"] == "openai/gpt-4"
        assert call_args[1]["name"] == "test_judge"
        assert "Check if formal" in call_args[1]["prompt"]
        
        assert result.name == "test_judge"
        assert result.value == "yes"


def test_judge_prepare_prompt():
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
    
    assert "Instructions: Check formality" in prompt
    assert "Examples:" in prompt
    assert "Example 1:" in prompt
    assert "Current evaluation:" in prompt
    assert "Inputs: {'question': 'How to reset?'}" in prompt
    assert "Outputs: Please navigate to settings." in prompt
    assert "Expectations: {'formal': True}" in prompt


def test_make_judge_with_model():
    with patch("mlflow.genai.judges.factory.get_default_model") as mock_get_default:
        judge = make_judge(
            name="formality_judge",
            instructions="Check if formal",
            model="openai/gpt-4",
        )
        
        assert judge.name == "formality_judge"
        assert judge.instructions == "Check if formal"
        assert judge.model == "openai/gpt-4"
        
        mock_get_default.assert_not_called()


def test_make_judge_without_model():
    with patch("mlflow.genai.judges.factory.get_default_model") as mock_get_default:
        mock_get_default.return_value = "databricks/default-judge"
        
        judge = make_judge(
            name="test_judge",
            instructions="Test instructions",
        )
        
        assert judge.model == "databricks/default-judge"
        mock_get_default.assert_called_once()


def test_make_judge_from_dspy_not_implemented():
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        make_judge_from_dspy("test", None)


def test_register_judge_creates_version():
    with patch('mlflow.genai.judges.factory._get_scorer_store') as mock_get_store, \
         patch('mlflow.genai.judges.factory._get_experiment_id') as mock_get_exp:
        mock_store = MagicMock()
        mock_store.register_scorer.return_value = 1
        mock_get_store.return_value = mock_store
        mock_get_exp.return_value = "test_exp"
        
        version = register_judge(
            name="test_judge",
            instructions="Test instructions",
            model="openai/gpt-4",
        )
        
        assert version == 1
        mock_store.register_scorer.assert_called_once()


def test_register_judge_increments_version():
    with patch('mlflow.genai.judges.factory._get_scorer_store') as mock_get_store, \
         patch('mlflow.genai.judges.factory._get_experiment_id') as mock_get_exp:
        mock_store = MagicMock()
        mock_store.register_scorer.side_effect = [1, 2]
        mock_get_store.return_value = mock_store
        mock_get_exp.return_value = "test_exp"
        
        v1 = register_judge(
            name="test_judge",
            instructions="Instructions v1",
            model="openai/gpt-4",
        )
        
        v2 = register_judge(
            name="test_judge",
            instructions="Instructions v2",
            model="openai/gpt-4",
        )
        
        assert v1 == 1
        assert v2 == 2


def test_load_judge_by_name_and_version():
    with patch('mlflow.genai.judges.factory._get_scorer_store') as mock_get_store, \
         patch('mlflow.genai.judges.factory._get_experiment_id') as mock_get_exp:
        mock_store = MagicMock()
        judge1 = Judge(name="test_judge", instructions="Instructions", model="openai/gpt-4")
        judge2 = Judge(name="test_judge", instructions="Instructions v2", model="openai/gpt-4")
        mock_store.get_scorer.side_effect = [judge1, judge2]
        mock_get_store.return_value = mock_store
        mock_get_exp.return_value = "test_exp"
        
        judge_v1 = load_judge("test_judge", version=1)
        judge_v2 = load_judge("test_judge", version=2)
        
        assert judge_v1.instructions == "Instructions"
        assert judge_v2.instructions == "Instructions v2"


def test_load_judge_latest_version():
    with patch('mlflow.genai.judges.factory._get_scorer_store') as mock_get_store, \
         patch('mlflow.genai.judges.factory._get_experiment_id') as mock_get_exp:
        mock_store = MagicMock()
        judge = Judge(name="test_judge", instructions="v3", model="openai/gpt-4")
        mock_store.get_scorer.return_value = judge
        mock_get_store.return_value = mock_store
        mock_get_exp.return_value = "test_exp"
        
        loaded = load_judge("test_judge")
        
        assert loaded.instructions == "v3"
        mock_store.get_scorer.assert_called_with("test_exp", "test_judge", None)


def test_list_judge_versions():
    with patch('mlflow.genai.judges.factory._get_scorer_store') as mock_get_store, \
         patch('mlflow.genai.judges.factory._get_experiment_id') as mock_get_exp:
        mock_store = MagicMock()
        judge1 = Judge(name="test_judge", instructions="v1", model="openai/gpt-4")
        judge2 = Judge(name="test_judge", instructions="v2", model="openai/gpt-4")
        mock_store.list_scorer_versions.return_value = [(judge1, 1), (judge2, 2)]
        mock_get_store.return_value = mock_store
        mock_get_exp.return_value = "test_exp"
        
        versions = list_judge_versions("test_judge")
        assert len(versions) == 2
        assert versions[0][1] == 1
        assert versions[1][1] == 2


def test_judge_serialization():
    judge = Judge(
        name="test_judge",
        instructions="Check validity",
        model="openai/gpt-4",
        examples=[{"test": "example"}],
    )
    
    serialized = judge.model_dump()
    
    assert serialized["name"] == "test_judge"
    assert serialized["builtin_scorer_class"] == "mlflow.genai.judges.base.Judge"
    assert serialized["builtin_scorer_pydantic_data"]["instructions"] == "Check validity"
    assert serialized["builtin_scorer_pydantic_data"]["model"] == "openai/gpt-4"
    assert serialized["builtin_scorer_pydantic_data"]["examples"] == [{"test": "example"}]


def test_judge_deserialization():
    judge = Judge(
        name="test_judge",
        instructions="Check validity",
        model="openai/gpt-4",
        examples=[{"input": "test", "output": "result"}],
    )
    
    serialized = judge.model_dump()
    deserialized = Judge.model_validate(serialized)
    
    assert deserialized.name == "test_judge"
    assert deserialized.instructions == "Check validity"
    assert deserialized.model == "openai/gpt-4"
    assert deserialized._examples == [{"input": "test", "output": "result"}]


def test_judge_roundtrip_serialization():
    original = Judge(
        name="complex_judge",
        instructions="Complex evaluation instructions with multiple criteria",
        model="anthropic/claude-3",
        examples=[
            {"inputs": {"q": "test1"}, "outputs": "answer1", "assessment": True},
            {"inputs": {"q": "test2"}, "outputs": "answer2", "assessment": False},
        ],
        aggregations=["mean", "max"],
    )
    
    serialized = original.model_dump()
    restored = Judge.model_validate(serialized)
    
    assert restored.name == original.name
    assert restored.instructions == original.instructions
    assert restored.model == original.model
    assert restored._examples == original._examples
    assert restored.aggregations == original.aggregations