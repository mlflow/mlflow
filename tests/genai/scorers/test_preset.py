import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.base import Scorer, scorer
from mlflow.genai.scorers.preset import Agent, ConversationalAgent, Preset, Rag
from mlflow.genai.scorers.validation import validate_scorers


class TestPresetConstruction:
    def test_basic_construction(self):
        from mlflow.genai.scorers.builtin_scorers import Completeness, Safety

        preset = Preset("my_preset", scorers=[Safety(), Completeness()])
        assert preset.name == "my_preset"
        assert len(preset) == 2

    def test_empty_name_raises(self):
        from mlflow.genai.scorers.builtin_scorers import Safety

        with pytest.raises(MlflowException, match="non-empty string"):
            Preset("", scorers=[Safety()])

    def test_non_string_name_raises(self):
        from mlflow.genai.scorers.builtin_scorers import Safety

        with pytest.raises(MlflowException, match="non-empty string"):
            Preset(123, scorers=[Safety()])

    def test_empty_scorers_raises(self):
        with pytest.raises(MlflowException, match="non-empty list"):
            Preset("test", scorers=[])

    def test_non_list_scorers_raises(self):
        with pytest.raises(MlflowException, match="non-empty list"):
            Preset("test", scorers="not a list")

    def test_non_scorer_in_list_raises(self):
        with pytest.raises(MlflowException, match="Scorer instances"):
            Preset("test", scorers=["not a scorer"])

    def test_duplicate_scorers_raises(self):
        from mlflow.genai.scorers.builtin_scorers import Safety

        with pytest.raises(MlflowException, match="Duplicate scorer"):
            Preset("test", scorers=[Safety(), Safety()])

    def test_same_type_different_names_allowed(self):
        from mlflow.genai.scorers.builtin_scorers import Guidelines

        preset = Preset(
            "test",
            scorers=[
                Guidelines(name="tone", guidelines=["Be professional"]),
                Guidelines(name="style", guidelines=["Be concise"]),
            ],
        )
        assert len(preset) == 2

    def test_custom_scorers_accepted(self):
        @scorer
        def my_scorer(inputs, outputs):
            return 1.0

        from mlflow.genai.scorers.builtin_scorers import Safety

        preset = Preset("test", scorers=[Safety(), my_scorer])
        assert len(preset) == 2


class TestPresetImmutability:
    def test_scorers_returns_copy(self):
        from mlflow.genai.scorers.builtin_scorers import Completeness, Safety

        preset = Preset("test", scorers=[Safety(), Completeness()])
        scorers1 = preset.scorers
        scorers2 = preset.scorers
        assert scorers1 is not scorers2
        assert len(scorers1) == len(scorers2) == 2

    def test_mutating_returned_list_does_not_affect_preset(self):
        from mlflow.genai.scorers.builtin_scorers import Completeness, Safety

        preset = Preset("test", scorers=[Safety(), Completeness()])
        scorers = preset.scorers
        scorers.clear()
        assert len(preset) == 2
        assert len(preset.scorers) == 2


class TestPresetIteration:
    def test_iter(self):
        from mlflow.genai.scorers.builtin_scorers import Completeness, Safety

        preset = Preset("test", scorers=[Safety(), Completeness()])
        names = [s.name for s in preset]
        assert "safety" in names
        assert "completeness" in names

    def test_len(self):
        from mlflow.genai.scorers.builtin_scorers import Completeness, Fluency, Safety

        preset = Preset("test", scorers=[Safety(), Completeness(), Fluency()])
        assert len(preset) == 3

    def test_repr(self):
        from mlflow.genai.scorers.builtin_scorers import Safety

        preset = Preset("my_preset", scorers=[Safety()])
        r = repr(preset)
        assert "my_preset" in r
        assert "safety" in r


class TestBuiltinPresets:
    @pytest.mark.parametrize(
        ("preset_cls", "expected_count", "expected_names"),
        [
            (
                Rag,
                5,
                {
                    "retrieval_relevance",
                    "retrieval_groundedness",
                    "relevance_to_query",
                    "safety",
                    "completeness",
                },
            ),
            (
                Agent,
                5,
                {
                    "tool_call_correctness",
                    "tool_call_efficiency",
                    "relevance_to_query",
                    "safety",
                    "completeness",
                },
            ),
            (
                ConversationalAgent,
                10,
                {
                    "tool_call_correctness",
                    "tool_call_efficiency",
                    "relevance_to_query",
                    "safety",
                    "completeness",
                    "user_frustration",
                    "conversation_completeness",
                    "conversational_safety",
                    "conversational_tool_call_efficiency",
                    "knowledge_retention",
                },
            ),
        ],
    )
    def test_builtin_preset_scorers(self, preset_cls, expected_count, expected_names):
        preset = preset_cls()
        assert len(preset) == expected_count
        actual_names = {s.name for s in preset}
        assert actual_names == expected_names

    def test_builtin_presets_create_fresh_instances(self):
        preset1 = Agent()
        preset2 = Agent()
        for s1, s2 in zip(preset1, preset2):
            assert s1 is not s2

    @pytest.mark.parametrize("preset_cls", [Rag, Agent, ConversationalAgent])
    def test_model_param_propagation(self, preset_cls):
        preset = preset_cls(model="openai:/gpt-4o")
        for s in preset:
            assert s.model == "openai:/gpt-4o"

    @pytest.mark.parametrize("preset_cls", [Rag, Agent, ConversationalAgent])
    def test_default_model_is_none(self, preset_cls):
        preset = preset_cls()
        for s in preset:
            assert s.model is None

    def test_rag_preset_name(self):
        assert Rag().name == "rag"

    def test_agent_preset_name(self):
        assert Agent().name == "agent"

    def test_conversational_agent_preset_name(self):
        assert ConversationalAgent().name == "conversational_agent"

    def test_conversational_agent_contains_all_agent_scorers(self):
        agent_names = {s.name for s in Agent()}
        conv_names = {s.name for s in ConversationalAgent()}
        assert agent_names.issubset(conv_names)


class TestValidateScorersWithPresets:
    def test_preset_flattened_in_validate_scorers(self):
        preset = Agent()
        result = validate_scorers([preset])
        assert len(result) == 5
        assert all(isinstance(s, Scorer) for s in result)

    def test_preset_mixed_with_individual_scorers(self):
        from mlflow.genai.scorers.builtin_scorers import Fluency

        preset = Rag()
        result = validate_scorers([preset, Fluency()])
        assert len(result) == 6
        assert all(isinstance(s, Scorer) for s in result)

    def test_multiple_presets_flattened(self):
        result = validate_scorers([Rag(), Agent()])
        assert len(result) == 10
        assert all(isinstance(s, Scorer) for s in result)

    def test_preset_scorers_property_works_directly(self):
        preset = Agent()
        result = validate_scorers(preset.scorers)
        assert len(result) == 5


class TestPresetImports:
    def test_import_from_scorers_module(self):
        from mlflow.genai.scorers import Agent, ConversationalAgent, Preset, Rag

        assert Preset is not None
        assert Rag is not None
        assert Agent is not None
        assert ConversationalAgent is not None


class TestPresetRegisterStub:
    def test_register_raises_not_implemented(self):
        from mlflow.genai.scorers.builtin_scorers import Safety

        preset = Preset("test", scorers=[Safety()])
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            preset.register()

    def test_copy_raises_not_implemented(self):
        from mlflow.genai.scorers.builtin_scorers import Safety

        preset = Preset("test", scorers=[Safety()])
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            preset.copy(to_experiment_id="123")
