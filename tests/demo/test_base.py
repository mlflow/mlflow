import pytest

from mlflow.demo.base import (
    DEMO_EXPERIMENT_NAME,
    DEMO_PROMPT_PREFIX,
    BaseDemoGenerator,
    DemoFeature,
    DemoResult,
)


def test_demo_feature_enum():
    assert DemoFeature.TRACES == "traces"
    assert DemoFeature.EVALUATION == "evaluation"
    assert isinstance(DemoFeature.TRACES, str)


def test_demo_result_with_enum():
    result = DemoResult(
        feature=DemoFeature.TRACES,
        entity_ids=["trace-1"],
        navigation_url="/traces",
    )
    assert result.feature == DemoFeature.TRACES
    assert result.feature == "traces"


def test_demo_result_fields():
    result = DemoResult(
        feature=DemoFeature.TRACES,
        entity_ids=["a", "b"],
        navigation_url="/test",
    )
    assert result.feature == DemoFeature.TRACES
    assert result.entity_ids == ["a", "b"]
    assert result.navigation_url == "/test"


def test_constants():
    assert DEMO_EXPERIMENT_NAME == "MLflow Demo"
    assert DEMO_PROMPT_PREFIX == "mlflow-demo"


def test_generator_requires_name():
    class NoNameGenerator(BaseDemoGenerator):
        def generate(self):
            pass

        def _data_exists(self):
            return False

    with pytest.raises(ValueError, match="must define 'name'"):
        NoNameGenerator()


def test_generator_with_name(stub_generator):
    assert stub_generator.name == DemoFeature.TRACES


def test_generator_default_version():
    class VersionedGenerator(BaseDemoGenerator):
        name = DemoFeature.TRACES

        def generate(self):
            pass

        def _data_exists(self):
            return False

    generator = VersionedGenerator()
    assert generator.version == 1


def test_generator_custom_version():
    class CustomVersionGenerator(BaseDemoGenerator):
        name = DemoFeature.EVALUATION
        version = 5

        def generate(self):
            pass

        def _data_exists(self):
            return False

    generator = CustomVersionGenerator()
    assert generator.version == 5


def test_generator_generate_returns_result(stub_generator):
    result = stub_generator.generate()
    assert isinstance(result, DemoResult)
    assert result.feature == DemoFeature.TRACES
    assert stub_generator.generate_called


def test_is_generated_false_when_no_data(stub_generator):
    stub_generator.data_exists_value = False
    assert stub_generator.is_generated() is False


def test_is_generated_true_when_data_and_version_match(stub_generator):
    stub_generator.data_exists_value = True
    stub_generator.stored_version_value = 1
    stub_generator.version = 1
    assert stub_generator.is_generated() is True


def test_is_generated_false_when_version_mismatch(stub_generator):
    stub_generator.data_exists_value = True
    stub_generator.stored_version_value = 1
    stub_generator.version = 2
    assert stub_generator.is_generated() is False


def test_is_generated_calls_delete_demo_on_version_mismatch(stub_generator):
    stub_generator.data_exists_value = True
    stub_generator.stored_version_value = 1
    stub_generator.version = 2
    stub_generator.is_generated()
    assert stub_generator.delete_demo_called is True


def test_is_generated_no_delete_when_no_data(stub_generator):
    stub_generator.data_exists_value = False
    stub_generator.is_generated()
    assert stub_generator.delete_demo_called is False


def test_is_generated_false_when_no_stored_version(stub_generator):
    stub_generator.data_exists_value = True
    stub_generator.stored_version_value = None
    assert stub_generator.is_generated() is False
