import pytest

from mlflow.demo.base import DemoFeature, DemoResult
from mlflow.demo.generators.judges import DEMO_JUDGE_PREFIX, JudgesDemoGenerator
from mlflow.genai.scorers.registry import list_scorers


@pytest.fixture
def judges_generator():
    generator = JudgesDemoGenerator()
    original_version = generator.version
    yield generator
    JudgesDemoGenerator.version = original_version


def test_generator_attributes():
    generator = JudgesDemoGenerator()
    assert generator.name == DemoFeature.JUDGES
    assert generator.version == 1


def test_data_exists_false_when_no_judges():
    generator = JudgesDemoGenerator()
    assert generator._data_exists() is False


def test_generate_creates_judges():
    generator = JudgesDemoGenerator()
    result = generator.generate()

    assert isinstance(result, DemoResult)
    assert result.feature == DemoFeature.JUDGES
    assert any("judges:" in e for e in result.entity_ids)
    assert "/judges" in result.navigation_url


def test_generate_creates_expected_judges():
    generator = JudgesDemoGenerator()
    generator.generate()

    import mlflow

    experiment = mlflow.get_experiment_by_name("MLflow Demo")
    scorers = list_scorers(experiment_id=experiment.experiment_id)
    demo_judges = [s for s in scorers if s.name.startswith(DEMO_JUDGE_PREFIX)]

    assert len(demo_judges) == 4

    judge_names = {s.name for s in demo_judges}
    expected_names = {
        f"{DEMO_JUDGE_PREFIX}.relevance",
        f"{DEMO_JUDGE_PREFIX}.correctness",
        f"{DEMO_JUDGE_PREFIX}.groundedness",
        f"{DEMO_JUDGE_PREFIX}.safety",
    }
    assert judge_names == expected_names


def test_data_exists_true_after_generate():
    generator = JudgesDemoGenerator()
    assert generator._data_exists() is False

    generator.generate()

    assert generator._data_exists() is True


def test_delete_demo_removes_judges():
    generator = JudgesDemoGenerator()
    generator.generate()
    assert generator._data_exists() is True

    generator.delete_demo()

    assert generator._data_exists() is False


def test_is_generated_checks_version(judges_generator):
    judges_generator.generate()
    judges_generator.store_version()

    assert judges_generator.is_generated() is True

    JudgesDemoGenerator.version = 99
    fresh_generator = JudgesDemoGenerator()
    assert fresh_generator.is_generated() is False
