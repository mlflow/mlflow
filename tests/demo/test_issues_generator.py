from mlflow.demo.base import DemoFeature
from mlflow.demo.generators.issues import IssuesDemoGenerator


def test_generator_attributes():
    generator = IssuesDemoGenerator()

    assert generator.name == DemoFeature.ISSUES
