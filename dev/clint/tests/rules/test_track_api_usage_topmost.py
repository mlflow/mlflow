from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import lint_file
from clint.rules import TrackApiUsageTopMost


def test_track_api_usage_topmost_check(index: SymbolIndex, tmp_path: Path) -> None:
    tmp_file = tmp_path / "file.py"
    tmp_file.write_text(
        """
from mlflow.telemetry.track import track_api_usage

# Valid: @track_api_usage is the topmost decorator

@track_api_usage
@property
def some_function():
    pass

@track_api_usage
def another_function():
    pass

class SomeClass:
    @track_api_usage
    @classmethod
    def class_method(cls):
        pass

# Invalid: @track_api_usage is not the outermost decorator

@property
@track_api_usage
def some_function():
    pass

class SomeClass:
    @staticmethod
    @track_api_usage
    def static_method():
        pass

@foo
@track_api_usage
@bar
def some_function():
    pass
"""
    )
    config = Config(select={TrackApiUsageTopMost.name})
    violations = lint_file(tmp_file, config, index)
    assert all(isinstance(v.rule, TrackApiUsageTopMost) for v in violations)
    assert len(violations) == 3
    assert violations[0].loc.lineno == 23
    assert violations[1].loc.lineno == 29
    assert violations[2].loc.lineno == 34
