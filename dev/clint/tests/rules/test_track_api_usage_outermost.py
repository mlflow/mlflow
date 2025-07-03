from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import lint_file
from clint.rules import TrackApiUsageOutermost


def test_track_api_usage_outermost_valid(
    index: SymbolIndex, config: Config, tmp_path: Path
) -> None:
    """Test that @track_api_usage as the outermost decorator does not trigger a violation."""
    tmp_file = tmp_path / "file.py"
    tmp_file.write_text(
        """
from mlflow.telemetry.track import track_api_usage

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
"""
    )
    results = lint_file(tmp_file, config, index)
    # Filter for only TrackApiUsageOutermost violations
    track_violations = [r for r in results if isinstance(r.rule, TrackApiUsageOutermost)]
    assert len(track_violations) == 0


def test_track_api_usage_outermost_invalid(
    index: SymbolIndex, config: Config, tmp_path: Path
) -> None:
    """Test that @track_api_usage not as outermost decorator triggers a violation."""
    tmp_file = tmp_path / "file.py"
    tmp_file.write_text(
        """
from mlflow.telemetry.track import track_api_usage

@property
@track_api_usage
def some_function():
    pass

class SomeClass:
    @staticmethod
    @track_api_usage
    def static_method():
        pass
"""
    )
    results = lint_file(tmp_file, config, index)
    # Filter for only TrackApiUsageOutermost violations
    track_violations = [r for r in results if isinstance(r.rule, TrackApiUsageOutermost)]
    assert len(track_violations) == 2
    # Check locations for each violation
    expected_locations = [(4, 1), (10, 5)]
    actual_locations = [(v.loc.lineno, v.loc.col_offset) for v in track_violations]
    assert sorted(actual_locations) == sorted(expected_locations)
