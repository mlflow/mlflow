from pathlib import Path

from clint.config import Config
from clint.index import SymbolIndex
from clint.linter import Location, lint_file
from clint.rules import TrackApiUsageOutermost


def test_track_api_usage_outermost_valid_position(
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
"""
    )
    results = lint_file(tmp_file, config, index)
    # Filter for only TrackApiUsageOutermost violations
    track_violations = [r for r in results if isinstance(r.rule, TrackApiUsageOutermost)]
    assert len(track_violations) == 0


def test_track_api_usage_outermost_invalid_position(
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
"""
    )
    results = lint_file(tmp_file, config, index)
    # Filter for only TrackApiUsageOutermost violations
    track_violations = [r for r in results if isinstance(r.rule, TrackApiUsageOutermost)]
    assert len(track_violations) == 1
    assert track_violations[0].loc == Location(4, 1)


def test_track_api_usage_outermost_multiple_decorators_invalid(
    index: SymbolIndex, config: Config, tmp_path: Path
) -> None:
    """Test @track_api_usage in middle of multiple decorators triggers violation."""
    tmp_file = tmp_path / "file.py"
    tmp_file.write_text(
        """
from mlflow.telemetry.track import track_api_usage

@staticmethod
@track_api_usage
@property
def some_function():
    pass
"""
    )
    results = lint_file(tmp_file, config, index)
    # Filter for only TrackApiUsageOutermost violations
    track_violations = [r for r in results if isinstance(r.rule, TrackApiUsageOutermost)]
    assert len(track_violations) == 1
    assert track_violations[0].loc == Location(4, 1)


def test_track_api_usage_outermost_class_decorator_valid(
    index: SymbolIndex, config: Config, tmp_path: Path
) -> None:
    """Test @track_api_usage as outermost decorator on class does not trigger violation."""
    tmp_file = tmp_path / "file.py"
    tmp_file.write_text(
        """
from mlflow.telemetry.track import track_api_usage

@track_api_usage
@dataclass
class SomeClass:
    pass
"""
    )
    results = lint_file(tmp_file, config, index)
    # Filter for only TrackApiUsageOutermost violations
    track_violations = [r for r in results if isinstance(r.rule, TrackApiUsageOutermost)]
    assert len(track_violations) == 0


def test_track_api_usage_outermost_class_decorator_invalid(
    index: SymbolIndex, config: Config, tmp_path: Path
) -> None:
    """Test @track_api_usage not as outermost decorator on class triggers violation."""
    tmp_file = tmp_path / "file.py"
    tmp_file.write_text(
        """
from mlflow.telemetry.track import track_api_usage

@dataclass
@track_api_usage
class SomeClass:
    pass
"""
    )
    results = lint_file(tmp_file, config, index)
    # Filter for only TrackApiUsageOutermost violations
    track_violations = [r for r in results if isinstance(r.rule, TrackApiUsageOutermost)]
    assert len(track_violations) == 1
    assert track_violations[0].loc == Location(4, 1)


def test_track_api_usage_outermost_no_decorator_no_violation(
    index: SymbolIndex, config: Config, tmp_path: Path
) -> None:
    """Test that functions without @track_api_usage decorator do not trigger violations."""
    tmp_file = tmp_path / "file.py"
    tmp_file.write_text(
        """
@property
def some_function():
    pass

def another_function():
    pass
"""
    )
    results = lint_file(tmp_file, config, index)
    # Filter for only TrackApiUsageOutermost violations
    track_violations = [r for r in results if isinstance(r.rule, TrackApiUsageOutermost)]
    assert len(track_violations) == 0


def test_track_api_usage_outermost_async_function_valid(
    index: SymbolIndex, config: Config, tmp_path: Path
) -> None:
    """Test @track_api_usage as outermost decorator on async function no violation."""
    tmp_file = tmp_path / "file.py"
    tmp_file.write_text(
        """
from mlflow.telemetry.track import track_api_usage

@track_api_usage
async def some_async_function():
    pass
"""
    )
    results = lint_file(tmp_file, config, index)
    # Filter for only TrackApiUsageOutermost violations
    track_violations = [r for r in results if isinstance(r.rule, TrackApiUsageOutermost)]
    assert len(track_violations) == 0


def test_track_api_usage_outermost_async_function_invalid(
    index: SymbolIndex, config: Config, tmp_path: Path
) -> None:
    """Test @track_api_usage not as outermost decorator on async function triggers violation."""
    tmp_file = tmp_path / "file.py"
    tmp_file.write_text(
        """
from mlflow.telemetry.track import track_api_usage

@staticmethod
@track_api_usage
async def some_async_function():
    pass
"""
    )
    results = lint_file(tmp_file, config, index)
    # Filter for only TrackApiUsageOutermost violations
    track_violations = [r for r in results if isinstance(r.rule, TrackApiUsageOutermost)]
    assert len(track_violations) == 1
    assert track_violations[0].loc == Location(4, 1)


def test_track_api_usage_outermost_class_method_valid(
    index: SymbolIndex, config: Config, tmp_path: Path
) -> None:
    """Test @track_api_usage as outermost decorator on class method does not trigger violation."""
    tmp_file = tmp_path / "file.py"
    tmp_file.write_text(
        """
from mlflow.telemetry.track import track_api_usage

class SomeClass:
    @track_api_usage
    @classmethod
    def class_method(cls):
        pass

    @track_api_usage
    @staticmethod
    def static_method():
        pass

    @track_api_usage
    def instance_method(self):
        pass
"""
    )
    results = lint_file(tmp_file, config, index)
    # Filter for only TrackApiUsageOutermost violations
    track_violations = [r for r in results if isinstance(r.rule, TrackApiUsageOutermost)]
    assert len(track_violations) == 0


def test_track_api_usage_outermost_class_method_invalid(
    index: SymbolIndex, config: Config, tmp_path: Path
) -> None:
    """Test @track_api_usage not as outermost decorator on class method triggers violation."""
    tmp_file = tmp_path / "file.py"
    tmp_file.write_text(
        """
from mlflow.telemetry.track import track_api_usage

class SomeClass:
    @classmethod
    @track_api_usage
    def class_method(cls):
        pass

    @staticmethod
    @track_api_usage
    def static_method():
        pass

    @property
    @track_api_usage
    def instance_method(self):
        pass
"""
    )
    results = lint_file(tmp_file, config, index)
    # Filter for only TrackApiUsageOutermost violations
    track_violations = [r for r in results if isinstance(r.rule, TrackApiUsageOutermost)]
    assert len(track_violations) == 3
    # Check locations for each violation
    expected_locations = [(5, 5), (10, 5), (15, 5)]
    actual_locations = [(v.loc.lineno, v.loc.col_offset) for v in track_violations]
    assert sorted(actual_locations) == sorted(expected_locations)
