from pathlib import Path

from tests import conftest


def test_generate_duration_stats_limits_to_top_30():
    # Clear any existing test results
    conftest._test_results.clear()

    # Create 50 test results for different files, each with duration >= 1.0s
    # to ensure they all pass the duration threshold
    base_path = Path.cwd() / "tests"
    for i in range(50):
        conftest._test_results.append(
            conftest.TestResult(
                path=base_path / f"test_file_{i:02d}.py",
                test_name=f"test_{i}",
                execution_time=float(50 - i),  # Decreasing durations from 50 to 1
            )
        )

    result = conftest.generate_duration_stats()

    # Count the number of data rows (excluding header and separator)
    lines = result.strip().split("\n")
    # First line is header, second line is separator, rest are data rows
    data_rows = lines[2:] if len(lines) > 2 else []

    assert len(data_rows) == 30, f"Expected 30 rows, got {len(data_rows)}"

    # Verify the result contains the top 30 files
    # Check that file_00 (duration 50) is in the result (rank 1)
    assert "test_file_00.py" in result
    # Check that file_29 (duration 21) is in the result (rank 30)
    assert "test_file_29.py" in result
    # Check that file_30 (duration 20) is NOT in the result
    assert "test_file_30.py" not in result
    # Check that file_49 (duration 1) is NOT in the result
    assert "test_file_49.py" not in result

    # Clean up
    conftest._test_results.clear()


def test_generate_duration_stats_filters_files_under_1s():
    conftest._test_results.clear()

    base_path = Path.cwd() / "tests"
    # Create test results where some files have duration < 1s
    for i in range(10):
        conftest._test_results.append(
            conftest.TestResult(
                path=base_path / "test_slow.py",
                test_name=f"test_{i}",
                execution_time=2.0,  # Total for this file: 20s
            )
        )
    for i in range(10):
        conftest._test_results.append(
            conftest.TestResult(
                path=base_path / "test_fast.py",
                test_name=f"test_{i}",
                execution_time=0.05,  # Total for this file: 0.5s (< 1s threshold)
            )
        )

    result = conftest.generate_duration_stats()

    # Only test_slow.py should appear
    assert "test_slow.py" in result
    assert "test_fast.py" not in result

    # Clean up
    conftest._test_results.clear()


def test_generate_duration_stats_empty_results():
    conftest._test_results.clear()

    result = conftest.generate_duration_stats()

    assert result == ""


def test_generate_duration_stats_all_files_under_threshold():
    conftest._test_results.clear()

    base_path = Path.cwd() / "tests"
    # Create test results where all files have duration < 1s
    for i in range(5):
        conftest._test_results.append(
            conftest.TestResult(
                path=base_path / f"test_file_{i}.py",
                test_name=f"test_{i}",
                execution_time=0.1,  # All under 1s
            )
        )

    result = conftest.generate_duration_stats()

    assert result == ""

    # Clean up
    conftest._test_results.clear()
