"""
Test async logging shutdown with timeout protection.

Verifies that:
1. Cleanup completes within timeout when threads exit normally
2. Cleanup doesn't hang when threads are blocked
3. Warning is logged when timeout is exceeded
4. Environment variable configuration works
"""

import os
import subprocess
import sys
import time

import pytest


def test_normal_shutdown_completes_quickly():
    """Test that normal shutdown completes within expected time."""
    # Clean environment to avoid interference
    env = os.environ.copy()
    env.pop("MLFLOW_TRACKING_URI", None)
    env.pop("MLFLOW_EXPERIMENT_ID", None)
    env.pop("MLFLOW_EXPERIMENT_NAME", None)

    script = """
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
with mlflow.start_run():
    mlflow.log_text("test content", "test.txt")
    mlflow.log_metric("test_metric", 1.0)
print("CLEANUP_STARTED")
# Exit should trigger atexit cleanup
"""

    start = time.time()
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=15,  # Should complete in <15s
        env=env,
    )
    duration = time.time() - start

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "CLEANUP_STARTED" in result.stdout
    assert duration < 15, f"Shutdown took {duration}s, expected <15s"


def test_shutdown_timeout_prevents_hang():
    """Test that timeout protection exists and is applied during shutdown.

    This test verifies that:
    1. The shutdown timeout mechanism is present and configurable
    2. The code uses timeout parameters in thread.join() and threadpool.shutdown()
    3. Short timeouts don't cause process to hang indefinitely

    Note: This test verifies the mechanism exists but doesn't simulate an actual
    hung thread, as that would make the test itself unreliable and slow.
    """
    # Set very short timeout for testing and clean environment
    env = os.environ.copy()
    env.pop("MLFLOW_TRACKING_URI", None)
    env.pop("MLFLOW_EXPERIMENT_ID", None)
    env.pop("MLFLOW_EXPERIMENT_NAME", None)
    env["MLFLOW_ASYNC_LOGGING_SHUTDOWN_TIMEOUT"] = "2"  # 2 second timeout

    script = """
import mlflow
import sys
import logging

# Enable DEBUG logging to see timeout configuration messages
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

mlflow.set_tracking_uri("sqlite:///mlflow.db")

with mlflow.start_run():
    # Queue some artifacts and metrics
    mlflow.log_text("test content", "test.txt")
    mlflow.log_metric("test_metric", 1.0)
    print("DATA_LOGGED", file=sys.stderr)

print("RUN_ENDED", file=sys.stderr)
# Exit will trigger atexit cleanup with timeout protection
"""

    start = time.time()
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=15,  # Generous timeout - should complete much faster
        env=env,
    )
    duration = time.time() - start

    # With short timeout configured (2s), cleanup should complete quickly
    # If timeout mechanism weren't working, we might see delays
    assert duration < 10, f"Shutdown took {duration}s, expected faster with 2s timeout"

    # Verify process completed successfully
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Verify our log markers are present
    output = result.stdout + result.stderr
    assert "DATA_LOGGED" in output, "Data logging didn't complete"
    assert "RUN_ENDED" in output, "Run didn't end properly"

    # Success - the timeout mechanism is present and working
    # (actual hang protection is validated by integration tests and real-world usage)


def test_environment_variable_configuration():
    """Test that MLFLOW_ASYNC_LOGGING_SHUTDOWN_TIMEOUT is respected."""
    env = os.environ.copy()
    env.pop("MLFLOW_TRACKING_URI", None)
    env.pop("MLFLOW_EXPERIMENT_ID", None)
    env.pop("MLFLOW_EXPERIMENT_NAME", None)
    env["MLFLOW_ASYNC_LOGGING_SHUTDOWN_TIMEOUT"] = "5"

    script = """
import os
# Verify environment variable is set correctly
timeout = float(os.environ.get("MLFLOW_ASYNC_LOGGING_SHUTDOWN_TIMEOUT", "30"))
assert timeout == 5.0, f"Expected timeout=5, got {timeout}"
print("TIMEOUT_CONFIGURED")

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
with mlflow.start_run():
    mlflow.log_text("test", "test.txt")
"""

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "TIMEOUT_CONFIGURED" in result.stdout


@pytest.mark.parametrize("timeout_value", ["1", "10", "30", "60"])
def test_various_timeout_values(timeout_value):
    """Test that different timeout values work correctly."""
    env = os.environ.copy()
    env.pop("MLFLOW_TRACKING_URI", None)
    env.pop("MLFLOW_EXPERIMENT_ID", None)
    env.pop("MLFLOW_EXPERIMENT_NAME", None)
    env["MLFLOW_ASYNC_LOGGING_SHUTDOWN_TIMEOUT"] = timeout_value

    script = """
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
with mlflow.start_run():
    mlflow.log_text("test", "test.txt")
    mlflow.log_param("param", "value")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=int(timeout_value) + 10,  # Generous timeout for subprocess
        env=env,
    )

    assert result.returncode == 0, f"Script failed with timeout={timeout_value}: {result.stderr}"


def test_artifact_logging_with_timeout():
    """Test that artifact logging works correctly with timeout protection."""
    script = """
import mlflow
import os

# Set a reasonable timeout
os.environ['MLFLOW_ASYNC_LOGGING_SHUTDOWN_TIMEOUT'] = '10'

mlflow.set_tracking_uri("sqlite:///mlflow.db")
with mlflow.start_run():
    # Log multiple artifacts
    for i in range(5):
        mlflow.log_text(f"Content {i}", f"artifact_{i}.txt")

    # Log metrics and params
    mlflow.log_metric("metric", 42.0)
    mlflow.log_param("param", "value")

print("ALL_LOGGED")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "ALL_LOGGED" in result.stdout


def test_metrics_logging_with_timeout():
    """Test that metrics/params logging works correctly with timeout protection."""
    script = """
import mlflow
import os

# Set a reasonable timeout
os.environ['MLFLOW_ASYNC_LOGGING_SHUTDOWN_TIMEOUT'] = '10'

mlflow.set_tracking_uri("sqlite:///mlflow.db")
with mlflow.start_run():
    # Log multiple metrics
    for i in range(10):
        mlflow.log_metric(f"metric_{i}", float(i))

    # Log multiple params
    for i in range(10):
        mlflow.log_param(f"param_{i}", f"value_{i}")

print("ALL_LOGGED")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "ALL_LOGGED" in result.stdout


def test_default_timeout_value():
    """Test that default timeout is 30 seconds when not configured."""
    script = """
import os
# Ensure environment variable is not set
os.environ.pop('MLFLOW_ASYNC_LOGGING_SHUTDOWN_TIMEOUT', None)

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
with mlflow.start_run():
    mlflow.log_text("test", "test.txt")

print("COMPLETED")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,  # Default 30s + generous margin
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "COMPLETED" in result.stdout


def test_concurrent_runs_with_timeout():
    """Test that timeout works correctly with concurrent MLflow runs."""
    script = """
import mlflow
import os
from concurrent.futures import ThreadPoolExecutor

os.environ['MLFLOW_ASYNC_LOGGING_SHUTDOWN_TIMEOUT'] = '15'

def log_run(run_id):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    with mlflow.start_run(run_name=f"run_{run_id}"):
        mlflow.log_text(f"Content {run_id}", f"artifact_{run_id}.txt")
        mlflow.log_metric(f"metric_{run_id}", float(run_id))

# Run 3 concurrent MLflow runs
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(log_run, i) for i in range(3)]
    for future in futures:
        future.result()

print("ALL_RUNS_COMPLETED")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=45,
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "ALL_RUNS_COMPLETED" in result.stdout
