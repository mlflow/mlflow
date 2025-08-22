import subprocess
import sys


def test_cli():
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "mcp",
            "run",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as proc:
        try:
            stdout, stderr = proc.communicate()
            assert "Starting MCP server" in stdout + stderr
        finally:
            proc.terminate()
