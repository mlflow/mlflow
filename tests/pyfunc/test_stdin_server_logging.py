from pathlib import Path


def test_stdin_server_does_not_use_global_basicconfig():
    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / "mlflow" / "pyfunc" / "stdin_server.py").read_text(encoding="utf-8")
    assert "logging.basicConfig(" not in source
