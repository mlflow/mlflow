from pathlib import Path
from unittest.mock import patch

from clint.index import SymbolIndex


def test_symbol_index_build_basic(tmp_path: Path) -> None:
    mlflow_dir = tmp_path / "mlflow"
    mlflow_dir.mkdir()

    test_file = mlflow_dir / "test.py"
    test_file.write_text("def test_function(): pass")

    mock_git_output = "mlflow/test.py\n"

    with (
        patch("clint.index.get_repo_root", return_value=tmp_path) as mock_repo_root,
        patch("subprocess.check_output", return_value=mock_git_output) as mock_check_output,
    ):
        index = SymbolIndex.build()
        assert isinstance(index, SymbolIndex)
        mock_repo_root.assert_called_once()
        mock_check_output.assert_called_once()


def test_symbol_index_build_skips_missing_files(tmp_path: Path) -> None:
    mlflow_dir = tmp_path / "mlflow"
    mlflow_dir.mkdir()

    existing_file = mlflow_dir / "existing.py"
    existing_file.write_text("def existing_function(): pass")

    mock_git_output = "mlflow/existing.py\nmlflow/deleted.py\n"

    with (
        patch("clint.index.get_repo_root", return_value=tmp_path) as mock_repo_root,
        patch("subprocess.check_output", return_value=mock_git_output) as mock_check_output,
    ):
        index = SymbolIndex.build()
        assert isinstance(index, SymbolIndex)
        mock_repo_root.assert_called_once()
        mock_check_output.assert_called_once()
