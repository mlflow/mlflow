import tempfile
import pytest
from pathlib import Path
import os

from clint.config import Config


def test_config_validate_exclude_paths_success():
    """Test that Config loads successfully when all exclude paths exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test directory and file
        test_file = Path(temp_dir) / "test_file.py"
        test_file.touch()
        test_dir = Path(temp_dir) / "test_dir"
        test_dir.mkdir()
        
        # Create pyproject.toml with valid exclude paths
        pyproject = Path(temp_dir) / "pyproject.toml"
        pyproject.write_text(f"""
[tool.clint]
exclude = [
    "{test_file}",
    "{test_dir}"
]
""")
        
        # Change to temp directory and load config
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            config = Config.load()
            assert len(config.exclude) == 2
            assert str(test_file) in config.exclude
            assert str(test_dir) in config.exclude
        finally:
            os.chdir(old_cwd)


def test_config_validate_exclude_paths_failure():
    """Test that Config raises ValueError when exclude paths don't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create pyproject.toml with non-existing exclude paths
        pyproject = Path(temp_dir) / "pyproject.toml"
        pyproject.write_text("""
[tool.clint]
exclude = [
    "non_existing_file.py",
    "non_existing_dir"
]
""")
        
        # Change to temp directory and try to load config
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            with pytest.raises(ValueError) as exc_info:
                Config.load()
            
            error_msg = str(exc_info.value)
            assert "Non-existing paths found in exclude field" in error_msg
            assert "non_existing_file.py" in error_msg
            assert "non_existing_dir" in error_msg
        finally:
            os.chdir(old_cwd)


def test_config_validate_exclude_paths_mixed():
    """Test that Config raises ValueError when some exclude paths don't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create one existing file
        existing_file = Path(temp_dir) / "existing_file.py"
        existing_file.touch()
        
        # Create pyproject.toml with mixed existing/non-existing paths
        pyproject = Path(temp_dir) / "pyproject.toml"
        pyproject.write_text(f"""
[tool.clint]
exclude = [
    "{existing_file}",
    "non_existing_file.py"
]
""")
        
        # Change to temp directory and try to load config
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            with pytest.raises(ValueError) as exc_info:
                Config.load()
            
            error_msg = str(exc_info.value)
            assert "Non-existing paths found in exclude field" in error_msg
            assert "non_existing_file.py" in error_msg
            # Should only contain the non-existing path
            assert str(existing_file) not in error_msg
        finally:
            os.chdir(old_cwd)


def test_config_empty_exclude_list():
    """Test that Config loads successfully when exclude list is empty."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create pyproject.toml with empty exclude list
        pyproject = Path(temp_dir) / "pyproject.toml"
        pyproject.write_text("""
[tool.clint]
exclude = []
""")
        
        # Change to temp directory and load config
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            config = Config.load()
            assert config.exclude == []
        finally:
            os.chdir(old_cwd)


def test_config_no_exclude_field():
    """Test that Config loads successfully when exclude field is not present."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create pyproject.toml without exclude field
        pyproject = Path(temp_dir) / "pyproject.toml"
        pyproject.write_text("""
[tool.clint]
select = ["test-rule"]
""")
        
        # Change to temp directory and load config
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            config = Config.load()
            assert config.exclude == []
        finally:
            os.chdir(old_cwd)