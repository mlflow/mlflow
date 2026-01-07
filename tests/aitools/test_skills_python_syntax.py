"""
Test that all Python scripts in mlflow/aitools/skills are valid Python code.

This test discovers all .py files under mlflow/aitools/skills and validates
that they can be compiled successfully, ensuring they contain valid Python syntax.
"""

import ast
from pathlib import Path

import pytest


def get_skills_directory() -> Path:
    """Get the path to the mlflow/aitools/skills directory."""
    return Path(__file__).resolve().parents[2] / "mlflow" / "aitools" / "skills"


def discover_python_scripts() -> list[Path]:
    """
    Discover all Python scripts in the skills directory.

    Returns:
        List of paths to .py files found under mlflow/aitools/skills
    """
    skills_dir = get_skills_directory()
    if not skills_dir.exists():
        return []

    # Find all .py files recursively
    python_files = list(skills_dir.rglob("*.py"))
    return sorted(python_files)


def test_skills_directory_exists():
    skills_dir = get_skills_directory()
    assert skills_dir.exists(), f"Skills directory not found: {skills_dir}"
    assert skills_dir.is_dir(), f"Skills path is not a directory: {skills_dir}"


def test_python_scripts_found():
    scripts = discover_python_scripts()
    assert len(scripts) > 0, "No Python scripts found in mlflow/aitools/skills directory"


@pytest.mark.parametrize("script_path", discover_python_scripts())
def test_python_script_syntax(script_path: Path):
    """
    Test that a Python script has valid syntax.

    This test compiles the script to check for syntax errors without executing it.

    Args:
        script_path: Path to the Python script to validate
    """
    try:
        with open(script_path, encoding="utf-8") as f:
            source_code = f.read()

        # Compile the source code to check for syntax errors
        compile(source_code, str(script_path), "exec")

        # Also use ast.parse for a more thorough check
        ast.parse(source_code, filename=str(script_path))

    except SyntaxError as e:
        pytest.fail(
            f"Syntax error in {script_path.relative_to(get_skills_directory().parent.parent)}:\n"
            f"  Line {e.lineno}: {e.msg}\n"
            f"  {e.text}"
        )
    except Exception as e:
        pytest.fail(
            f"Failed to validate {script_path.relative_to(get_skills_directory().parent.parent)}: "
            f"{type(e).__name__}: {e}"
        )


def test_all_scripts_summary():
    """
    Summary test that verifies script discovery.

    This test always passes and tracks the number of discovered scripts.
    """
    scripts = discover_python_scripts()
    # Assert we have scripts to ensure the discovery mechanism is working
    assert len(scripts) > 0, "No Python scripts discovered in mlflow/aitools/skills"
