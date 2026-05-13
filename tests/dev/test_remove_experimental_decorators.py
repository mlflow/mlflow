import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = "dev/remove_experimental_decorators.py"


def test_script_with_specific_file(tmp_path: Path) -> None:
    test_file = tmp_path / "test.py"
    test_file.write_text("""
@experimental(version="1.0.0")
def func():
    pass
""")

    output = subprocess.check_output(
        [sys.executable, SCRIPT_PATH, "--dry-run", test_file], text=True
    )

    assert "Would remove" in output
    assert "experimental(version='1.0.0')" in output
    assert (
        test_file.read_text()
        == """
@experimental(version="1.0.0")
def func():
    pass
"""
    )


def test_script_without_files() -> None:
    subprocess.check_call([sys.executable, SCRIPT_PATH, "--dry-run"])


def test_script_removes_decorator_without_dry_run(tmp_path: Path) -> None:
    test_file = tmp_path / "test.py"
    test_file.write_text("""
@experimental(version="1.0.0")
def func():
    pass
""")

    subprocess.check_call([sys.executable, SCRIPT_PATH, test_file])
    content = test_file.read_text()
    assert (
        content
        == """
def func():
    pass
"""
    )


def test_script_with_multiline_decorator(tmp_path: Path) -> None:
    test_file = tmp_path / "test.py"
    test_file.write_text("""
@experimental(
    version="1.0.0",
)
def func():
    pass
""")

    output = subprocess.check_output([sys.executable, SCRIPT_PATH, test_file], text=True)
    assert "Removed" in output
    assert (
        test_file.read_text()
        == """
def func():
    pass
"""
    )


def test_script_with_multiple_decorators(tmp_path: Path) -> None:
    test_file = tmp_path / "test.py"
    test_file.write_text("""
@experimental(version="1.0.0")
def func1():
    pass

@experimental(version="1.1.0")
class MyClass:
    @experimental(version="1.2.0")
    def method(self):
        pass

def regular_func():
    pass
""")

    output = subprocess.check_output([sys.executable, SCRIPT_PATH, test_file], text=True)
    assert output.count("Removed") == 3  # Should remove all 3 decorators
    content = test_file.read_text()
    assert (
        content
        == """
def func1():
    pass

class MyClass:
    def method(self):
        pass

def regular_func():
    pass
"""
    )


def test_script_with_cutoff_days_argument(tmp_path: Path) -> None:
    test_file = tmp_path / "test.py"
    test_file.write_text("""
@experimental(version="1.0.0")
def func():
    pass
""")

    # Test with a very large cutoff (should not remove anything)
    output = subprocess.check_output(
        [sys.executable, SCRIPT_PATH, "--cutoff-days", "9999", "--dry-run", test_file], text=True
    )
    assert "Would remove" not in output

    # Test with default cutoff (180 days, should remove old decorators)
    output = subprocess.check_output(
        [sys.executable, SCRIPT_PATH, "--dry-run", test_file], text=True
    )
    assert "Would remove" in output

    # Test with explicit cutoff of 180 days
    output = subprocess.check_output(
        [sys.executable, SCRIPT_PATH, "--cutoff-days", "180", "--dry-run", test_file], text=True
    )
    assert "Would remove" in output


def test_skip_preserves_decorator(tmp_path: Path) -> None:
    test_file = tmp_path / "test.py"
    test_file.write_text("""
@experimental(version="1.0.0", skip=True)
def func():
    pass
""")
    original_content = test_file.read_text()

    subprocess.check_call([sys.executable, SCRIPT_PATH, test_file])
    assert test_file.read_text() == original_content


def test_skip_dry_run_shows_skipped(tmp_path: Path) -> None:
    test_file = tmp_path / "test.py"
    test_file.write_text("""
@experimental(version="1.0.0", skip=True)
def func():
    pass
""")

    output = subprocess.check_output(
        [sys.executable, SCRIPT_PATH, "--dry-run", test_file], text=True
    )
    assert "Skipped (skip=True)" in output
    assert "Would remove" not in output


def test_skip_false_still_removed(tmp_path: Path) -> None:
    test_file = tmp_path / "test.py"
    test_file.write_text("""
@experimental(version="1.0.0", skip=False)
def func():
    pass
""")

    subprocess.check_call([sys.executable, SCRIPT_PATH, test_file])
    content = test_file.read_text()
    assert "@experimental" not in content


def test_skip_mixed_file(tmp_path: Path) -> None:
    test_file = tmp_path / "test.py"
    test_file.write_text("""
@experimental(version="1.0.0", skip=True)
def func_keep():
    pass

@experimental(version="1.0.0")
def func_remove():
    pass
""")

    subprocess.check_call([sys.executable, SCRIPT_PATH, test_file])
    content = test_file.read_text()
    assert "func_keep" in content
    assert "func_remove" in content
    assert (
        content
        == """
@experimental(version="1.0.0", skip=True)
def func_keep():
    pass

def func_remove():
    pass
"""
    )
