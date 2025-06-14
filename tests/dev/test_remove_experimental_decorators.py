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
