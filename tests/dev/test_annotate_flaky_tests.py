import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))

from annotate_flaky_tests import _split_nodeid, annotate_file


def _annotate(tmp_path: Path, source: str, nodeid_suffix: str, attempts: int = 3):
    """Write `source` to a temp file, annotate `<file>::<nodeid_suffix>`, return the result."""
    f = tmp_path / "test_sample.py"
    f.write_text(source)
    _, qualifiers = _split_nodeid(f"x.py::{nodeid_suffix}")
    result = annotate_file(f, qualifiers, attempts, "REPORT")
    return result, f.read_text()


def test_split_nodeid_strips_parametrization_and_keeps_class_chain():
    assert _split_nodeid("tests/f.py::TestX::test_y[case-1]") == ("tests/f.py", ["TestX", "test_y"])
    assert _split_nodeid("tests/f.py::test_z") == ("tests/f.py", ["test_z"])
    assert _split_nodeid("no-colons") is None
    assert _split_nodeid("nota.txt::test") is None


def test_module_level_decorated_test_indents_at_column_zero(tmp_path: Path):
    # Regression: indenting from the decorator's col_offset (after the '@') over-indented
    # the insert into an IndentationError. Must indent from the def's column instead.
    source = (
        'import pytest\n\n\n@pytest.mark.parametrize("x", [1, 2])\ndef test_a(x):\n    assert x\n'
    )
    result, out = _annotate(tmp_path, source, "test_a")
    assert result.applied
    ast.parse(out)  # must stay valid Python
    assert "\n@pytest.mark.flaky(attempts=3)\n@pytest.mark.parametrize" in out


def test_class_name_collision_targets_the_right_method(tmp_path: Path):
    # Regression: a bare ast.walk by name annotated the first matching function. The full
    # '::' qualifier chain must land the marker on TestBeta.test_dupe only.
    source = (
        "import pytest\n\n\n"
        "class TestAlpha:\n    def test_dupe(self):\n        assert True\n\n\n"
        "class TestBeta:\n    def test_dupe(self):\n        assert True\n"
    )
    result, out = _annotate(tmp_path, source, "TestBeta::test_dupe", attempts=2)
    assert result.applied
    ast.parse(out)
    alpha, beta = out.split("class TestBeta:")
    assert "flaky" not in alpha  # TestAlpha untouched
    assert "    @pytest.mark.flaky(attempts=2)\n" in beta  # 4-space method indent


def test_missing_pytest_import_is_added_after_future(tmp_path: Path):
    # Regression: inserting @pytest.mark.flaky into a module without `import pytest`
    # raised NameError at collection. The import must be added, and after __future__.
    source = (
        '"""Doc."""\n\nfrom __future__ import annotations\n\n\ndef test_a():\n    assert True\n'
    )
    result, out = _annotate(tmp_path, source, "test_a")
    assert result.applied
    ast.parse(out)
    assert "import pytest\n" in out
    assert out.index("from __future__") < out.index("import pytest")
    assert out.index("import pytest") < out.index("@pytest.mark.flaky")


def test_existing_pytest_import_not_duplicated(tmp_path: Path):
    source = "import pytest\n\n\ndef test_a():\n    assert True\n"
    _, out = _annotate(tmp_path, source, "test_a")
    ast.parse(out)
    assert out.count("import pytest") == 1


def test_already_flaky_is_skipped(tmp_path: Path):
    source = "import pytest\n\n\n@pytest.mark.flaky(attempts=3)\ndef test_a():\n    assert True\n"
    result, out = _annotate(tmp_path, source, "test_a")
    assert not result.applied
    assert result.note == "already marked flaky"
    assert out.count("flaky") == 1


def test_missing_function_is_reported_not_applied(tmp_path: Path):
    result, out = _annotate(tmp_path, "def test_a():\n    pass\n", "test_does_not_exist")
    assert not result.applied
    assert result.note == "function not found"
    assert "flaky" not in out


def test_unparsable_file_is_handled_gracefully(tmp_path: Path):
    result, out = _annotate(tmp_path, "def test_a(:\n    pass\n", "test_a")
    assert not result.applied
    assert result.note == "file did not parse"
