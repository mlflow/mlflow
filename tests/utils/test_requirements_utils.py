import os
import sys
import importlib
from unittest import mock

import importlib_metadata
import pytest

import mlflow
from mlflow.utils.requirements_utils import (
    _is_comment,
    _is_empty,
    _is_requirements_file,
    _strip_inline_comment,
    _join_continued_lines,
    _parse_requirements,
    _prune_packages,
    _strip_local_version_label,
    _get_installed_version,
    _get_pinned_requirement,
    _infer_requirements,
    _normalize_package_name,
    _PyPIPackageIndex,
)


def test_is_comment():
    assert _is_comment("# comment")
    assert _is_comment("#")
    assert _is_comment("### comment ###")
    assert not _is_comment("comment")
    assert not _is_comment("")


def test_is_empty():
    assert _is_empty("")
    assert not _is_empty(" ")
    assert not _is_empty("a")


def test_is_requirements_file():
    assert _is_requirements_file("-r req.txt")
    assert _is_requirements_file("-r  req.txt")
    assert _is_requirements_file("--requirement req.txt")
    assert _is_requirements_file("--requirement  req.txt")
    assert not _is_requirements_file("req")


def test_strip_inline_comment():
    assert _strip_inline_comment("aaa # comment") == "aaa"
    assert _strip_inline_comment("aaa   # comment") == "aaa"
    assert _strip_inline_comment("aaa #   comment") == "aaa"
    assert _strip_inline_comment("aaa # com1 # com2") == "aaa"
    # Ensure a URI fragment is not stripped
    assert (
        _strip_inline_comment("git+https://git/repo.git#subdirectory=subdir")
        == "git+https://git/repo.git#subdirectory=subdir"
    )


def test_join_continued_lines():
    assert list(_join_continued_lines(["a"])) == ["a"]
    assert list(_join_continued_lines(["a\\", "b"])) == ["ab"]
    assert list(_join_continued_lines(["a\\", "b\\", "c"])) == ["abc"]
    assert list(_join_continued_lines(["a\\", " b"])) == ["a b"]
    assert list(_join_continued_lines(["a\\", " b\\", " c"])) == ["a b c"]
    assert list(_join_continued_lines(["a\\", "\\", "b"])) == ["ab"]
    assert list(_join_continued_lines(["a\\", "b", "c\\", "d"])) == ["ab", "cd"]
    assert list(_join_continued_lines(["a\\", "", "b"])) == ["a", "b"]
    assert list(_join_continued_lines(["a\\"])) == ["a"]
    assert list(_join_continued_lines(["\\", "a"])) == ["a"]


def test_parse_requirements(request, tmpdir):
    """
    Ensures `_parse_requirements` returns the same result as `pip._internal.req.parse_requirements`
    """
    from pip._internal.req import parse_requirements as pip_parse_requirements
    from pip._internal.network.session import PipSession

    root_req_src = """
# No version specifier
noverspec
no-ver-spec

# Version specifiers
verspec<1.0
ver-spec == 2.0

# Environment marker
env-marker; python_version < "3.8"

inline-comm # Inline comment
inlinecomm                        # Inline comment

# Git URIs
git+https://github.com/git/uri
git+https://github.com/sub/dir#subdirectory=subdir

# Requirements files
-r {relative_req}
--requirement {absolute_req}

# Constraints files
-c {relative_con}
--constraint {absolute_con}

# Line continuation
line-cont\
==\
1.0

# Line continuation with spaces
line-cont-space \
== \
1.0

# Line continuation with a blank line
line-cont-blank\

# Line continuation at EOF
line-cont-eof\
""".strip()

    try:
        os.chdir(tmpdir)
        root_req = tmpdir.join("requirements.txt")
        # Requirements files
        rel_req = tmpdir.join("relative_req.txt")
        abs_req = tmpdir.join("absolute_req.txt")
        # Constraints files
        rel_con = tmpdir.join("relative_con.txt")
        abs_con = tmpdir.join("absolute_con.txt")

        # pip's requirements parser collapses an absolute requirements file path:
        # https://github.com/pypa/pip/issues/10121
        # As a workaround, use a relative path on Windows.
        absolute_req = abs_req.basename if os.name == "nt" else abs_req.strpath
        absolute_con = abs_con.basename if os.name == "nt" else abs_con.strpath
        root_req.write(
            root_req_src.format(
                relative_req=rel_req.basename,
                absolute_req=absolute_req,
                relative_con=rel_con.basename,
                absolute_con=absolute_con,
            )
        )
        rel_req.write("rel-req-xxx\nrel-req-yyy")
        abs_req.write("abs-req-zzz")
        rel_con.write("rel-con-xxx\nrel-con-yyy")
        abs_con.write("abs-con-zzz")

        expected_cons = [
            "rel-con-xxx",
            "rel-con-yyy",
            "abs-con-zzz",
        ]

        expected_reqs = [
            "noverspec",
            "no-ver-spec",
            "verspec<1.0",
            "ver-spec == 2.0",
            'env-marker; python_version < "3.8"',
            "inline-comm",
            "inlinecomm",
            "git+https://github.com/git/uri",
            "git+https://github.com/sub/dir#subdirectory=subdir",
            "rel-req-xxx",
            "rel-req-yyy",
            "abs-req-zzz",
            "line-cont==1.0",
            "line-cont-space == 1.0",
            "line-cont-blank",
            "line-cont-eof",
        ]

        parsed_reqs = list(_parse_requirements(root_req.basename, is_constraint=False))
        pip_reqs = list(pip_parse_requirements(root_req.basename, session=PipSession()))
        # Requirements
        assert [r.req_str for r in parsed_reqs if not r.is_constraint] == expected_reqs
        assert [r.requirement for r in pip_reqs if not r.constraint] == expected_reqs
        # Constraints
        assert [r.req_str for r in parsed_reqs if r.is_constraint] == expected_cons
        assert [r.requirement for r in pip_reqs if r.constraint] == expected_cons
    finally:
        os.chdir(request.config.invocation_dir)


def test_normalize_package_name():
    assert _normalize_package_name("abc") == "abc"
    assert _normalize_package_name("ABC") == "abc"
    assert _normalize_package_name("a-b-c") == "a-b-c"
    assert _normalize_package_name("a.b.c") == "a-b-c"
    assert _normalize_package_name("a_b_c") == "a-b-c"
    assert _normalize_package_name("a--b--c") == "a-b-c"
    assert _normalize_package_name("a-._b-._c") == "a-b-c"


def test_prune_packages():
    assert _prune_packages(["mlflow"]) == {"mlflow"}
    assert _prune_packages(["mlflow", "packaging"]) == {"mlflow"}
    assert _prune_packages(["mlflow", "scikit-learn"]) == {"mlflow", "scikit-learn"}


def test_capture_imported_modules():
    from mlflow.utils._capture_modules import _CaptureImportedModules

    with _CaptureImportedModules() as cap:
        # pylint: disable=unused-import,unused-variable
        import math

        __import__("pandas")
        importlib.import_module("numpy")

    assert "math" in cap.imported_modules
    assert "pandas" in cap.imported_modules
    assert "numpy" in cap.imported_modules


def test_strip_local_version_label():
    assert _strip_local_version_label("1.2.3") == "1.2.3"
    assert _strip_local_version_label("1.2.3+ab") == "1.2.3"
    assert _strip_local_version_label("1.2.3rc0+ab") == "1.2.3rc0"
    assert _strip_local_version_label("1.2.3.dev0+ab") == "1.2.3.dev0"
    assert _strip_local_version_label("1.2.3.post0+ab") == "1.2.3.post0"
    assert _strip_local_version_label("invalid") == "invalid"


def test_get_installed_version(tmpdir):
    import numpy as np
    import pandas as pd
    import sklearn

    assert _get_installed_version("mlflow") == mlflow.__version__
    assert _get_installed_version("numpy") == np.__version__
    assert _get_installed_version("pandas") == pd.__version__
    assert _get_installed_version("scikit-learn", module="sklearn") == sklearn.__version__

    not_found_package = tmpdir.join("not_found.py")
    not_found_package.write("__version__ = '1.2.3'")
    sys.path.insert(0, tmpdir.strpath)
    with pytest.raises(importlib_metadata.PackageNotFoundError, match=r".+"):
        importlib_metadata.version("not_found")
    assert _get_installed_version("not_found") == "1.2.3"


def test_get_pinned_requirement(tmpdir):
    assert _get_pinned_requirement("mlflow") == f"mlflow=={mlflow.__version__}"
    assert _get_pinned_requirement("mlflow", version="1.2.3") == "mlflow==1.2.3"

    not_found_package = tmpdir.join("not_found.py")
    not_found_package.write("__version__ = '1.2.3'")
    sys.path.insert(0, tmpdir.strpath)
    with pytest.raises(importlib_metadata.PackageNotFoundError, match=r".+"):
        importlib_metadata.version("not_found")
    assert _get_pinned_requirement("not_found") == "not_found==1.2.3"


def test_get_pinned_requirement_local_version_label(tmpdir):
    package = tmpdir.join("my_package.py")
    lvl = "abc.def.ghi"  # Local version label
    package.write(f"__version__ = '1.2.3+{lvl}'")
    sys.path.insert(0, tmpdir.strpath)

    with mock.patch("mlflow.utils.requirements_utils._logger.warning") as mock_warning:
        req = _get_pinned_requirement("my_package")
        mock_warning.assert_called_once()
        (first_pos_arg,) = mock_warning.call_args[0]
        assert first_pos_arg.startswith(
            f"Found my_package version (1.2.3+{lvl}) contains a local version label (+{lvl})."
        )
    assert req == "my_package==1.2.3"


def test_infer_requirements_excludes_mlflow():
    with mock.patch(
        "mlflow.utils.requirements_utils._capture_imported_modules",
        return_value=["mlflow", "pytest"],
    ):
        mlflow_package = "mlflow-skinny" if "MLFLOW_SKINNY" in os.environ else "mlflow"
        assert mlflow_package in importlib_metadata.packages_distributions()["mlflow"]
        assert _infer_requirements("path/to/model", "sklearn") == [f"pytest=={pytest.__version__}"]


def test_infer_requirements_prints_warning_for_unrecognized_packages():
    with mock.patch(
        "mlflow.utils.requirements_utils._capture_imported_modules",
        return_value=["sklearn"],
    ), mock.patch(
        "mlflow.utils.requirements_utils._PYPI_PACKAGE_INDEX",
        _PyPIPackageIndex(date="2022-01-01", package_names=set()),
    ), mock.patch(
        "mlflow.utils.requirements_utils._logger.warning"
    ) as mock_warning:
        _infer_requirements("path/to/model", "sklearn")

        mock_warning.assert_called_once()
        warning_template = mock_warning.call_args[0][0]
        date, unrecognized_packages = mock_warning.call_args[0][1:3]
        warning_text = warning_template % (date, unrecognized_packages)
        assert "not found in the public PyPI package index" in warning_text
        assert "scikit-learn" in warning_text


def test_infer_requirements_does_not_print_warning_for_recognized_packages():
    with mock.patch(
        "mlflow.utils.requirements_utils._capture_imported_modules",
        return_value=["sklearn"],
    ), mock.patch(
        "mlflow.utils.requirements_utils._PYPI_PACKAGE_INDEX",
        _PyPIPackageIndex(date="2022-01-01", package_names=set(["scikit-learn"])),
    ), mock.patch(
        "mlflow.utils.requirements_utils._logger.warning"
    ) as mock_warning:
        _infer_requirements("path/to/model", "sklearn")
        mock_warning.assert_not_called()
