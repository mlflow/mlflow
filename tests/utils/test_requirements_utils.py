import os

from mlflow.utils.requirements_utils import (
    _is_comment,
    _is_empty,
    _is_requirements_file,
    _strip_inline_comment,
    _join_continued_lines,
    _parse_requirements,
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
        rel_req = tmpdir.join("relative.txt")
        abs_req = tmpdir.join("absolute.txt")

        # pip's requirements parser collapses an absolute requirements file path:
        # https://github.com/pypa/pip/issues/10121
        # As a workaround, use a relative path on Windows.
        absolute_req = abs_req.basename if os.name == "nt" else abs_req.strpath
        root_req.write(
            root_req_src.format(relative_req=rel_req.basename, absolute_req=absolute_req)
        )
        rel_req.write("rel-xxx\nrel-yyy")
        abs_req.write("abs-zzz")

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
            "rel-xxx",
            "rel-yyy",
            "abs-zzz",
            "line-cont==1.0",
            "line-cont-space == 1.0",
            "line-cont-blank",
            "line-cont-eof",
        ]
        parsed_reqs = list(_parse_requirements(root_req.basename))
        pip_reqs = [
            req.requirement
            for req in pip_parse_requirements(root_req.basename, session=PipSession())
        ]
        assert pip_reqs == expected_reqs
        assert parsed_reqs == expected_reqs
    finally:
        os.chdir(request.config.invocation_dir)
