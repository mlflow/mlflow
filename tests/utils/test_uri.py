import posixpath
import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.utils.uri import (
    is_databricks_uri,
    is_http_uri,
    is_local_uri,
    extract_db_type_from_uri,
    get_uri_scheme,
    append_to_uri_path,
    extract_and_normalize_path,
    is_databricks_acled_artifacts_uri,
    get_db_info_from_uri,
    construct_run_url,
)


def test_extract_db_type_from_uri():
    uri = "{}://username:password@host:port/database"
    for legit_db in DATABASE_ENGINES:
        assert legit_db == extract_db_type_from_uri(uri.format(legit_db))
        assert legit_db == get_uri_scheme(uri.format(legit_db))

        with_driver = legit_db + "+driver-string"
        assert legit_db == extract_db_type_from_uri(uri.format(with_driver))
        assert legit_db == get_uri_scheme(uri.format(with_driver))

    for unsupported_db in ["a", "aa", "sql"]:
        with pytest.raises(MlflowException):
            extract_db_type_from_uri(unsupported_db)


@pytest.mark.parametrize(
    "server_uri, result",
    [
        ("databricks://aAbB", ("aAbB", None)),
        ("databricks://profile/prefix", ("profile", "prefix")),
        ("nondatabricks://profile/prefix", (None, None)),
        ("databricks://profile", ("profile", None)),
        ("databricks://profile/", ("profile", None)),
        ("databricks://", ("", None)),
        ("databricks://aAbB/", ("aAbB", None)),
    ],
)
def test_get_db_info_from_uri(server_uri, result):
    assert get_db_info_from_uri(server_uri) == result


@pytest.mark.parametrize(
    "hostname, experiment_id, run_id, workspace_id, result",
    [
        (
            "https://www.databricks.com/",
            "19201",
            "2231",
            "12211",
            "https://www.databricks.com/?o=12211#mlflow/experiments/19201/runs/2231",
        ),
        (
            "https://www.databricks.com/",
            "19201",
            "2231",
            None,
            "https://www.databricks.com/#mlflow/experiments/19201/runs/2231",
        ),
        (
            "https://www.databricks.com/",
            "19201",
            "2231",
            "0",
            "https://www.databricks.com/#mlflow/experiments/19201/runs/2231",
        ),
        (
            "https://www.databricks.com/",
            "19201",
            "2231",
            "0",
            "https://www.databricks.com/#mlflow/experiments/19201/runs/2231",
        ),
    ],
)
def test_construct_run_url(hostname, experiment_id, run_id, workspace_id, result):
    assert construct_run_url(hostname, experiment_id, run_id, workspace_id) == result


@pytest.mark.parametrize(
    "hostname, experiment_id, run_id, workspace_id",
    [
        (None, "19201", "2231", "0"),
        ("https://www.databricks.com/", None, "2231", "0"),
        ("https://www.databricks.com/", "19201", None, "0"),
    ],
)
def test_construct_run_url_errors(hostname, experiment_id, run_id, workspace_id):
    with pytest.raises(MlflowException):
        construct_run_url(hostname, experiment_id, run_id, workspace_id)


def test_uri_types():
    assert is_local_uri("mlruns")
    assert is_local_uri("./mlruns")
    assert is_local_uri("file:///foo/mlruns")
    assert is_local_uri("file:foo/mlruns")
    assert not is_local_uri("https://whatever")
    assert not is_local_uri("http://whatever")
    assert not is_local_uri("databricks")
    assert not is_local_uri("databricks:whatever")
    assert not is_local_uri("databricks://whatever")

    assert is_databricks_uri("databricks")
    assert is_databricks_uri("databricks:whatever")
    assert is_databricks_uri("databricks://whatever")
    assert not is_databricks_uri("mlruns")
    assert not is_databricks_uri("http://whatever")

    assert is_http_uri("http://whatever")
    assert is_http_uri("https://whatever")
    assert not is_http_uri("file://whatever")
    assert not is_http_uri("databricks://whatever")
    assert not is_http_uri("mlruns")


def validate_append_to_uri_path_test_cases(cases):
    for input_uri, input_path, expected_output_uri in cases:
        assert append_to_uri_path(input_uri, input_path) == expected_output_uri
        assert append_to_uri_path(input_uri, *posixpath.split(input_path)) == expected_output_uri


def test_append_to_uri_path_joins_uri_paths_and_posixpaths_correctly():
    validate_append_to_uri_path_test_cases(
        [
            ("", "path", "path"),
            ("", "/path", "/path"),
            ("path", "", "path/"),
            ("path", "subpath", "path/subpath"),
            ("path/", "subpath", "path/subpath"),
            ("path/", "/subpath", "path/subpath"),
            ("path", "/subpath", "path/subpath"),
            ("/path", "/subpath", "/path/subpath"),
            ("//path", "/subpath", "//path/subpath"),
            ("///path", "/subpath", "///path/subpath"),
            ("/path", "/subpath/subdir", "/path/subpath/subdir"),
            ("file:path", "", "file:path/"),
            ("file:path/", "", "file:path/"),
            ("file:path", "subpath", "file:path/subpath"),
            ("file:path", "/subpath", "file:path/subpath"),
            ("file:/", "", "file:///"),
            ("file:/path", "/subpath", "file:///path/subpath"),
            ("file:///", "", "file:///"),
            ("file:///", "subpath", "file:///subpath"),
            ("file:///path", "/subpath", "file:///path/subpath"),
            ("file:///path/", "subpath", "file:///path/subpath"),
            ("file:///path", "subpath", "file:///path/subpath"),
            ("s3://", "", "s3:"),
            ("s3://", "subpath", "s3:subpath"),
            ("s3://", "/subpath", "s3:/subpath"),
            ("s3://host", "subpath", "s3://host/subpath"),
            ("s3://host", "/subpath", "s3://host/subpath"),
            ("s3://host/", "subpath", "s3://host/subpath"),
            ("s3://host/", "/subpath", "s3://host/subpath"),
            ("s3://host", "subpath/subdir", "s3://host/subpath/subdir"),
        ]
    )


def test_append_to_uri_path_handles_special_uri_characters_in_posixpaths():
    """
    Certain characters are treated specially when parsing and interpreting URIs. However, in the
    case where a URI input for `append_to_uri_path` is simply a POSIX path, these characters should
    not receive special treatment. This test case verifies that `append_to_uri_path` properly joins
    POSIX paths containing these characters.
    """
    for special_char in [
        ".",
        "-",
        "+",
        ":",
        "?",
        "@",
        "&",
        "$",
        "%",
        "/",
        "[",
        "]",
        "(",
        ")",
        "*",
        "'",
        ",",
    ]:

        def char_case(*case_args):
            return tuple([item.format(c=special_char) for item in case_args])

        validate_append_to_uri_path_test_cases(
            [
                char_case("", "{c}subpath", "{c}subpath"),
                char_case("", "/{c}subpath", "/{c}subpath"),
                char_case("dirwith{c}{c}chars", "", "dirwith{c}{c}chars/"),
                char_case("dirwith{c}{c}chars", "subpath", "dirwith{c}{c}chars/subpath"),
                char_case("{c}{c}charsdir", "", "{c}{c}charsdir/"),
                char_case("/{c}{c}charsdir", "", "/{c}{c}charsdir/"),
                char_case("/{c}{c}charsdir", "subpath", "/{c}{c}charsdir/subpath"),
                char_case("/{c}{c}charsdir", "subpath", "/{c}{c}charsdir/subpath"),
            ]
        )

    validate_append_to_uri_path_test_cases(
        [
            ("#?charsdir:", ":?subpath#", "#?charsdir:/:?subpath#"),
            ("/#--+charsdir.//:", "/../:?subpath#", "/#--+charsdir.//:/../:?subpath#"),
            ("$@''(,", ")]*%", "$@''(,/)]*%"),
        ]
    )


def test_append_to_uri_path_preserves_uri_schemes_hosts_queries_and_fragments():
    validate_append_to_uri_path_test_cases(
        [
            ("dbscheme+dbdriver:", "", "dbscheme+dbdriver:"),
            ("dbscheme+dbdriver:", "subpath", "dbscheme+dbdriver:subpath"),
            ("dbscheme+dbdriver:path", "subpath", "dbscheme+dbdriver:path/subpath"),
            ("dbscheme+dbdriver://host/path", "/subpath", "dbscheme+dbdriver://host/path/subpath",),
            ("dbscheme+dbdriver:///path", "subpath", "dbscheme+dbdriver:/path/subpath"),
            ("dbscheme+dbdriver:?somequery", "subpath", "dbscheme+dbdriver:subpath?somequery",),
            ("dbscheme+dbdriver:?somequery", "/subpath", "dbscheme+dbdriver:/subpath?somequery",),
            ("dbscheme+dbdriver:/?somequery", "subpath", "dbscheme+dbdriver:/subpath?somequery",),
            ("dbscheme+dbdriver://?somequery", "subpath", "dbscheme+dbdriver:subpath?somequery",),
            (
                "dbscheme+dbdriver:///?somequery",
                "/subpath",
                "dbscheme+dbdriver:/subpath?somequery",
            ),
            ("dbscheme+dbdriver:#somefrag", "subpath", "dbscheme+dbdriver:subpath#somefrag",),
            ("dbscheme+dbdriver:#somefrag", "/subpath", "dbscheme+dbdriver:/subpath#somefrag",),
            ("dbscheme+dbdriver:/#somefrag", "subpath", "dbscheme+dbdriver:/subpath#somefrag",),
            ("dbscheme+dbdriver://#somefrag", "subpath", "dbscheme+dbdriver:subpath#somefrag",),
            ("dbscheme+dbdriver:///#somefrag", "/subpath", "dbscheme+dbdriver:/subpath#somefrag",),
            (
                "dbscheme+dbdriver://root:password?creds=mycreds",
                "subpath",
                "dbscheme+dbdriver://root:password/subpath?creds=mycreds",
            ),
            (
                "dbscheme+dbdriver://root:password/path/?creds=mycreds",
                "/subpath/anotherpath",
                "dbscheme+dbdriver://root:password/path/subpath/anotherpath?creds=mycreds",
            ),
            (
                "dbscheme+dbdriver://root:password///path/?creds=mycreds",
                "subpath/anotherpath",
                "dbscheme+dbdriver://root:password///path/subpath/anotherpath?creds=mycreds",
            ),
            (
                "dbscheme+dbdriver://root:password///path/?creds=mycreds",
                "/subpath",
                "dbscheme+dbdriver://root:password///path/subpath?creds=mycreds",
            ),
            (
                "dbscheme+dbdriver://root:password#myfragment",
                "/subpath",
                "dbscheme+dbdriver://root:password/subpath#myfragment",
            ),
            (
                "dbscheme+dbdriver://root:password//path/#myfragmentwith$pecial@",
                "subpath/anotherpath",
                "dbscheme+dbdriver://root:password//path/subpath/anotherpath#myfragmentwith$pecial@",
            ),
            (
                "dbscheme+dbdriver://root:password@myhostname?creds=mycreds#myfragmentwith$pecial@",
                "subpath",
                "dbscheme+dbdriver://root:password@myhostname/subpath?creds=mycreds#myfragmentwith$pecial@",
            ),
            (
                "dbscheme+dbdriver://root:password@myhostname.com/path?creds=mycreds#*frag@*",
                "subpath/dir",
                "dbscheme+dbdriver://root:password@myhostname.com/path/subpath/dir?creds=mycreds#*frag@*",
            ),
            (
                "dbscheme-dbdriver://root:password@myhostname.com/path?creds=mycreds#*frag@*",
                "subpath/dir",
                "dbscheme-dbdriver://root:password@myhostname.com/path/subpath/dir?creds=mycreds#*frag@*",
            ),
            (
                "dbscheme+dbdriver://root:password@myhostname.com/path?creds=mycreds,param=value#*frag@*",
                "subpath/dir",
                "dbscheme+dbdriver://root:password@myhostname.com/path/subpath/dir?"
                "creds=mycreds,param=value#*frag@*",
            ),
        ]
    )


def test_extract_and_normalize_path():
    base_uri = "databricks/mlflow-tracking/EXP_ID/RUN_ID/artifacts"
    assert (
        extract_and_normalize_path("dbfs:databricks/mlflow-tracking/EXP_ID/RUN_ID/artifacts")
        == base_uri
    )
    assert (
        extract_and_normalize_path("dbfs:/databricks/mlflow-tracking/EXP_ID/RUN_ID/artifacts")
        == base_uri
    )
    assert (
        extract_and_normalize_path("dbfs:///databricks/mlflow-tracking/EXP_ID/RUN_ID/artifacts")
        == base_uri
    )
    assert (
        extract_and_normalize_path(
            "dbfs:/databricks///mlflow-tracking///EXP_ID///RUN_ID///artifacts/"
        )
        == base_uri
    )
    assert (
        extract_and_normalize_path(
            "dbfs:///databricks///mlflow-tracking//EXP_ID//RUN_ID///artifacts//"
        )
        == base_uri
    )
    assert (
        extract_and_normalize_path(
            "dbfs:databricks///mlflow-tracking//EXP_ID//RUN_ID///artifacts//"
        )
        == base_uri
    )


def test_is_databricks_acled_artifacts_uri():
    assert is_databricks_acled_artifacts_uri(
        "dbfs:databricks/mlflow-tracking/EXP_ID/RUN_ID/artifacts"
    )
    assert is_databricks_acled_artifacts_uri(
        "dbfs:/databricks/mlflow-tracking/EXP_ID/RUN_ID/artifacts"
    )
    assert is_databricks_acled_artifacts_uri(
        "dbfs:///databricks/mlflow-tracking/EXP_ID/RUN_ID/artifacts"
    )
    assert is_databricks_acled_artifacts_uri(
        "dbfs:/databricks///mlflow-tracking///EXP_ID///RUN_ID///artifacts/"
    )
    assert is_databricks_acled_artifacts_uri(
        "dbfs:///databricks///mlflow-tracking//EXP_ID//RUN_ID///artifacts//"
    )
    assert is_databricks_acled_artifacts_uri(
        "dbfs:databricks///mlflow-tracking//EXP_ID//RUN_ID///artifacts//"
    )
    assert not is_databricks_acled_artifacts_uri(
        "dbfs:/databricks/mlflow//EXP_ID//RUN_ID///artifacts//"
    )
