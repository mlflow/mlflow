import posixpath
import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.utils.uri import (
    add_databricks_profile_info_to_artifact_uri,
    append_to_uri_path,
    construct_run_url,
    extract_and_normalize_path,
    extract_db_type_from_uri,
    get_databricks_profile_uri_from_artifact_uri,
    get_db_info_from_uri,
    get_uri_scheme,
    is_databricks_acled_artifacts_uri,
    is_databricks_uri,
    is_http_uri,
    is_local_uri,
    is_valid_dbfs_uri,
    remove_databricks_profile_info_from_artifact_uri,
    dbfs_hdfs_uri_to_fuse_path,
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
        with pytest.raises(MlflowException, match="Invalid database engine"):
            extract_db_type_from_uri(unsupported_db)


@pytest.mark.parametrize(
    "server_uri, result",
    [
        ("databricks://aAbB", ("aAbB", None)),
        ("databricks://aAbB/", ("aAbB", None)),
        ("databricks://aAbB/path", ("aAbB", None)),
        ("databricks://profile:prefix", ("profile", "prefix")),
        ("databricks://profile:prefix/extra", ("profile", "prefix")),
        ("nondatabricks://profile:prefix", (None, None)),
        ("databricks://profile", ("profile", None)),
        ("databricks://profile/", ("profile", None)),
    ],
)
def test_get_db_info_from_uri(server_uri, result):
    assert get_db_info_from_uri(server_uri) == result


@pytest.mark.parametrize(
    "server_uri",
    ["databricks:/profile:prefix", "databricks:/", "databricks://"],
)
def test_get_db_info_from_uri_errors_no_netloc(server_uri):
    with pytest.raises(MlflowException, match="URI is formatted incorrectly"):
        get_db_info_from_uri(server_uri)


@pytest.mark.parametrize(
    "server_uri",
    [
        "databricks://profile:prefix:extra",
        "databricks://profile:prefix:extra  ",
        "databricks://profile:prefix extra",
        "databricks://profile:prefix  ",
        "databricks://profile ",
        "databricks://profile:",
        "databricks://profile: ",
    ],
)
def test_get_db_info_from_uri_errors_invalid_profile(server_uri):
    with pytest.raises(MlflowException, match="Unsupported Databricks profile"):
        get_db_info_from_uri(server_uri)


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
    with pytest.raises(
        MlflowException, match="Hostname, experiment ID, and run ID are all required"
    ):
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

    def create_char_case(special_char):
        def char_case(*case_args):
            return tuple([item.format(c=special_char) for item in case_args])

        return char_case

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
        char_case = create_char_case(special_char)
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
            ("dbscheme+dbdriver://host/path", "/subpath", "dbscheme+dbdriver://host/path/subpath"),
            ("dbscheme+dbdriver:///path", "subpath", "dbscheme+dbdriver:/path/subpath"),
            ("dbscheme+dbdriver:?somequery", "subpath", "dbscheme+dbdriver:subpath?somequery"),
            ("dbscheme+dbdriver:?somequery", "/subpath", "dbscheme+dbdriver:/subpath?somequery"),
            ("dbscheme+dbdriver:/?somequery", "subpath", "dbscheme+dbdriver:/subpath?somequery"),
            ("dbscheme+dbdriver://?somequery", "subpath", "dbscheme+dbdriver:subpath?somequery"),
            ("dbscheme+dbdriver:///?somequery", "/subpath", "dbscheme+dbdriver:/subpath?somequery"),
            ("dbscheme+dbdriver:#somefrag", "subpath", "dbscheme+dbdriver:subpath#somefrag"),
            ("dbscheme+dbdriver:#somefrag", "/subpath", "dbscheme+dbdriver:/subpath#somefrag"),
            ("dbscheme+dbdriver:/#somefrag", "subpath", "dbscheme+dbdriver:/subpath#somefrag"),
            ("dbscheme+dbdriver://#somefrag", "subpath", "dbscheme+dbdriver:subpath#somefrag"),
            ("dbscheme+dbdriver:///#somefrag", "/subpath", "dbscheme+dbdriver:/subpath#somefrag"),
            (
                "dbscheme+dbdriver://root:password?creds=creds",
                "subpath",
                "dbscheme+dbdriver://root:password/subpath?creds=creds",
            ),
            (
                "dbscheme+dbdriver://root:password/path/?creds=creds",
                "/subpath/anotherpath",
                "dbscheme+dbdriver://root:password/path/subpath/anotherpath?creds=creds",
            ),
            (
                "dbscheme+dbdriver://root:password///path/?creds=creds",
                "subpath/anotherpath",
                "dbscheme+dbdriver://root:password///path/subpath/anotherpath?creds=creds",
            ),
            (
                "dbscheme+dbdriver://root:password///path/?creds=creds",
                "/subpath",
                "dbscheme+dbdriver://root:password///path/subpath?creds=creds",
            ),
            (
                "dbscheme+dbdriver://root:password#myfragment",
                "/subpath",
                "dbscheme+dbdriver://root:password/subpath#myfragment",
            ),
            (
                "dbscheme+dbdriver://root:password//path/#fragmentwith$pecial@",
                "subpath/anotherpath",
                "dbscheme+dbdriver://root:password//path/subpath/anotherpath#fragmentwith$pecial@",
            ),
            (
                "dbscheme+dbdriver://root:password@host?creds=creds#fragmentwith$pecial@",
                "subpath",
                "dbscheme+dbdriver://root:password@host/subpath?creds=creds#fragmentwith$pecial@",
            ),
            (
                "dbscheme+dbdriver://root:password@host.com/path?creds=creds#*frag@*",
                "subpath/dir",
                "dbscheme+dbdriver://root:password@host.com/path/subpath/dir?creds=creds#*frag@*",
            ),
            (
                "dbscheme-dbdriver://root:password@host.com/path?creds=creds#*frag@*",
                "subpath/dir",
                "dbscheme-dbdriver://root:password@host.com/path/subpath/dir?creds=creds#*frag@*",
            ),
            (
                "dbscheme+dbdriver://root:password@host.com/path?creds=creds,param=value#*frag@*",
                "subpath/dir",
                "dbscheme+dbdriver://root:password@host.com/path/subpath/dir?"
                "creds=creds,param=value#*frag@*",
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


@pytest.mark.parametrize(
    "uri, result",
    [
        # URIs with no databricks profile info -> return None
        ("ftp://user:pass@realhost:port/path/to/nowhere", None),
        ("dbfs:/path/to/nowhere", None),
        ("dbfs://nondatabricks/path/to/nowhere", None),
        ("dbfs://incorrect:netloc:format/path/to/nowhere", None),
        # URIs with legit databricks profile info
        ("dbfs://databricks", "databricks"),
        ("dbfs://databricks/", "databricks"),
        ("dbfs://databricks/path/to/nowhere", "databricks"),
        ("dbfs://databricks:port/path/to/nowhere", "databricks"),
        ("dbfs://@databricks/path/to/nowhere", "databricks"),
        ("dbfs://@databricks:port/path/to/nowhere", "databricks"),
        ("dbfs://profile@databricks/path/to/nowhere", "databricks://profile"),
        ("dbfs://profile@databricks:port/path/to/nowhere", "databricks://profile"),
        ("dbfs://scope:key_prefix@databricks/path/abc", "databricks://scope:key_prefix"),
        ("dbfs://scope:key_prefix@databricks:port/path/abc", "databricks://scope:key_prefix"),
        # Doesn't care about the scheme of the artifact URI
        ("runs://scope:key_prefix@databricks/path/abc", "databricks://scope:key_prefix"),
        ("models://scope:key_prefix@databricks/path/abc", "databricks://scope:key_prefix"),
        ("s3://scope:key_prefix@databricks/path/abc", "databricks://scope:key_prefix"),
    ],
)
def test_get_databricks_profile_uri_from_artifact_uri(uri, result):
    assert get_databricks_profile_uri_from_artifact_uri(uri) == result


@pytest.mark.parametrize(
    "uri",
    [
        # Treats secret key prefixes with ":" to be invalid
        "dbfs://incorrect:netloc:format@databricks/path/a",
        "dbfs://scope::key_prefix@databricks/path/abc",
        "dbfs://scope:key_prefix:@databricks/path/abc",
    ],
)
def test_get_databricks_profile_uri_from_artifact_uri_error_cases(uri):
    with pytest.raises(MlflowException, match="Unsupported Databricks profile"):
        get_databricks_profile_uri_from_artifact_uri(uri)


@pytest.mark.parametrize(
    "uri, result",
    [
        # URIs with no databricks profile info should stay the same
        (
            "ftp://user:pass@realhost:port/path/nowhere",
            "ftp://user:pass@realhost:port/path/nowhere",
        ),
        ("dbfs:/path/to/nowhere", "dbfs:/path/to/nowhere"),
        ("dbfs://nondatabricks/path/to/nowhere", "dbfs://nondatabricks/path/to/nowhere"),
        ("dbfs://incorrect:netloc:format/path/", "dbfs://incorrect:netloc:format/path/"),
        # URIs with legit databricks profile info
        ("dbfs://databricks", "dbfs:"),
        ("dbfs://databricks/", "dbfs:/"),
        ("dbfs://databricks/path/to/nowhere", "dbfs:/path/to/nowhere"),
        ("dbfs://databricks:port/path/to/nowhere", "dbfs:/path/to/nowhere"),
        ("dbfs://@databricks/path/to/nowhere", "dbfs:/path/to/nowhere"),
        ("dbfs://@databricks:port/path/to/nowhere", "dbfs:/path/to/nowhere"),
        ("dbfs://profile@databricks/path/to/nowhere", "dbfs:/path/to/nowhere"),
        ("dbfs://profile@databricks:port/path/to/nowhere", "dbfs:/path/to/nowhere"),
        ("dbfs://scope:key_prefix@databricks/path/abc", "dbfs:/path/abc"),
        ("dbfs://scope:key_prefix@databricks:port/path/abc", "dbfs:/path/abc"),
        # Treats secret key prefixes with ":" to be valid
        ("dbfs://incorrect:netloc:format@databricks/path/to/nowhere", "dbfs:/path/to/nowhere"),
        # Doesn't care about the scheme of the artifact URI
        ("runs://scope:key_prefix@databricks/path/abc", "runs:/path/abc"),
        ("models://scope:key_prefix@databricks/path/abc", "models:/path/abc"),
        ("s3://scope:key_prefix@databricks/path/abc", "s3:/path/abc"),
    ],
)
def test_remove_databricks_profile_info_from_artifact_uri(uri, result):
    assert remove_databricks_profile_info_from_artifact_uri(uri) == result


@pytest.mark.parametrize(
    "artifact_uri, profile_uri, result",
    [
        # test various profile URIs
        ("dbfs:/path/a/b", "databricks", "dbfs://databricks/path/a/b"),
        ("dbfs:/path/a/b/", "databricks", "dbfs://databricks/path/a/b/"),
        ("dbfs:/path/a/b/", "databricks://Profile", "dbfs://Profile@databricks/path/a/b/"),
        ("dbfs:/path/a/b/", "databricks://profile/", "dbfs://profile@databricks/path/a/b/"),
        ("dbfs:/path/a/b/", "databricks://scope:key", "dbfs://scope:key@databricks/path/a/b/"),
        (
            "dbfs:/path/a/b/",
            "databricks://scope:key/random_stuff",
            "dbfs://scope:key@databricks/path/a/b/",
        ),
        ("dbfs:/path/a/b/", "nondatabricks://profile", "dbfs:/path/a/b/"),
        # test various artifact schemes
        ("runs:/path/a/b/", "databricks://Profile", "runs://Profile@databricks/path/a/b/"),
        ("runs:/path/a/b/", "nondatabricks://profile", "runs:/path/a/b/"),
        ("models:/path/a/b/", "databricks://profile", "models://profile@databricks/path/a/b/"),
        ("models:/path/a/b/", "nondatabricks://Profile", "models:/path/a/b/"),
        ("s3:/path/a/b/", "databricks://Profile", "s3:/path/a/b/"),
        ("s3:/path/a/b/", "nondatabricks://profile", "s3:/path/a/b/"),
        ("ftp:/path/a/b/", "databricks://profile", "ftp:/path/a/b/"),
        ("ftp:/path/a/b/", "nondatabricks://Profile", "ftp:/path/a/b/"),
        # test artifact URIs already with authority
        ("ftp://user:pass@host:port/a/b", "databricks://Profile", "ftp://user:pass@host:port/a/b"),
        ("ftp://user:pass@host:port/a/b", "nothing://Profile", "ftp://user:pass@host:port/a/b"),
        ("dbfs://databricks", "databricks://OtherProfile", "dbfs://databricks"),
        ("dbfs://databricks", "nondatabricks://Profile", "dbfs://databricks"),
        ("dbfs://databricks/path/a/b", "databricks://OtherProfile", "dbfs://databricks/path/a/b"),
        ("dbfs://databricks/path/a/b", "nondatabricks://Profile", "dbfs://databricks/path/a/b"),
        ("dbfs://@databricks/path/a/b", "databricks://OtherProfile", "dbfs://@databricks/path/a/b"),
        ("dbfs://@databricks/path/a/b", "nondatabricks://Profile", "dbfs://@databricks/path/a/b"),
        (
            "dbfs://profile@databricks/pp",
            "databricks://OtherProfile",
            "dbfs://profile@databricks/pp",
        ),
        (
            "dbfs://profile@databricks/path",
            "databricks://profile",
            "dbfs://profile@databricks/path",
        ),
        (
            "dbfs://profile@databricks/path",
            "nondatabricks://Profile",
            "dbfs://profile@databricks/path",
        ),
    ],
)
def test_add_databricks_profile_info_to_artifact_uri(artifact_uri, profile_uri, result):
    assert add_databricks_profile_info_to_artifact_uri(artifact_uri, profile_uri) == result


@pytest.mark.parametrize(
    "artifact_uri, profile_uri",
    [
        ("dbfs:/path/a/b", "databricks://not:legit:auth"),
        ("dbfs:/path/a/b/", "databricks://scope::key"),
        ("dbfs:/path/a/b/", "databricks://scope:key:/"),
        ("dbfs:/path/a/b/", "databricks://scope:key "),
    ],
)
def test_add_databricks_profile_info_to_artifact_uri_errors(artifact_uri, profile_uri):
    with pytest.raises(MlflowException, match="Unsupported Databricks profile"):
        add_databricks_profile_info_to_artifact_uri(artifact_uri, profile_uri)


@pytest.mark.parametrize(
    "uri, result",
    [
        ("dbfs:/path/a/b", True),
        ("dbfs://databricks/a/b", True),
        ("dbfs://@databricks/a/b", True),
        ("dbfs://profile@databricks/a/b", True),
        ("dbfs://scope:key@databricks/a/b", True),
        ("dbfs://scope:key:@databricks/a/b", False),
        ("dbfs://scope::key@databricks/a/b", False),
        ("dbfs://profile@notdatabricks/a/b", False),
        ("dbfs://scope:key@notdatabricks/a/b", False),
        ("dbfs://scope:key/a/b", False),
        ("dbfs://notdatabricks/a/b", False),
        ("s3:/path/a/b", False),
        ("ftp://user:pass@host:port/path/a/b", False),
        ("ftp://user:pass@databricks/path/a/b", False),
    ],
)
def test_is_valid_dbfs_uri(uri, result):
    assert is_valid_dbfs_uri(uri) == result


@pytest.mark.parametrize(
    "uri, result",
    [
        ("/tmp/path", "/dbfs/tmp/path"),
        ("dbfs:/path", "/dbfs/path"),
        ("dbfs:/path/a/b", "/dbfs/path/a/b"),
        ("dbfs:/dbfs/123/abc", "/dbfs/dbfs/123/abc"),
    ],
)
def test_dbfs_hdfs_uri_to_fuse_path(uri, result):
    assert dbfs_hdfs_uri_to_fuse_path(uri) == result


@pytest.mark.parametrize(
    "path",
    ["some/relative/local/path", "s3:/some/s3/path", "C:/cool/windows/path"],
)
def test_dbfs_hdfs_uri_to_fuse_path_raises(path):
    with pytest.raises(MlflowException, match="did not start with expected DBFS URI prefix"):
        dbfs_hdfs_uri_to_fuse_path(path)
