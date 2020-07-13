import posixpath
from six.moves import urllib

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import DATABASE_ENGINES
from mlflow.utils.validation import _validate_db_type_string

_INVALID_DB_URI_MSG = "Please refer to https://mlflow.org/docs/latest/tracking.html#storage for " \
                      "format specifications."


def is_local_uri(uri):
    """Returns true if this is a local file path (/foo or file:/foo)."""
    scheme = urllib.parse.urlparse(uri).scheme
    return uri != 'databricks' and (scheme == '' or scheme == 'file')


def is_http_uri(uri):
    scheme = urllib.parse.urlparse(uri).scheme
    return scheme == 'http' or scheme == 'https'


def is_databricks_uri(uri):
    """Databricks URIs look like 'databricks' (default profile) or 'databricks://profile'"""
    scheme = urllib.parse.urlparse(uri).scheme
    return scheme == 'databricks' or uri == 'databricks'


def construct_db_uri_from_profile(profile):
    if profile:
        return 'databricks://' + profile


def get_db_info_from_uri(uri):
    """
    Get the Databricks profile specified by the tracking URI (if any), otherwise
    returns None.
    """
    parsed_uri = urllib.parse.urlparse(uri)
    if parsed_uri.scheme == "databricks":
        parsed_path = parsed_uri.path.lstrip('/') or None
        parsed_profile = parsed_uri.netloc
        return parsed_profile, parsed_path
    return None, None


def extract_db_type_from_uri(db_uri):
    """
    Parse the specified DB URI to extract the database type. Confirm the database type is
    supported. If a driver is specified, confirm it passes a plausible regex.
    """
    scheme = urllib.parse.urlparse(db_uri).scheme
    scheme_plus_count = scheme.count('+')

    if scheme_plus_count == 0:
        db_type = scheme
    elif scheme_plus_count == 1:
        db_type, _ = scheme.split('+')
    else:
        error_msg = "Invalid database URI: '%s'. %s" % (db_uri, _INVALID_DB_URI_MSG)
        raise MlflowException(error_msg, INVALID_PARAMETER_VALUE)

    _validate_db_type_string(db_type)

    return db_type


def get_uri_scheme(uri_or_path):
    scheme = urllib.parse.urlparse(uri_or_path).scheme
    if any([scheme.lower().startswith(db) for db in DATABASE_ENGINES]):
        return extract_db_type_from_uri(uri_or_path)
    else:
        return scheme


def extract_and_normalize_path(uri):
    parsed_uri_path = urllib.parse.urlparse(uri).path
    normalized_path = posixpath.normpath(parsed_uri_path)
    return normalized_path.lstrip("/")


def append_to_uri_path(uri, *paths):
    """
    Appends the specified POSIX `paths` to the path component of the specified `uri`.

    :param uri: The input URI, represented as a string.
    :param paths: The POSIX paths to append to the specified `uri`'s path component.
    :return: A new URI with a path component consisting of the specified `paths` appended to
             the path component of the specified `uri`.

    >>> uri1 = "s3://root/base/path?param=value"
    >>> uri1 = append_to_uri_path(uri1, "some/subpath", "/anotherpath")
    >>> assert uri1 == "s3://root/base/path/some/subpath/anotherpath?param=value"
    >>> uri2 = "a/posix/path"
    >>> uri2 = append_to_uri_path(uri2, "/some", "subpath")
    >>> assert uri2 == "a/posixpath/some/subpath"
    """
    path = ""
    for subpath in paths:
        path = _join_posixpaths_and_append_absolute_suffixes(path, subpath)

    parsed_uri = urllib.parse.urlparse(uri)
    if len(parsed_uri.scheme) == 0:
        # If the input URI does not define a scheme, we assume that it is a POSIX path
        # and join it with the specified input paths
        return _join_posixpaths_and_append_absolute_suffixes(uri, path)

    prefix = ""
    if not parsed_uri.path.startswith("/"):
        # For certain URI schemes (e.g., "file:"), urllib's unparse routine does
        # not preserve the relative URI path component properly. In certain cases,
        # urlunparse converts relative paths to absolute paths. We introduce this logic
        # to circumvent urlunparse's erroneous conversion
        prefix = parsed_uri.scheme + ":"
        parsed_uri = parsed_uri._replace(scheme="")

    new_uri_path = _join_posixpaths_and_append_absolute_suffixes(parsed_uri.path, path)
    new_parsed_uri = parsed_uri._replace(path=new_uri_path)
    return prefix + urllib.parse.urlunparse(new_parsed_uri)


def _join_posixpaths_and_append_absolute_suffixes(prefix_path, suffix_path):
    """
    Joins the POSIX path `prefix_path` with the POSIX path `suffix_path`. Unlike posixpath.join(),
    if `suffix_path` is an absolute path, it is appended to prefix_path.

    >>> result1 = _join_posixpaths_and_append_absolute_suffixes("relpath1", "relpath2")
    >>> assert result1 == "relpath1/relpath2"
    >>> result2 = _join_posixpaths_and_append_absolute_suffixes("relpath", "/absolutepath")
    >>> assert result2 == "relpath/absolutepath"
    >>> result3 = _join_posixpaths_and_append_absolute_suffixes("/absolutepath", "relpath")
    >>> assert result3 == "/absolutepath/relpath"
    >>> result4 = _join_posixpaths_and_append_absolute_suffixes("/absolutepath1", "/absolutepath2")
    >>> assert result4 == "/absolutepath1/absolutepath2"
    """
    if len(prefix_path) == 0:
        return suffix_path

    # If the specified prefix path is non-empty, we must relativize the suffix path by removing
    # the leading slash, if present. Otherwise, posixpath.join() would omit the prefix from the
    # joined path
    suffix_path = suffix_path.lstrip(posixpath.sep)
    return posixpath.join(prefix_path, suffix_path)


def is_databricks_acled_artifacts_uri(artifact_uri):
    _ACLED_ARTIFACT_URI = "databricks/mlflow-tracking/"
    artifact_uri_path = extract_and_normalize_path(artifact_uri)
    return artifact_uri_path.startswith(_ACLED_ARTIFACT_URI)
