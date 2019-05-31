from sys import version_info

import numpy as np
import pandas as pd
from six.moves import urllib

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

from mlflow.store.dbmodels.db_types import DATABASE_ENGINES
from mlflow.utils.annotations import deprecated, experimental, keyword_only
from mlflow.utils.validation import _validate_db_type_string

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)
_INVALID_DB_URI_MSG = "Please refer to https://mlflow.org/docs/latest/tracking.html#storage for " \
                      "format specifications."


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


def get_major_minor_py_version(py_version):
    return ".".join(py_version.split(".")[:2])


def get_unique_resource_id(max_length=None):
    """
    Obtains a unique id that can be included in a resource name. This unique id is a valid
    DNS subname.

    :param max_length: The maximum length of the identifier
    :return: A unique identifier that can be appended to a user-readable resource name to avoid
             naming collisions.
    """
    import uuid
    import base64
    if max_length is not None and max_length <= 0:
        raise ValueError(
            "The specified maximum length for the unique resource id must be positive!")

    uuid_bytes = uuid.uuid4().bytes
    # Use base64 encoding to shorten the UUID length. Note that the replacement of the
    # unsupported '+' symbol maintains uniqueness because the UUID byte string is of a fixed,
    # 16-byte length
    uuid_b64 = base64.b64encode(uuid_bytes)
    if version_info >= (3, 0):
        # In Python3, `uuid_b64` is a `bytes` object. It needs to be
        # converted to a string
        uuid_b64 = uuid_b64.decode("ascii")
    unique_id = uuid_b64.rstrip('=\n').replace("/", "-").replace("+", "AB").lower()
    if max_length is not None:
        unique_id = unique_id[:int(max_length)]
    return unique_id


def get_uri_scheme(uri_or_path):
    scheme = urllib.parse.urlparse(uri_or_path).scheme
    if any([scheme.lower().startswith(db) for db in DATABASE_ENGINES]):
        return extract_db_type_from_uri(uri_or_path)
    else:
        return scheme
