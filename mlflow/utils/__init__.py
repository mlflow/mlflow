from sys import version_info

import numpy as np
import pandas as pd


PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)


def get_jsonable_obj(data, pandas_orient="records"):
    """Attempt to make the data json-able via standard library.
    Look for some commonly used types that are not jsonable and convert them into json-able ones.
    Unknown data types are returned as is.

    :param data: data to be converted, works with pandas and numpy, rest will be returned as is.
    :param pandas_orient: If `data` is a Pandas DataFrame, it will be converted to a JSON
                          dictionary using this Pandas serialization orientation.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient=pandas_orient)
    if isinstance(data, pd.Series):
        return pd.DataFrame(data).to_dict(orient=pandas_orient)
    else:  # by default just return whatever this is and hope for the best
        return data


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
