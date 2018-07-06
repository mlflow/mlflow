from sys import version_info

import numpy as np
import pandas as pd


PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)


def ndarray2list(ndarray):
    """
    Convert n-dimensional numpy array into nested lists and convert the elements types to native
    python so that the list is json-able using standard json library.
    :param ndarray: numpy array
    :return: list representation of the numpy array with element types convereted to native python
    """
    if len(ndarray.shape) <= 1:
        return [x.item() for x in ndarray]
    return [ndarray2list(ndarray[i, :]) for i in range(0, ndarray.shape[0])]


def get_jsonable_obj(data):
    """Attempt to make the data json-able via standard library.
    Look for some commonly used types that are not jsonable and convert them into json-able ones.
    Unknown data types are returned as is.

    :param data: data to be converted, works with padnas and numpy, rest will be returned as is.
    """
    if isinstance(data, np.ndarray):
        return ndarray2list(data)
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient='records')
    if isinstance(data, pd.Series):
        return pd.DataFrame(data).to_dict(orient='records')
    else:  # by default just return whatever this is and hope for the best
        return data
