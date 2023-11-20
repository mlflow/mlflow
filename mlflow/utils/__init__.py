import base64
import inspect
import logging
import socket
import subprocess
import uuid
from contextlib import closing
from itertools import islice
from sys import version_info

_logger = logging.getLogger(__name__)


PYTHON_VERSION = f"{version_info.major}.{version_info.minor}.{version_info.micro}"


_logger = logging.getLogger(__name__)


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
    if max_length is not None and max_length <= 0:
        raise ValueError(
            "The specified maximum length for the unique resource id must be positive!"
        )

    uuid_bytes = uuid.uuid4().bytes
    # Use base64 encoding to shorten the UUID length. Note that the replacement of the
    # unsupported '+' symbol maintains uniqueness because the UUID byte string is of a fixed,
    # 16-byte length
    uuid_b64 = base64.b64encode(uuid_bytes)
    # In Python3, `uuid_b64` is a `bytes` object. It needs to be
    # converted to a string
    uuid_b64 = uuid_b64.decode("ascii")
    unique_id = uuid_b64.rstrip("=\n").replace("/", "-").replace("+", "AB").lower()
    if max_length is not None:
        unique_id = unique_id[: int(max_length)]
    return unique_id


def reraise(tp, value, tb=None):
    # Taken from: https://github.com/benjaminp/six/blob/1.15.0/six.py#L694-L700
    try:
        if value is None:
            value = tp()
        if value.__traceback__ is not tb:
            raise value.with_traceback(tb)
        raise value
    finally:
        value = None
        tb = None


def chunk_list(l, chunk_size):
    for i in range(0, len(l), chunk_size):
        yield l[i : i + chunk_size]


def _chunk_dict(d, chunk_size):
    """
    Splits a dictionary into chunks of the specified size.
    Taken from: https://stackoverflow.com/a/22878842
    """
    it = iter(d)
    for _ in range(0, len(d), chunk_size):
        yield {k: d[k] for k in islice(it, chunk_size)}


def _truncate_and_ellipsize(value, max_length):
    """
    Truncates the string representation of the specified value to the specified
    maximum length, if necessary. The end of the string is ellipsized if truncation occurs
    """
    value = str(value)
    if len(value) > max_length:
        return value[: (max_length - 3)] + "..."
    else:
        return value


def _truncate_dict(d, max_key_length=None, max_value_length=None):
    """
    Truncates keys and/or values in a dictionary to the specified maximum length.
    Truncated items will be converted to strings and ellipsized.
    """
    key_is_none = max_key_length is None
    val_is_none = max_value_length is None

    if key_is_none and val_is_none:
        raise ValueError("Must specify at least either `max_key_length` or `max_value_length`")

    truncated = {}
    for k, v in d.items():
        should_truncate_key = (not key_is_none) and (len(str(k)) > max_key_length)
        should_truncate_val = (not val_is_none) and (len(str(v)) > max_value_length)

        new_k = _truncate_and_ellipsize(k, max_key_length) if should_truncate_key else k
        if should_truncate_key:
            # Use the truncated key for warning logs to avoid noisy printing to stdout
            msg = f"Truncated the key `{new_k}`"
            _logger.warning(msg)

        new_v = _truncate_and_ellipsize(v, max_value_length) if should_truncate_val else v
        if should_truncate_val:
            # Use the truncated key and value for warning logs to avoid noisy printing to stdout
            msg = f"Truncated the value of the key `{new_k}`. Truncated value: `{new_v}`"
            _logger.warning(msg)

        truncated[new_k] = new_v

    return truncated


def merge_dicts(dict_a, dict_b, raise_on_duplicates=True):
    """
    This function takes two dictionaries and returns one singular merged dictionary.

    :param dict_a: The first dictionary.
    :param dict_b: The second dictonary.
    :param raise_on_duplicates: If True, the function raises ValueError if there are duplicate keys.
                                Otherwise, duplicate keys in `dict_b` will override the ones in
                                `dict_a`.
    :return: A merged dictionary.
    """
    duplicate_keys = dict_a.keys() & dict_b.keys()
    if raise_on_duplicates and len(duplicate_keys) > 0:
        raise ValueError(f"The two merging dictionaries contains duplicate keys: {duplicate_keys}.")
    return {**dict_a, **dict_b}


def _get_fully_qualified_class_name(obj):
    """
    Obtains the fully qualified class name of the given object.
    """
    return obj.__class__.__module__ + "." + obj.__class__.__name__


def _inspect_original_var_name(var, fallback_name):
    """
    Inspect variable name, will search above frames and fetch the same instance variable name
    in the most outer frame.
    If inspect failed, return fallback_name
    """
    if var is None:
        return fallback_name
    try:
        original_var_name = fallback_name

        frame = inspect.currentframe().f_back
        while frame is not None:
            arg_info = inspect.getargvalues(frame)  # pylint: disable=deprecated-method

            fixed_args = [arg_info.locals[arg_name] for arg_name in arg_info.args]
            varlen_args = list(arg_info.locals[arg_info.varargs]) if arg_info.varargs else []
            keyword_args = (
                list(arg_info.locals[arg_info.keywords].values()) if arg_info.keywords else []
            )

            all_args = fixed_args + varlen_args + keyword_args

            # check whether `var` is in arg list first. If yes, go to check parent frame.
            if any(var is arg for arg in all_args):
                # the var is passed in from caller, check parent frame.
                frame = frame.f_back
                continue

            for var_name, var_val in frame.f_locals.items():
                if var_val is var:
                    original_var_name = var_name
                    break

            break

        return original_var_name

    except Exception:
        return fallback_name


def find_free_port():
    """
    Find free socket port on local machine.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def check_port_connectivity():
    port = find_free_port()
    try:
        with subprocess.Popen(
            ["nc", "-l", "-p", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) as server:
            with subprocess.Popen(
                ["nc", "-zv", "localhost", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ) as client:
                client.wait()
                server.terminate()
                return client.returncode == 0
    except Exception as e:
        _logger.warning("Failed to check port connectivity: %s", e)
        return False


def is_iterator(obj):
    """
    :param obj: any object.
    :return: boolean representing whether or not 'obj' is an iterator.
    """
    return (hasattr(obj, "__next__") or hasattr(obj, "next")) and hasattr(obj, "__iter__")


def _is_in_ipython_notebook():
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except Exception:
        return False


def get_results_from_paginated_fn(paginated_fn, max_results_per_page, max_results=None):
    """
    Gets results by calling the ``paginated_fn`` until either no more results remain or
    the specified ``max_results`` threshold has been reached.

    :param paginated_fn:
    :type paginated_fn: This function is expected to take in the number of results to retrieve
        per page and a pagination token, and return a PagedList object
    :param max_results_per_page:
    :type max_results_per_page: The maximum number of results to retrieve per page
    :param max_results:
    :type max_results: The maximum number of results to retrieve overall. If unspecified,
                       all results will be retrieved.
    :return: Returns a list of entities, as determined by the paginated_fn parameter, with no more
        entities than specified by max_results
    :rtype: list[object]
    """
    all_results = []
    next_page_token = None
    returns_all = max_results is None
    while returns_all or len(all_results) < max_results:
        num_to_get = max_results_per_page if returns_all else max_results - len(all_results)
        if num_to_get < max_results_per_page:
            page_results = paginated_fn(num_to_get, next_page_token)
        else:
            page_results = paginated_fn(max_results_per_page, next_page_token)
        all_results.extend(page_results)
        if hasattr(page_results, "token") and page_results.token:
            next_page_token = page_results.token
        else:
            break
    return all_results


class AttrDict(dict):
    """
    Dict-like object that exposes its keys as attributes.

    Examples
    --------
    >>> d = AttrDict({'a': 1, 'b': 2})
    >>> d.a
    1
    >>> d = AttrDict({'a': 1, 'b': {'c': 3, 'd': 4}})
    >>> d.b.c
    3
    """

    def __getattr__(self, attr):
        value = self[attr]
        if isinstance(value, dict):
            return AttrDict(value)
        return value
