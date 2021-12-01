import logging
from itertools import islice
from sys import version_info


_logger = logging.getLogger(__name__)


PYTHON_VERSION = "{major}.{minor}.{micro}".format(
    major=version_info.major, minor=version_info.minor, micro=version_info.micro
)


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
    import uuid
    import base64

    if max_length is not None and max_length <= 0:
        raise ValueError(
            "The specified maximum length for the unique resource id must be positive!"
        )

    uuid_bytes = uuid.uuid4().bytes
    # Use base64 encoding to shorten the UUID length. Note that the replacement of the
    # unsupported '+' symbol maintains uniqueness because the UUID byte string is of a fixed,
    # 16-byte length
    uuid_b64 = base64.b64encode(uuid_bytes)
    if version_info >= (3, 0):
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
            msg = "Truncated the key `{}`".format(new_k)
            _logger.warning(msg)

        new_v = _truncate_and_ellipsize(v, max_value_length) if should_truncate_val else v
        if should_truncate_val:
            # Use the truncated key and value for warning logs to avoid noisy printing to stdout
            msg = "Truncated the value of the key `{}`. Truncated value: `{}`".format(new_k, new_v)
            _logger.warning(msg)

        truncated[new_k] = new_v

    return truncated


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
    import inspect

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
