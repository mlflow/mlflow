# DO NO IMPORT MLFLOW IN THIS FILE.
# This file is imported by download_cloud_file_chunk.py.
# Importing mlflow is time-consuming and we want to avoid that in artifact download subprocesses.
import os
import random
from functools import lru_cache

import requests
import urllib3
from packaging.version import Version
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError
from urllib3.util import Retry

# Response codes that generally indicate transient network failures and merit client retries,
# based on guidance from cloud service providers
# (https://docs.microsoft.com/en-us/azure/architecture/best-practices/retry-service-specific#general-rest-and-retry-guidelines)
_TRANSIENT_FAILURE_RESPONSE_CODES = frozenset(
    [
        408,  # Request Timeout
        429,  # Too Many Requests
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
    ]
)


class JitteredRetry(Retry):
    """
    urllib3 < 2 doesn't support `backoff_jitter`. This class is a workaround for that.
    """

    def __init__(self, *args, backoff_jitter=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.backoff_jitter = backoff_jitter

    def get_backoff_time(self):
        """
        Source: https://github.com/urllib3/urllib3/commit/214b184923388328919b0a4b0c15bff603aa51be
        """
        backoff_value = super().get_backoff_time()
        if self.backoff_jitter != 0.0:
            backoff_value += random.random() * self.backoff_jitter
        return float(max(0, min(Retry.DEFAULT_BACKOFF_MAX, backoff_value)))


def augmented_raise_for_status(response):
    """Wrap the standard `requests.response.raise_for_status()` method and return reason"""
    try:
        response.raise_for_status()
    except HTTPError as e:
        if response.text:
            raise HTTPError(
                f"{e}. Response text: {response.text}", request=e.request, response=e.response
            )
        else:
            raise e


def download_chunk(*, range_start, range_end, headers, download_path, http_uri):
    combined_headers = {**headers, "Range": f"bytes={range_start}-{range_end}"}

    with cloud_storage_http_request(
        "get",
        http_uri,
        stream=False,
        headers=combined_headers,
        timeout=10,
    ) as response:
        expected_length = response.headers.get("Content-Length")
        if expected_length is not None:
            actual_length = response.raw.tell()
            expected_length = int(expected_length)
            if actual_length < expected_length:
                raise IOError(
                    "Incomplete read ({} bytes read, {} more expected)".format(
                        actual_length, expected_length - actual_length
                    )
                )
        # File will have been created upstream. Use r+b to ensure chunks
        # don't overwrite the entire file.
        augmented_raise_for_status(response)
        with open(download_path, "r+b") as f:
            f.seek(range_start)
            f.write(response.content)


@lru_cache(maxsize=64)
def _cached_get_request_session(
    max_retries,
    backoff_factor,
    backoff_jitter,
    retry_codes,
    raise_on_status,
    # To create a new Session object for each process, we use the process id as the cache key.
    # This is to avoid sharing the same Session object across processes, which can lead to issues
    # such as https://stackoverflow.com/q/3724900.
    _pid,
):
    """
    This function should not be called directly. Instead, use `_get_request_session` below.
    """
    assert 0 <= max_retries < 10
    assert 0 <= backoff_factor < 120

    retry_kwargs = {
        "total": max_retries,
        "connect": max_retries,
        "read": max_retries,
        "redirect": max_retries,
        "status": max_retries,
        "status_forcelist": retry_codes,
        "backoff_factor": backoff_factor,
        "backoff_jitter": backoff_jitter,
        "raise_on_status": raise_on_status,
    }
    urllib3_version = Version(urllib3.__version__)
    if urllib3_version >= Version("1.26.0"):
        retry_kwargs["allowed_methods"] = None
    else:
        retry_kwargs["method_whitelist"] = None

    if urllib3_version < Version("2.0"):
        retry = JitteredRetry(**retry_kwargs)
    else:
        retry = Retry(**retry_kwargs)
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _get_request_session(max_retries, backoff_factor, backoff_jitter, retry_codes, raise_on_status):
    """
    Returns a `Requests.Session` object for making an HTTP request.

    :param max_retries: Maximum total number of retries.
    :param backoff_factor: a time factor for exponential backoff. e.g. value 5 means the HTTP
      request will be retried with interval 5, 10, 20... seconds. A value of 0 turns off the
      exponential backoff.
    :param backoff_jitter: A random jitter to add to the backoff interval.
    :param retry_codes: a list of HTTP response error codes that qualifies for retry.
    :param raise_on_status: whether to raise an exception, or return a response, if status falls
      in retry_codes range and retries have been exhausted.
    :return: requests.Session object.
    """
    return _cached_get_request_session(
        max_retries,
        backoff_factor,
        backoff_jitter,
        retry_codes,
        raise_on_status,
        _pid=os.getpid(),
    )


def _get_http_response_with_retries(
    method,
    url,
    max_retries,
    backoff_factor,
    backoff_jitter,
    retry_codes,
    raise_on_status=True,
    **kwargs,
):
    """
    Performs an HTTP request using Python's `requests` module with an automatic retry policy.

    :param method: a string indicating the method to use, e.g. "GET", "POST", "PUT".
    :param url: the target URL address for the HTTP request.
    :param max_retries: Maximum total number of retries.
    :param backoff_factor: a time factor for exponential backoff. e.g. value 5 means the HTTP
      request will be retried with interval 5, 10, 20... seconds. A value of 0 turns off the
      exponential backoff.
    :param backoff_jitter: A random jitter to add to the backoff interval.
    :param retry_codes: a list of HTTP response error codes that qualifies for retry.
    :param raise_on_status: whether to raise an exception, or return a response, if status falls
      in retry_codes range and retries have been exhausted.
    :param kwargs: Additional keyword arguments to pass to `requests.Session.request()`

    :return: requests.Response object.
    """
    session = _get_request_session(
        max_retries, backoff_factor, backoff_jitter, retry_codes, raise_on_status
    )
    return session.request(method, url, **kwargs)


def cloud_storage_http_request(
    method,
    url,
    max_retries=5,
    backoff_factor=2,
    backoff_jitter=1.0,
    retry_codes=_TRANSIENT_FAILURE_RESPONSE_CODES,
    timeout=None,
    **kwargs,
):
    """
    Performs an HTTP PUT/GET/PATCH request using Python's `requests` module with automatic retry.

    :param method: string of 'PUT' or 'GET' or 'PATCH', specify to do http PUT or GET or PATCH
    :param url: the target URL address for the HTTP request.
    :param max_retries: maximum number of retries before throwing an exception.
    :param backoff_factor: a time factor for exponential backoff. e.g. value 5 means the HTTP
      request will be retried with interval 5, 10, 20... seconds. A value of 0 turns off the
      exponential backoff.
    :param backoff_jitter: A random jitter to add to the backoff interval.
    :param retry_codes: a list of HTTP response error codes that qualifies for retry.
    :param timeout: wait for timeout seconds for response from remote server for connect and
      read request. Default to None owing to long duration operation in read / write.
    :param kwargs: Additional keyword arguments to pass to `requests.Session.request()`

    :return requests.Response object.
    """
    if method.lower() not in ("put", "get", "patch", "delete"):
        raise ValueError("Illegal http method: " + method)
    return _get_http_response_with_retries(
        method,
        url,
        max_retries,
        backoff_factor,
        backoff_jitter,
        retry_codes,
        timeout=timeout,
        **kwargs,
    )
