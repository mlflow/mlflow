"""
This script should be executed in a fresh python interpreter process using `subprocess`.
"""
import argparse
import json
import os
import requests
import sys
import urllib3

from functools import lru_cache
from packaging.version import Version
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


def download_chunk(range_start, range_end, headers, download_path, http_uri):
    combined_headers = {**headers, "Range": f"bytes={range_start}-{range_end}"}

    with cloud_storage_http_request(
        "get", http_uri, stream=False, headers=combined_headers
    ) as response:
        # File will have been created upstream. Use r+b to ensure chunks
        # don't overwrite the entire file.
        with open(download_path, "r+b") as f:
            f.seek(range_start)
            f.write(response.content)


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


@lru_cache(maxsize=64)
def _cached_get_request_session(
    max_retries,
    backoff_factor,
    retry_codes,
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
    }
    if Version(urllib3.__version__) >= Version("1.26.0"):
        retry_kwargs["allowed_methods"] = None
    else:
        retry_kwargs["method_whitelist"] = None

    retry = Retry(**retry_kwargs)
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _get_request_session(max_retries, backoff_factor, retry_codes):
    """
    Returns a `Requests.Session` object for making an HTTP request.

    :param max_retries: Maximum total number of retries.
    :param backoff_factor: a time factor for exponential backoff. e.g. value 5 means the HTTP
      request will be retried with interval 5, 10, 20... seconds. A value of 0 turns off the
      exponential backoff.
    :param retry_codes: a list of HTTP response error codes that qualifies for retry.
    :return: requests.Session object.
    """
    return _cached_get_request_session(
        max_retries,
        backoff_factor,
        retry_codes,
        _pid=os.getpid(),
    )


def cloud_storage_http_request(
    method,
    url,
    max_retries=5,
    backoff_factor=2,
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
        retry_codes,
        timeout=timeout,
        **kwargs,
    )


def _get_http_response_with_retries(
    method, url, max_retries, backoff_factor, retry_codes, **kwargs
):
    """
    Performs an HTTP request using Python's `requests` module with an automatic retry policy.

    :param method: a string indicating the method to use, e.g. "GET", "POST", "PUT".
    :param url: the target URL address for the HTTP request.
    :param max_retries: Maximum total number of retries.
    :param backoff_factor: a time factor for exponential backoff. e.g. value 5 means the HTTP
      request will be retried with interval 5, 10, 20... seconds. A value of 0 turns off the
      exponential backoff.
    :param retry_codes: a list of HTTP response error codes that qualifies for retry.
    :param kwargs: Additional keyword arguments to pass to `requests.Session.request()`

    :return: requests.Response object.
    """
    session = _get_request_session(max_retries, backoff_factor, retry_codes)
    return session.request(method, url, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--range-start", required=True, type=int)
    parser.add_argument("--range-end", required=True, type=int)
    parser.add_argument("--headers", required=True, type=str)
    parser.add_argument("--download-path", required=True, type=str)
    parser.add_argument("--http-uri", required=True, type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        download_chunk(
            range_start=args.range_start,
            range_end=args.range_end,
            headers=json.loads(args.headers),
            download_path=args.download_path,
            http_uri=args.http_uri,
        )
    except requests.HTTPError as e:
        print(  # pylint: disable=print-function
            json.dumps(
                {
                    "error_status_code": e.response.status_code,
                    "error_text": str(e),
                }
            ),
            file=sys.stdout,
        )


if __name__ == "__main__":
    main()
