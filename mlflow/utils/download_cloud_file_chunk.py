"""
This script should be executed in a fresh python interpreter process using `subprocess`.
"""
import argparse
import importlib.util
import json
import os
import sys

from requests.exceptions import ChunkedEncodingError, ConnectionError, HTTPError


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--range-start", required=True, type=int)
    parser.add_argument("--range-end", required=True, type=int)
    parser.add_argument("--headers", required=True, type=str)
    parser.add_argument("--download-path", required=True, type=str)
    parser.add_argument("--http-uri", required=True, type=str)
    parser.add_argument("--temp-file", required=True, type=str)
    return parser.parse_args()


def main():
    file_path = os.path.join(os.path.dirname(__file__), "request_utils.py")
    module_name = "mlflow.utils.request_utils"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    download_chunk = module.download_chunk

    args = parse_args()

    try:
        download_chunk(
            range_start=args.range_start,
            range_end=args.range_end,
            headers=json.loads(args.headers),
            download_path=args.download_path,
            http_uri=args.http_uri,
        )
    except (ConnectionError, ChunkedEncodingError):
        with open(args.temp_file, "w") as f:
            json.dump({"retryable": True}, f)
        raise
    except HTTPError as e:
        with open(args.temp_file, "w") as f:
            json.dump(
                {
                    "retryable": e.response.status_code in (401, 403, 408),
                    "status_code": e.response.status_code,
                },
                f,
            )
        raise


if __name__ == "__main__":
    main()
