"""
This script should be executed in a fresh python interpreter process using `subprocess`.
"""
import argparse
import json
import requests
import sys
import time



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--range-start", required=True, type=int)
    parser.add_argument("--range-end", required=True, type=int)
    parser.add_argument("--headers", required=True, type=str)
    parser.add_argument("--download-path", required=True, type=str)
    parser.add_argument("--http-uri", required=True, type=str)
    return parser.parse_args()


def main():
    before = time.time()
    args = parse_args()
    after = time.time()
    print("ARGPARSE TIME", (after - before))

    before = time.time()
    from mlflow.utils.file_utils import download_chunk
    after = time.time()
    print("MLFLOW IMPORT TIME", (after - before))
    try:
        download_chunk(
            range_start=args.range_start,
            range_end=args.range_end,
            headers=json.loads(args.headers),
            download_path=args.download_path,
            http_uri=args.http_uri,
        )
    except requests.HTTPError as e:
        print(
            json.dumps(
                {  # pylint: disable=print-function
                    "error_status_code": e.response.status_code,
                    "error_text": str(e),
                }
            ),
            file=sys.stdout,
        )


if __name__ == "__main__":
    before = time.time()
    main()
    after = time.time()
    print("MAIN TIME", after - before, file=sys.stdout)
