import json
import logging
import os
import sys
from functools import partial
from multiprocessing import Pool

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRANSIENT_FAILURE_RESPONSE_CODES = [
    408,  # Request Timeout
    429,  # Too Many Requests
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
]
REQUEST_TIMEOUT = 120


def download_trace(trace_id, databricks_host, databricks_auth_headers):
    url_path = f"/api/3.0/mlflow/traces/{trace_id}/credentials-for-data-download"
    url = f"{databricks_host.rstrip('/')}{url_path}"

    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=8, backoff_factor=0.25, status_forcelist=TRANSIENT_FAILURE_RESPONSE_CODES
            )
        ),
    )

    try:
        credentials_response = session.get(
            url, headers=databricks_auth_headers, timeout=REQUEST_TIMEOUT
        )
        credentials_response.raise_for_status()  # Raise an error for bad responses
        credential_info = credentials_response.json()["credential_info"]
        data_response = session.get(
            credential_info["signed_uri"],
            headers=credential_info.get("headers"),
            timeout=REQUEST_TIMEOUT,
        )
        data_response.raise_for_status()  # Raise an error for bad responses
        return {"trace_id": trace_id, "trace_data": data_response.json()}
    except Exception as e:
        if "429" in str(e):
            return download_trace(trace_id, databricks_host, databricks_auth_headers)
        logger.warning(f"Exception while downloading trace {trace_id}: {e}")


if __name__ == "__main__":
    databricks_host = sys.argv[1]
    databricks_auth_headers = json.loads(sys.argv[2])
    dst_dir = sys.argv[3]

    with open(os.path.join(dst_dir, "trace_ids.json")) as f:
        trace_ids = json.load(f)

    download_trace_partial = partial(
        download_trace,
        databricks_host=databricks_host,
        databricks_auth_headers=databricks_auth_headers,
    )

    with Pool(processes=min(64, os.cpu_count() * 4)) as pool:
        trace_ids_and_data = pool.map(download_trace_partial, trace_ids)

    trace_ids_and_data = {
        td["trace_id"]: td["trace_data"] for td in trace_ids_and_data if td is not None
    }

    with open(os.path.join(dst_dir, "trace_ids_and_data.json"), "w") as f:
        json.dump(trace_ids_and_data, f)
