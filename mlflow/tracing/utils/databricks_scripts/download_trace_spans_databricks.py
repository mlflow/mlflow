import json
import os
import sys
from functools import partial
from multiprocessing import Pool

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
    url_path = f"/api/3.0/mlflow/traces/{trace_id}/download"
    url = f"{databricks_host.lstrip('/')}{url_path}"

    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=10, backoff_factor=0.25, status_forcelist=TRANSIENT_FAILURE_RESPONSE_CODES
            )
        ),
    )

    credentials_response = session.get(
        url, headers=databricks_auth_headers, timeout=REQUEST_TIMEOUT
    )

    if credentials_response.status_code == 200:
        credential_info = credentials_response.json()["credential_info"]
        data_response = session.get(
            credential_info["signed_uri"],
            headers=credential_info.get("headers"),
            timeout=REQUEST_TIMEOUT,
        )
        if data_response.status_code == 200:
            return {"trace_id": trace_id, "trace_data": data_response.json()}

    # TODO: Handle failure cases after retries


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
    with Pool(processes=64) as pool:
        trace_ids_and_data = pool.map(download_trace_partial, trace_ids)

    # Transform the list of dictionaries into a dictionary with trace_id as key
    trace_ids_and_data = {
        trace_data["trace_id"]: trace_data["trace_data"]
        for trace_data in trace_ids_and_data
        if trace_data is not None
    }

    with open(os.path.join(dst_dir, "trace_ids_and_data.json"), "w") as f:
        f.write(json.dumps(trace_ids_and_data))
