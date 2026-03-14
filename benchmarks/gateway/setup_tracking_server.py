"""
Set up a gateway endpoint in a running MLflow tracking server for benchmarking.

Creates: secret -> model definition -> endpoint (3 API calls).
The endpoint routes to a fake OpenAI server for controlled latency measurement.

Usage:
    python setup_tracking_server.py --tracking-uri http://127.0.0.1:5000 --fake-server-url http://127.0.0.1:9000/v1
"""

import argparse
import json
import sys
import urllib.request

ENDPOINT_NAME = "benchmark-chat"
SECRET_NAME = "benchmark-secret"
MODEL_DEF_NAME = "benchmark-model"


def api_call(base_url, path, body):
    url = f"{base_url}/api/3.0/mlflow/{path}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"ERROR {e.code} calling {url}: {error_body}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Set up gateway endpoint for benchmarking")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--fake-server-url", default="http://127.0.0.1:9000/v1")
    parser.add_argument("--endpoint-name", default=ENDPOINT_NAME)
    parser.add_argument(
        "--no-usage-tracking",
        action="store_true",
        help="Disable usage tracking (tracing) on the endpoint",
    )
    args = parser.parse_args()

    base = args.tracking_uri.rstrip("/")

    # 1. Create secret
    print(f"Creating secret '{SECRET_NAME}'...")
    secret_resp = api_call(
        base,
        "gateway/secrets/create",
        {
            "secret_name": SECRET_NAME,
            "secret_value": {"api_key": "fake-benchmark-key"},
            "provider": "openai",
            "auth_config": {"api_base": args.fake_server_url},
        },
    )
    secret_id = secret_resp["secret"]["secret_id"]
    print(f"  secret_id: {secret_id}")

    # 2. Create model definition
    print(f"Creating model definition '{MODEL_DEF_NAME}'...")
    model_resp = api_call(
        base,
        "gateway/model-definitions/create",
        {
            "name": MODEL_DEF_NAME,
            "secret_id": secret_id,
            "provider": "openai",
            "model_name": "gpt-4o-mini",
        },
    )
    model_def_id = model_resp["model_definition"]["model_definition_id"]
    print(f"  model_definition_id: {model_def_id}")

    # 3. Create endpoint
    usage_tracking = not args.no_usage_tracking
    print(f"Creating endpoint '{args.endpoint_name}' (usage_tracking={usage_tracking})...")
    endpoint_body = {
        "name": args.endpoint_name,
        "model_configs": [
            {
                "model_definition_id": model_def_id,
                "linkage_type": "PRIMARY",
                "weight": 1.0,
            }
        ],
        "usage_tracking": usage_tracking,
    }
    endpoint_resp = api_call(base, "gateway/endpoints/create", endpoint_body)
    endpoint_id = endpoint_resp["endpoint"]["endpoint_id"]
    print(f"  endpoint_id: {endpoint_id}")

    invoke_url = f"{base}/gateway/{args.endpoint_name}/mlflow/invocations"
    print(f"\nEndpoint ready. Invoke URL: {invoke_url}")


if __name__ == "__main__":
    main()
