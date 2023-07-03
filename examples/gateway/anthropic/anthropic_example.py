import json
import requests
from mlflow.gateway import query, set_gateway_uri, MlflowGatewayClient


def main():
    gateway_uri = "http://localhost:5000"

    # Using the fluent API
    set_gateway_uri(gateway_uri)

    # Completions request using the fluent API
    response_fluent = query(
        route="completions-claude",
        data={
            "prompt": "How many average size European ferrets can fit inside a standard olympic "
            "size swimming pool?",
            "max_tokens": 5000,
        },
    )
    print(f"Fluent API response: {response_fluent}")

    # Using the client API
    gateway_client = MlflowGatewayClient(gateway_uri)

    # Completions request using the client API
    response_client = gateway_client.query(
        route="completions-claude",
        data={
            "prompt": "What would happen if I hooked up my guitar to a 500kW speaker?",
            "temperature": 0.9,
            "max_tokens": 10000,
        },
    )

    print(f"Client API response: {response_client}")

    # REST request using requests library
    url = "http://127.0.0.1:5000/gateway/completions-claude/invocations"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": "What would happen if a black hole swallowed another black hole?",
    }

    response_rest = requests.post(url, headers=headers, json=data)
    print(f"REST API response: {response_rest.json()}")

    # CURL command for making a REST request from the terminal
    # This command is equivalent to the REST request above
    print(
        f"""
    You can also use the following curl command to make a request from your terminal:

    curl -X POST {url} 
        -H "Content-Type: application/json" 
        -d '{json.dumps({
            "prompt": "Could AI ever develop consciousness?", 
            "max_tokens": 200, 
            "temperature": 0.88
        })}'
    """
    )


if __name__ == "__main__":
    main()
