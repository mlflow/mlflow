import json
import requests
from mlflow.gateway import query, set_gateway_uri, MlflowGatewayClient


def main():
    gateway_uri = "http://localhost:5000"

    # Using the fluent API
    set_gateway_uri(gateway_uri)

    # Completions request using the fluent API
    response_fluent = query(
        route="completions",
        data={
            "prompt": "How many patties could be stacked on a cheeseburger before issues arise?",
            "max_tokens": 200,
            "temperature": 0.25,
        },
    )
    print(f"Fluent API response: {response_fluent}")

    # Using the client API
    gateway_client = MlflowGatewayClient(gateway_uri)

    # Completions request using the client API
    response_client = gateway_client.query(
        route="completions",
        data={
            "prompt": "What would happen if we stocked Europa with trout?",
            "temperature": 0.9,
            "max_tokens": 200,
        },
    )

    print(f"Client API response: {response_client}")

    # REST request using requests library
    url = "http://127.0.0.1:5000/gateway/completions/invocations"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": "Is it possible to vine swing like Tarzan or is that a myth? Asking for a friend.",
        "max_tokens": 200,
        "temperature": 0.95,
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
            "prompt": "How could long-term durable memory be introduced into an attention model?", 
            "max_tokens": 200, 
            "temperature": 0.88
        })}'
    """
    )


if __name__ == "__main__":
    main()
