"""
# How to run
mlflow deployments start-server \
    --config-path examples/deployments/deployments_server/openai/config.yaml --port 7000
python test.py
"""

import requests

req = {
    "messages": [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco, Tokyo, and Paris?",
        }
    ],
}

# Without tools
print("--- Without tools ---")
print(requests.post("http://127.0.0.1:7000/endpoints/chat/invocations", json=req).json())

# With tools
print("--- With tools ---")
print(
    requests.post(
        "http://127.0.0.1:7000/endpoints/chat/invocations",
        json={
            **req,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                    },
                }
            ],
        },
    ).json(),
)
