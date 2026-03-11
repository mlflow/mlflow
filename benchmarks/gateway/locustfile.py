"""
Locust load test definition for the MLflow AI Gateway.

Hits chat (weight 8), completions (weight 1), and embeddings (weight 1) endpoints.
Captures X-MLflow-Gateway-Overhead-Ms header as a custom Locust metric.
"""

import locust

CHAT_PAYLOAD = {
    "messages": [
        {"role": "user", "content": "Hello, this is a benchmark request."},
    ],
    "temperature": 0.0,
    "max_tokens": 50,
}

COMPLETIONS_PAYLOAD = {
    "prompt": "Hello, this is a benchmark request.",
    "temperature": 0.0,
    "max_tokens": 50,
}

EMBEDDINGS_PAYLOAD = {
    "input": ["Hello, this is a benchmark request."],
}

HEADERS = {"Content-Type": "application/json"}


class GatewayUser(locust.HttpUser):
    wait_time = locust.between(0, 0)

    @locust.tag("chat")
    @locust.task(8)
    def chat(self):
        with self.client.post(
            "/gateway/benchmark-chat/invocations",
            json=CHAT_PAYLOAD,
            headers=HEADERS,
            catch_response=True,
        ) as resp:
            _check_response(resp, "chat")

    @locust.tag("completions")
    @locust.task(1)
    def completions(self):
        with self.client.post(
            "/gateway/benchmark-completions/invocations",
            json=COMPLETIONS_PAYLOAD,
            headers=HEADERS,
            catch_response=True,
        ) as resp:
            _check_response(resp, "completions")

    @locust.tag("embeddings")
    @locust.task(1)
    def embeddings(self):
        with self.client.post(
            "/gateway/benchmark-embeddings/invocations",
            json=EMBEDDINGS_PAYLOAD,
            headers=HEADERS,
            catch_response=True,
        ) as resp:
            _check_response(resp, "embeddings")


def _check_response(resp, endpoint_name):
    if resp.status_code != 200:
        resp.failure(f"{endpoint_name} returned {resp.status_code}: {resp.text[:200]}")
        return

    if overhead := resp.headers.get("X-MLflow-Gateway-Overhead-Ms"):
        locust.events.request.fire(
            request_type="OVERHEAD",
            name=f"{endpoint_name}_overhead",
            response_time=float(overhead),
            response_length=0,
            exception=None,
            context={},
        )
