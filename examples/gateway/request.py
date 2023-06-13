import requests


def main():
    PREFIX = "http://127.0.0.1:5000/gateway/routes"

    print("Completions")
    with requests.post(
        f"{PREFIX}/completions",
        json={"prompt": "Hello, my name is"},
    ) as resp:
        resp.raise_for_status()
        print(resp.json())

    print("Chat")
    with requests.post(
        f"{PREFIX}/chat",
        json={"messages": [{"role": "user", "content": "Tell me a joke"}]},
    ) as resp:
        resp.raise_for_status()
        print(resp.json())

    print("Embeddings")
    with requests.post(
        f"{PREFIX}/embeddings",
        json={"text": "hello world"},
    ) as resp:
        resp.raise_for_status()
        print(resp.json())


if __name__ == "__main__":
    main()
