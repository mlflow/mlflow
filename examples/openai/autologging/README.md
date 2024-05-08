# OpenAI Autologging Examples

## Using OpenAI client

The recommended way of using `openai` is to instantiate a client 
using `openai.OpenAI()`. You can run the following example to use
autologging using such client.

```shell
python3 examples/openai/autologging/instantiate_client.py --api-key="your-api-key"
```

## Using module-level client

`openai` exposes a global client instance that can be used to make requests.
You can run the following example to use autologging with the global client.

```shell
export OPENAI_API_KEY="your-api-key"
python3 examples/openai/autologging/global_client.py
```
