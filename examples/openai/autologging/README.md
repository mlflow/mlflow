# OpenAI Autologging Examples

## Using OpenAI client

The recommended way of using `openai` is to instantiate a client
using `openai.OpenAI()`. You can run the following example to use
autologging using such client.

Before running these examples, ensure that you have the following additional libraries installed:

```shell
pip install tenacity tiktoken 'openai>=1.17'
```

You can run the example via your command prompt as follows:

```shell
python examples/openai/autologging/instantiated_client.py --api-key="your-api-key"
```

## Using module-level client

`openai` exposes a module client instance that can be used to make requests.
You can run the following example to use autologging with the module client.

```shell
export OPENAI_API_KEY="your-api-key"
python examples/openai/autologging/module_client.py
```
