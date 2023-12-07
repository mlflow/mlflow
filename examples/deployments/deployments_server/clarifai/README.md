## Example endpoint configuration for Clarifai

\
[Clarifai](https://clarifai.com/) enables easy and efficient deployment of cutting-edge large language models (LLMs) for text processing. Developers can leverage Clarifai's platform to serve up predictions from state-of-the-art open source LLMs including options like [Llama2](https://clarifai.com/meta/Llama-2/models/llama2-7b-chat), [Falcon](https://clarifai.com/tiiuae/falcon/models/falcon-40b-instruct), [Mistral](https://clarifai.com/mistralai/completion/models/mistral-7B-Instruct), [WizardLM](https://clarifai.com/explore/models?searchQuery=wizard&page=1&perPage=24), [Zephyr](https://clarifai.com/huggingface-research/zephyr/models/zephyr-7B-alpha) and many more for text generation.

Explore more Clarifai hosted LLMs [here](https://clarifai.com/explore/models).

### Getting Started

#### Setting a Clarifai PAT

This example requires a [Clarifai PAT](https://docs.clarifai.com/clarifai-basics/authentication/personal-access-tokens/):

```sh
export CLARIFAI_PAT=...
```

### Update the config.yaml

To configure access to Clarifai LLMs, update the MLflow Desployments server `config.yaml` file. The example below shows how to specify routes for both text completions and text embeddings using Clarifai models:

```yaml
endpoints:
  - name: completions
    endpoint_type: llm/v1/completions
    model:
      provider: clarifai
      name: mistral-7B-Instruct
      config:
        CLARIFAI_PAT: $CLARIFAI_PAT
        user_id: mistralai
        app_id: completion

  - name: embeddings
    endpoint_type: llm/v1/embeddings
    model:
      provider: clarifai
      name: multimodal-clip-embed
      config:
        CLARIFAI_PAT: $CLARIFAI_PAT
        user_id: clarifai
        app_id: main
```

This configuration file specifies two routes: 'completions' and 'embeddings', both using models 'mistral-7B-Instruct' and 'multimodal-clip-embed' hosted in Clarifai Platform, respectively.

### Starting the MLflow Deployments server

After the configuration file is created, you can start the gateway by running the following command:

```
mlflow deployments start-server --config-path examples/deployments/deployments_server/clarifai/config.yaml --port 7000
```

### Querying the MLflow Deployments server

See the [example script](example.py) within this directory to see how to query the `mistral-7B-Instruct` and `multimodal-clip-embed` model that is served.
