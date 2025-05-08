## Example endpoint configuration for Huggingface Text Generation Inference

[Huggingface Text Generation Inference (TGI)](https://huggingface.co/docs/text-generation-inference/index) is a comprehensive toolkit designed for deploying and serving Large Language Models (LLMs) efficiently. It offers optimized support for various popular open-source LLMs such as Llama, Falcon, StarCoder, BLOOM, and GPT-Neo. TGI comes with various built-in optimizations and features, such as:

- Simple launcher to serve most popular LLMs
- Tensor Parallelism for faster inference on multiple GPUs
- Safetensors weight loading
- Optimized transformers code for inference using Flash Attention and Paged Attention on the most popular architectures

It should be noted that only a [selection of models](https://huggingface.co/docs/text-generation-inference/supported_models) are optimized for TGI, which uses custom CUDA kernels for faster inference. You can add the flag `--disable-custom-kernels`` at the end of the docker run command if you wish to disable them. If the above list lacks the model you would like to serve, or in the case you created a custom created model, you can try to initialize and serve the model anyways. However, since the model is not optimized for TGI, performance is not guaranteed.

For a more detailed description of all features, please go to the [documentation](https://huggingface.co/docs/text-generation-inference/index).

## Getting Started

> **NOTE** This example is tested on a Linux Machine (Debian 11) with a NVIDIA A100 GPU.

To configure the MLflow AI Gateway with Huggingface Text Generation Inference, a few additional steps need to be followed. The initial step involves deploying a Huggingface model on the TGI server, which is illustrated in the next section.

The recommended approach for deploying the TGI server is by utilizing the [official Docker container](ghcr.io/huggingface/text-generation-inference:1.1.1). Docker is an open-source platform that provides a streamlined solution for automating the deployment, scaling, and management of applications through containers. These containers encompass all the essential dependencies required for seamless execution, including libraries, binaries, and configuration files. To install Docker, please refer to the [installation guide](https://docs.docker.com/get-docker/).

Before proceeding, it is important to verify that your machine has the appropriate hardware to initiate the server. TGI optimized models are compatible with NVIDIA A100, A10G, and T4 GPUs. While other GPU hardware may still provide performance advantages, certain operations such as flash attention and paged attention will not be executed. If you intend to run the container on a machine lacking GPUs or CUDA support, you can eliminate the `--gpus all` flag and include `--disable-custom-kernels`. However, please note that the CPU is not the intended platform for the server, and this choice significantly impacts performance.

#### Installing the NVIDIA Container Toolkit

To begin, the installation of the NVIDIA container toolkit is necessary. This toolkit is essential for running GPU-accelerated containers. Execute the following command to acquire all the requisite packages [ref the code]:

```sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && \
    sudo apt-get update
```

Install the NVIDIA Container toolkit by running the following command.

```
sudo apt-get install -y nvidia-container-toolkit
```

#### Running the TGI server.

After you installed the NVIDIA Container toolkit, you can run the following Docker command to to start a TGI server on your local machine on port `8000`. This will load a [falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) model on the TGI server.

```
model=tiiuae/falcon-7b-instruct
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
docker run --gpus all --shm-size 1g -p 8000:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.1.1 --model-id $model
```

After the TGI server is deployed, run the following script to verify that it is working correctly:

```
import requests
headers = {
    "Content-Type": "application/json",
}
data = {
    'inputs': 'What is Deep Learning?',
    'parameters': {
        'max_new_tokens': 20,
    },
}
response = requests.post('http://127.0.0.1:8000/generate', headers=headers, json=data)
print(response.json())
# {'generated_text': '\nDeep learning is a branch of machine learning that uses artificial neural networks to learn and make decisions.'}
```

## Update the config.yaml to add a new embeddings endpoint

After you started the server, update the MLflow AI Gateway configuration file [config.yaml](config.yaml) and add the server as a new endpoint:

```
endpoints:
  - name: completions
    endpoint_type: llm/v1/completions
    model:
      provider: "huggingface-text-generation-inference"
      name: llm
      config:
        hf_server_url: http://127.0.0.1:8000/generate
```

## Starting the MLflow AI Gateway

After the configuration file is created, you can start the MLflow AI Gateway by running the following command:

```
mlflow gateway start --config-path examples/gateway/huggingface/config.yaml --port 7000
```

## Querying the endpoint

See the [example script](example.py) within this directory to see how to query the `falcon-7b-instruct` model that is served.

## Setting the parameters of TGI

When you make a request to the MLflow Depoyments server, the information you provide in the request body will be sent to TGI. This gives you more control over the output you receive from TGI. However, it's important to note that you cannot turn off `details` and `decoder_input_details`, as they are necessary for TGI endpoints to work correctly.
