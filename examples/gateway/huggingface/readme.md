## Example route configuration for Huggingface Text Generation Inference

To see an example of specifying the completion routes for Huggingface Text Generation Inference servers, see [the configuration](config.yaml) YAML file.

This configuration file specifies the 'completions' route, using the 'falcon-7b-instruct' model.

## Setting a HF Text Generation Inference server

This example requires one [HF Text Generation Inference server](https://huggingface.co/docs/text-generation-inference/index) running on port 8080 on your local machine. The easiest way of getting started is by using the official Docker container. You can use this example, retrieved from the official [documentation](https://huggingface.co/docs/text-generation-inference/quicktour):

```
model=tiiuae/falcon-7b-instruct
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.1.1 --model-id $model
```
