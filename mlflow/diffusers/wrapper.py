import io
import logging
from types import MappingProxyType
from typing import Any

import pandas as pd

from mlflow.diffusers import _detect_device
from mlflow.exceptions import MlflowException

_logger = logging.getLogger(__name__)


class _DiffusersAdapterWrapper:
    def __init__(self, adapter_path, flavor_conf, model_config=None):
        self._adapter_path = adapter_path
        self._flavor_conf = flavor_conf
        self._model_config = MappingProxyType(model_config or {})
        self._pipeline = None

    def _load_pipeline(self):
        from diffusers import DiffusionPipeline

        base_model_id = self._flavor_conf["base_model_id"]
        adapter_type = self._flavor_conf.get("adapter_type", "lora")
        device = _detect_device(self._model_config.get("device", None))
        torch_dtype = self._model_config.get("torch_dtype", "auto")

        _logger.info("Loading base pipeline: %s", base_model_id)
        pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch_dtype)

        _logger.info("Loading LoRA adapter from: %s", self._adapter_path)
        if adapter_type == "lora":
            pipe.load_lora_weights(self._adapter_path)
        else:
            raise MlflowException(
                f"Loading adapter type '{adapter_type}' is not yet supported. "
                f"Currently only 'lora' adapters can be loaded at inference time."
            )

        self._pipeline = pipe.to(device)

    def get_raw_model(self):
        if self._pipeline is None:
            self._load_pipeline()
        return self._pipeline

    def predict(self, data, params: dict[str, Any] | None = None):
        if self._pipeline is None:
            self._load_pipeline()

        # Extract prompts
        if isinstance(data, pd.DataFrame):
            if "prompt" not in data.columns:
                raise ValueError(
                    f"Input DataFrame must contain a 'prompt' column. "
                    f"Got columns: {list(data.columns)}"
                )
            prompts = data["prompt"].tolist()
        elif isinstance(data, str):
            prompts = [data]
        elif isinstance(data, dict):
            if "prompt" not in data:
                raise ValueError(
                    f"Input dict must contain a 'prompt' key. Got keys: {list(data.keys())}"
                )
            prompts = data["prompt"]
            if isinstance(prompts, str):
                prompts = [prompts]
        elif isinstance(data, list):
            prompts = data
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")

        if not prompts:
            raise ValueError("No prompts provided. Input must contain at least one prompt string.")

        # Extract generation params
        params = params or {}
        gen_kwargs = {
            "num_inference_steps": params.get("num_inference_steps", 30),
            "guidance_scale": params.get("guidance_scale", 7.5),
            "height": params.get("height", 512),
            "width": params.get("width", 512),
        }

        output = self._pipeline(prompt=prompts, **gen_kwargs)

        if not hasattr(output, "images") or output.images is None:
            raise MlflowException(
                "Pipeline returned no images. The output may have been filtered "
                "by the safety checker, or the pipeline does not support image generation."
            )

        results = []
        for image in output.images:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            results.append(buf.getvalue())

        return results
