import io
import logging
import threading
from types import MappingProxyType
from typing import Any

import pandas as pd

from mlflow.diffusers import _detect_device
from mlflow.exceptions import MlflowException

_logger = logging.getLogger(__name__)


class _DiffusersAdapterWrapper:
    def __init__(
        self,
        adapter_path: str,
        flavor_conf: dict[str, Any],
        model_config: dict[str, Any] | None = None,
    ):
        self._adapter_path = adapter_path
        self._flavor_conf = flavor_conf
        self._model_config = MappingProxyType(model_config or {})
        self._pipeline = None
        self._load_lock = threading.Lock()

    def _load_pipeline(self):
        from diffusers import DiffusionPipeline

        base_model = self._model_config.get("base_model") or self._flavor_conf["base_model"]
        base_model_revision = self._flavor_conf.get("base_model_revision")
        device = _detect_device(self._model_config.get("device"))
        torch_dtype = self._model_config.get("torch_dtype", "auto")

        load_kwargs = {"torch_dtype": torch_dtype}
        if base_model_revision:
            load_kwargs["revision"] = base_model_revision

        weight_name = self._flavor_conf.get("weight_name")
        lora_kwargs = {}
        if weight_name:
            lora_kwargs["weight_name"] = weight_name

        _logger.info("Loading base pipeline: %s", base_model)
        try:
            pipe = DiffusionPipeline.from_pretrained(base_model, **load_kwargs)
        except OSError as e:
            raise MlflowException(
                f"Failed to load base model '{base_model}'. If the model has moved, "
                "pass the correct location via "
                "model_config={{'base_model': '<new_path_or_hub_id>'}} "
                "when loading with mlflow.pyfunc.load_model()."
            ) from e

        _logger.info("Loading LoRA adapter from: %s", self._adapter_path)
        pipe.load_lora_weights(self._adapter_path, **lora_kwargs)

        self._pipeline = pipe.to(device)

    def get_raw_model(self):
        if self._pipeline is None:
            with self._load_lock:
                if self._pipeline is None:
                    self._load_pipeline()
        return self._pipeline

    def _flatten_prompts(self, prompts):
        """Flatten nested lists produced by schema enforcement."""
        flat = []
        for item in prompts:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat

    def predict(self, data, params: dict[str, Any] | None = None):
        pipeline = self.get_raw_model()

        if isinstance(data, pd.DataFrame):
            if "prompt" in data.columns:
                prompts = data["prompt"].tolist()
            elif len(data.columns) == 1:
                # Schema enforcement wraps scalar strings into a single-column DataFrame
                prompts = data.iloc[:, 0].tolist()
            else:
                raise MlflowException(
                    f"Input DataFrame must contain a 'prompt' column. "
                    f"Got columns: {list(data.columns)}"
                )
            # Schema enforcement may wrap {"prompt": ["a","b"]} into a
            # single-row DataFrame where the cell contains a list, producing
            # [["a","b"]] after tolist(). Flatten to ["a","b"].
            prompts = self._flatten_prompts(prompts)
        elif isinstance(data, str):
            prompts = [data]
        elif isinstance(data, dict):
            if "prompt" not in data:
                raise MlflowException(
                    f"Input dict must contain a 'prompt' key. Got keys: {list(data.keys())}"
                )
            prompts = data["prompt"]
            if isinstance(prompts, str):
                prompts = [prompts]
            elif isinstance(prompts, list):
                prompts = self._flatten_prompts(prompts)
            else:
                raise MlflowException(
                    "'prompt' value must be a string or list of strings, "
                    f"got {type(prompts).__name__}."
                )
        elif isinstance(data, list):
            prompts = self._flatten_prompts(data)
        else:
            raise MlflowException(f"Unsupported input type: {type(data)}")

        if not prompts:
            raise MlflowException(
                "No prompts provided. Input must contain at least one prompt string."
            )

        if any(p is None for p in prompts):
            raise MlflowException(
                "Prompt values must be strings, not None. "
                "Check your input for missing or null values."
            )

        params = params or {}
        param_keys = ("num_inference_steps", "guidance_scale", "height", "width", "negative_prompt")
        gen_kwargs = {k: params[k] for k in param_keys if k in params}
        # Drop empty-string negative_prompt so the pipeline uses its own default
        if gen_kwargs.get("negative_prompt") == "":
            del gen_kwargs["negative_prompt"]

        output = pipeline(prompt=prompts, **gen_kwargs)

        if not hasattr(output, "images") or not output.images:
            raise MlflowException(
                "Pipeline returned no images. The output may have been filtered "
                "by the safety checker, or the pipeline does not support image generation."
            )

        results = []
        for image in output.images:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            results.append(buf.getvalue())
            buf.close()

        return results
