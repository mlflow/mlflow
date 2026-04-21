"""
Demo: MLflow Diffusers Adapter Flavor (LoRA)

This script demonstrates the full workflow of logging and loading a diffusion
model LoRA adapter using the native mlflow.diffusers flavor.

No GPU or real model weights required — uses a fake adapter for validation.
"""

import tempfile
from pathlib import Path

import numpy as np
import yaml
from safetensors.numpy import save_file

import mlflow
import mlflow.diffusers


def create_fake_lora_adapter(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simulate LoRA weight matrices (small random tensors)
    tensors = {
        "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_down.weight": (
            np.random.randn(4, 320).astype(np.float32)
        ),
        "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_up.weight": (
            np.random.randn(320, 4).astype(np.float32)
        ),
    }
    adapter_file = output_dir / "pytorch_lora_weights.safetensors"
    save_file(tensors, str(adapter_file))

    print(f"Created fake LoRA adapter at: {adapter_file}")
    print(f"  Adapter size: {adapter_file.stat().st_size} bytes")
    return output_dir


def demo_log_and_load():
    """Demonstrate the full log -> load -> inspect cycle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create fake adapter
        adapter_dir = create_fake_lora_adapter(Path(tmpdir) / "my_lora")

        # 2. Log the adapter with MLflow
        print("\n--- Logging adapter with mlflow.diffusers.log_model() ---")
        mlflow.set_experiment("diffusers-adapter-poc")

        with mlflow.start_run(run_name="lora-adapter-demo") as run:
            model_info = mlflow.diffusers.log_model(
                adapter_path=str(adapter_dir),
                base_model="black-forest-labs/FLUX.1-dev",
                adapter_type="lora",
                name="lora_model",
                metadata={
                    "lora_rank": 4,
                    "training_steps": 1000,
                    "trigger_word": "sks style",
                },
            )

            print(f"  Run ID: {run.info.run_id}")
            print(f"  Model URI: {model_info.model_uri}")

        # 3. Inspect the MLmodel file
        print("\n--- MLmodel file contents ---")
        model_uri = f"runs:/{run.info.run_id}/lora_model"
        local_path = mlflow.artifacts.download_artifacts(model_uri)
        mlmodel_path = Path(local_path) / "MLmodel"

        with open(mlmodel_path) as f:
            mlmodel = yaml.safe_load(f)

        print(yaml.dump(mlmodel, default_flow_style=False, indent=2))

        # 4. Load the model back
        print("--- Loading model back with mlflow.diffusers.load_model() ---")
        loaded = mlflow.diffusers.load_model(model_uri)

        print(f"  Type: {type(loaded).__name__}")
        print(f"  Base model: {loaded.base_model}")
        print(f"  Adapter type: {loaded.adapter_type}")
        print(f"  Adapter path: {loaded.adapter_path}")
        print(f"  Adapter files: {list(Path(loaded.adapter_path).iterdir())}")

        # 5. Verify flavor config from MLmodel
        print("\n--- Flavor config ---")
        flavor_conf = mlmodel["flavors"]["diffusers"]
        print(f"  base_model: {flavor_conf['base_model']}")
        print(f"  adapter_type: {flavor_conf['adapter_type']}")
        print(f"  adapter_weights: {flavor_conf['adapter_weights']}")

        # 6. Show that pyfunc interface is available
        print("\n--- Pyfunc model interface ---")
        print("  mlflow.pyfunc.load_model() would return a wrapper with predict()")
        print("  predict() accepts: DataFrame/dict with 'prompt' column")
        print("  predict() returns: list of PNG-encoded image bytes")
        print("  (Skipping actual pyfunc load — requires base model download)")

        print("\n--- Demo complete! ---")
        print(
            "The adapter is logged as a first-class MLflow model with full model registry support."
        )
        print(
            "To generate images, call loaded.load_pipeline() on a machine "
            "with the base model available."
        )


if __name__ == "__main__":
    demo_log_and_load()
