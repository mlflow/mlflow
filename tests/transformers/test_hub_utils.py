import pytest

from mlflow.transformers.flavor_config import FlavorKey


@pytest.mark.parametrize(
    ("repo_id", "framework", "expected_files"),
    [
        (
            "distilgpt2",
            "pt",
            ["model.safetensors", "config.json"],
        ),
        (
            "hf-internal-testing/tiny-random-bert-sharded-safetensors",
            "pt",
            [
                "config.json",
                "model.safetensors.index.json",
                "model-00001-of-00005.safetensors",
                "model-00002-of-00005.safetensors",
                "model-00003-of-00005.safetensors",
                "model-00004-of-00005.safetensors",
                "model-00005-of-00005.safetensors",
            ],
        ),
        (
            "google/tapas-small-finetuned-wtq",
            "pt",
            ["pytorch_model.bin", "config.json"],
        ),
        (
            "google/tapas-small-finetuned-wtq",
            "tf",
            ["tf_model.h5", "config.json"],
        ),
    ],
)
def test_download_model_weights_from_hub(tmp_path, repo_id, framework, expected_files):
    from mlflow.transformers.hub_utils import download_model_weights_from_hub

    flavor_conf = {
        FlavorKey.MODEL_NAME: repo_id,
        FlavorKey.FRAMEWORK: framework,
        FlavorKey.MODEL_REVISION: None,
    }
    download_model_weights_from_hub(flavor_conf, tmp_path)

    for file in expected_files:
        assert (tmp_path / file).exists()
