import json
import os
import posixpath

import pytest

import mlflow
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.time import get_current_time_millis


@pytest.mark.parametrize("subdir", [None, ".", "dir", "dir1/dir2", "dir/.."])
def test_log_image_numpy(subdir):
    import numpy as np
    from PIL import Image

    filename = "image.png"
    artifact_file = filename if subdir is None else posixpath.join(subdir, filename)

    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    with mlflow.start_run():
        mlflow.log_image(image, artifact_file)

        artifact_path = None if subdir is None else posixpath.normpath(subdir)
        artifact_uri = mlflow.get_artifact_uri(artifact_path)
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        assert os.listdir(run_artifact_dir) == [filename]

        logged_path = os.path.join(run_artifact_dir, filename)
        loaded_image = np.asarray(Image.open(logged_path), dtype=np.uint8)
        np.testing.assert_array_equal(loaded_image, image)


@pytest.mark.parametrize("subdir", [None, ".", "dir", "dir1/dir2", "dir/.."])
def test_log_image_pillow(subdir):
    from PIL import Image, ImageChops

    filename = "image.png"
    artifact_file = filename if subdir is None else posixpath.join(subdir, filename)

    image = Image.new("RGB", (100, 100))

    with mlflow.start_run():
        mlflow.log_image(image, artifact_file)

        artifact_path = None if subdir is None else posixpath.normpath(subdir)
        artifact_uri = mlflow.get_artifact_uri(artifact_path)
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        assert os.listdir(run_artifact_dir) == [filename]

        logged_path = os.path.join(run_artifact_dir, filename)
        loaded_image = Image.open(logged_path)
        # How to check Pillow image equality: https://stackoverflow.com/a/6204954/6943581
        assert ImageChops.difference(loaded_image, image).getbbox() is None


@pytest.mark.parametrize(
    "size",
    [
        (100, 100),  # Grayscale (2D)
        (100, 100, 1),  # Grayscale (3D)
        (100, 100, 3),  # RGB
        (100, 100, 4),  # RGBA
    ],
)
def test_log_image_numpy_shape(size):
    import numpy as np

    filename = "image.png"
    image = np.random.randint(0, 256, size=size, dtype=np.uint8)

    with mlflow.start_run():
        mlflow.log_image(image, filename)
        artifact_uri = mlflow.get_artifact_uri()
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        assert os.listdir(run_artifact_dir) == [filename]


@pytest.mark.parametrize(
    "dtype",
    [
        # Ref.: https://numpy.org/doc/stable/user/basics.types.html#array-types-and-conversions-between-types
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "bool",
    ],
)
def test_log_image_numpy_dtype(dtype):
    import numpy as np

    filename = "image.png"
    image = np.random.randint(0, 2, size=(100, 100, 3)).astype(np.dtype(dtype))

    with mlflow.start_run():
        mlflow.log_image(image, filename)
        artifact_uri = mlflow.get_artifact_uri()
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        assert os.listdir(run_artifact_dir) == [filename]


@pytest.mark.parametrize(
    "array",
    # 1 pixel images with out-of-range values
    [[[-1]], [[256]], [[-0.1]], [[1.1]]],
)
def test_log_image_numpy_emits_warning_for_out_of_range_values(array):
    import numpy as np

    image = np.array(array).astype(type(array[0][0]))
    if isinstance(array[0][0], int):
        with mlflow.start_run(), pytest.raises(
            ValueError, match="Integer pixel values out of acceptable range"
        ):
            mlflow.log_image(image, "image.png")
    else:
        with mlflow.start_run(), pytest.raises(
            ValueError, match="Float pixel values out of acceptable range"
        ):
            mlflow.log_image(image, "image.png")


def test_log_image_numpy_raises_exception_for_invalid_array_data_type():
    import numpy as np

    with mlflow.start_run(), pytest.raises(TypeError, match="Invalid array data type"):
        mlflow.log_image(np.tile("a", (1, 1, 3)), "image.png")


def test_log_image_numpy_raises_exception_for_invalid_array_shape():
    import numpy as np

    with mlflow.start_run(), pytest.raises(ValueError, match="`image` must be a 2D or 3D array"):
        mlflow.log_image(np.zeros((1,), dtype=np.uint8), "image.png")


def test_log_image_numpy_raises_exception_for_invalid_channel_length():
    import numpy as np

    with mlflow.start_run(), pytest.raises(ValueError, match="Invalid channel length"):
        mlflow.log_image(np.zeros((1, 1, 5), dtype=np.uint8), "image.png")


def test_log_image_raises_exception_for_unsupported_image_object_type():
    with mlflow.start_run(), pytest.raises(TypeError, match="Unsupported image object type"):
        mlflow.log_image("not_image", "image.png")


def test_log_image_with_steps():
    import numpy as np
    from PIL import Image

    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    with mlflow.start_run():
        mlflow.log_image(image, key="dog", step=0)

        logged_path = "images/dog"
        artifact_uri = mlflow.get_artifact_uri(logged_path)
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        files = os.listdir(run_artifact_dir)

        # .png file for the image, and .json file for metadata
        assert len(files) == 2
        for file in files:
            assert file.startswith("dog_step_0")
            logged_path = os.path.join(run_artifact_dir, file)
            if file.endswith(".png"):
                loaded_image = np.asarray(Image.open(logged_path), dtype=np.uint8)
                np.testing.assert_array_equal(loaded_image, image)
            elif file.endswith(".json"):
                with open(logged_path) as f:
                    metadata = json.load(f)
                    assert metadata["filepath"].startswith("images/dog/dog_step_0")
                    assert metadata["key"] == "dog"
                    assert metadata["step"] == 0
                    assert metadata["timestamp"] <= get_current_time_millis()


def test_log_image_with_timestamp():
    import numpy as np
    from PIL import Image

    image = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    with mlflow.start_run():
        mlflow.log_image(image, key="dog", timestamp=100)

        logged_path = "images/dog"
        artifact_uri = mlflow.get_artifact_uri(logged_path)
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        files = os.listdir(run_artifact_dir)

        # .png file for the image, and .json file for metadata
        assert len(files) == 2
        for file in files:
            assert file.startswith("dog_step_0")
            logged_path = os.path.join(run_artifact_dir, file)
            if file.endswith(".png"):
                loaded_image = np.asarray(Image.open(logged_path), dtype=np.uint8)
                np.testing.assert_array_equal(loaded_image, image)
            elif file.endswith(".json"):
                with open(logged_path) as f:
                    metadata = json.load(f)
                    assert metadata["filepath"].startswith("images/dog/dog_step_0")
                    assert metadata["key"] == "dog"
                    assert metadata["step"] == 0
                    assert metadata["timestamp"] == 100


def test_duplicated_log_image_with_step():
    """
    MLflow will save both files if there are multiple calls to log_image
    with the same key and step.
    """
    import numpy as np

    image1 = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    image2 = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    with mlflow.start_run():
        mlflow.log_image(image1, key="dog", step=100)
        mlflow.log_image(image2, key="dog", step=100)

        logged_path = "images/dog"
        artifact_uri = mlflow.get_artifact_uri(logged_path)
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        files = os.listdir(run_artifact_dir)
        assert len(files) == 4


def test_duplicated_log_image_with_timestamp():
    """
    MLflow will save both files if there are multiple calls to log_image
    with the same key, step, and timestamp.
    """
    import numpy as np

    image1 = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    image2 = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    with mlflow.start_run():
        mlflow.log_image(image1, key="dog", step=100, timestamp=100)
        mlflow.log_image(image2, key="dog", step=100, timestamp=100)

        logged_path = "images/dog"
        artifact_uri = mlflow.get_artifact_uri(logged_path)
        run_artifact_dir = local_file_uri_to_path(artifact_uri)
        files = os.listdir(run_artifact_dir)
        assert len(files) == 4


@pytest.mark.parametrize(
    "args",
    [
        {"key": "image"},
        {"step": 0},
        {"timestamp": 0},
        {"timestamp": 0, "step": 0},
        ["image"],
        ["image", 0],
    ],
)
def test_log_image_raises_exception_for_unexpected_arguments_used(args):
    # It will overwrite if the user wants the exact same timestamp for the logged images
    import numpy as np

    exception = "The `artifact_file` parameter cannot be used in conjunction"
    if isinstance(args, dict):
        with mlflow.start_run(), pytest.raises(TypeError, match=exception):
            mlflow.log_image(np.zeros((1,), dtype=np.uint8), "image.png", **args)
    elif isinstance(args, list):
        with mlflow.start_run(), pytest.raises(TypeError, match=exception):
            mlflow.log_image(np.zeros((1,), dtype=np.uint8), "image.png", *args)


def test_log_image_raises_exception_for_missing_arguments():
    import numpy as np

    exception = "Invalid arguments: Please specify exactly one of `artifact_file` or `key`"
    with mlflow.start_run(), pytest.raises(TypeError, match=exception):
        mlflow.log_image(np.zeros((1,), dtype=np.uint8))
