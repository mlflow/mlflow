"""
Internal module implementing multi-media objects and utilities in MLflow. Multi-media objects are
exposed to users at the top-level :py:mod:`mlflow` module.
"""

import warnings
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    import numpy
    import PIL


COMPRESSED_IMAGE_SIZE = 256


def compress_image_size(
    image: "PIL.Image.Image", max_size: int | None = COMPRESSED_IMAGE_SIZE
) -> "PIL.Image.Image":
    """
    Scale the image to fit within a square with length `max_size` while maintaining
    the aspect ratio.
    """
    # scale the image to max(width, height) <= compressed_file_max_size
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(height * (new_width / width))
    else:
        new_height = max_size
        new_width = int(width * (new_height / height))
    return image.resize((new_width, new_height))


def convert_to_pil_image(image: Union["numpy.ndarray", list[Any]]) -> "PIL.Image.Image":
    """
    Convert a numpy array to a PIL image.
    """
    import numpy as np

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to serialize a numpy array as an image. "
            "Please install it via: `pip install Pillow`"
        ) from exc

    def _normalize_to_uint8(x):
        is_int = np.issubdtype(x.dtype, np.integer)
        low = 0
        high = 255 if is_int else 1
        if x.min() < low or x.max() > high:
            if is_int:
                raise ValueError(
                    "Integer pixel values out of acceptable range [0, 255]. "
                    f"Found minimum value {x.min()} and maximum value {x.max()}. "
                    "Ensure all pixel values are within the specified range."
                )
            else:
                warnings.warn(
                    "Float pixel values out of acceptable range [0.0, 1.0]. "
                    f"Found minimum value {x.min()} and maximum value {x.max()}. "
                    "Rescaling values to [0.0, 1.0] with min/max scaler.",
                    stacklevel=2,
                )
                # Min-max scaling
                x = (x - x.min()) / (x.max() - x.min())

        # float or bool
        if not is_int:
            x = x * 255

        return x.astype(np.uint8)

    # Ref.: https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html#numpy-dtype-kind
    valid_data_types = {
        "b": "bool",
        "i": "signed integer",
        "u": "unsigned integer",
        "f": "floating",
    }

    if image.dtype.kind not in valid_data_types:
        raise TypeError(
            f"Invalid array data type: '{image.dtype}'. "
            f"Must be one of {list(valid_data_types.values())}"
        )

    if image.ndim not in [2, 3]:
        raise ValueError(f"`image` must be a 2D or 3D array but got image shape: {image.shape}")

    if (image.ndim == 3) and (image.shape[2] not in [1, 3, 4]):
        raise ValueError(f"Invalid channel length: {image.shape[2]}. Must be one of [1, 3, 4]")

    # squeeze a 3D grayscale image since `Image.fromarray` doesn't accept it.
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]

    image = _normalize_to_uint8(image)
    return Image.fromarray(image)


# MLflow media object: Image
class Image:
    """
    `mlflow.Image` is an image media object that provides a lightweight option
    for handling images in MLflow.
    The image can be a numpy array, a PIL image, or a file path to an image. The image is
    stored as a PIL image and can be logged to MLflow using `mlflow.log_image` or
    `mlflow.log_table`.

    Args:
        image: Image can be a numpy array, a PIL image, or a file path to an image.

    .. code-block:: python
        :caption: Example

        import mlflow
        import numpy as np
        from PIL import Image

        # Create an image as a numpy array
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :50] = [255, 128, 0]
        # Create an Image object
        image_obj = mlflow.Image(image)
        # Convert the Image object to a list of pixel values
        pixel_values = image_obj.to_list()
    """

    def __init__(self, image: Union["numpy.ndarray", "PIL.Image.Image", str, list[Any]]):
        import numpy as np

        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "`mlflow.Image` requires Pillow to serialize a numpy array as an image. "
                "Please install it via: `pip install Pillow`."
            ) from exc

        if isinstance(image, str):
            self.image = Image.open(image)
        elif isinstance(image, (list, np.ndarray)):
            self.image = convert_to_pil_image(np.array(image))
        elif isinstance(image, Image.Image):
            self.image = image
        else:
            raise TypeError(
                f"Unsupported image object type: {type(image)}. "
                "`image` must be one of numpy.ndarray, "
                "PIL.Image.Image, or a filepath to an image."
            )
        self.size = self.image.size

    def to_list(self):
        """
        Convert the image to a list of pixel values.

        Returns:
            List of pixel values.
        """
        return list(self.image.getdata())

    def to_array(self):
        """
        Convert the image to a numpy array.

        Returns:
            Numpy array of pixel values.
        """
        import numpy as np

        return np.array(self.image)

    def to_pil(self):
        """
        Convert the image to a PIL image.

        Returns:
            PIL image.
        """
        return self.image

    def save(self, path: str):
        """
        Save the image to a file.

        Args:
            path: File path to save the image.
        """
        self.image.save(path)

    def resize(self, size: tuple[int, int]):
        """
        Resize the image to the specified size.

        Args:
            size: Size to resize the image to.

        Returns:
            A copy of the resized image object.
        """
        image = self.image.resize(size)
        return Image(image)
