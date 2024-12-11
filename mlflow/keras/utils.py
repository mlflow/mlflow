import numpy as np

from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec


def get_model_signature(model):
    def replace_none_in_shape(shape):
        return [-1 if dim_size is None else dim_size for dim_size in shape]

    input_shape = model.input_shape
    input_dtype = model.input_dtype
    output_shape = model.output_shape
    output_dtype = model.compute_dtype

    if isinstance(input_shape, list):
        input_schema = Schema(
            [
                TensorSpec(np.dtype(input_dtype), replace_none_in_shape(shape))
                for shape in input_shape
            ]
        )
    else:
        input_schema = Schema(
            [TensorSpec(np.dtype(input_dtype), replace_none_in_shape(input_shape))]
        )
    if isinstance(output_shape, list):
        output_schema = Schema(
            [
                TensorSpec(np.dtype(output_dtype), replace_none_in_shape(shape))
                for shape in output_shape
            ]
        )
    else:
        output_schema = Schema(
            [TensorSpec(np.dtype(output_dtype), replace_none_in_shape(output_shape))]
        )
    return ModelSignature(inputs=input_schema, outputs=output_schema)
