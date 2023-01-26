"""
The :py:mod:`mlflow.models.signature` module provides an API for specification of model signature.

Model signature defines schema of model input and output. See :py:class:`mlflow.types.schema.Schema`
for more details on Schema and data types.
"""
from typing import Dict, Any, Union, TYPE_CHECKING

import pandas as pd
import numpy as np

from mlflow.types.schema import Schema
from mlflow.types.utils import _infer_schema


# At runtime, we don't need  `pyspark.sql.dataframe`
if TYPE_CHECKING:
    try:
        import pyspark.sql.dataframe

        MlflowInferableDataset = Union[
            pd.DataFrame, np.ndarray, Dict[str, np.ndarray], pyspark.sql.dataframe.DataFrame
        ]
    except ImportError:
        MlflowInferableDataset = Union[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]


class ModelSignature:
    """
    ModelSignature specifies schema of model's inputs and outputs.

    ModelSignature can be :py:func:`inferred <mlflow.models.infer_signature>` from training dataset
    and model predictions using or constructed by hand by passing an input and output
    :py:class:`Schema <mlflow.types.Schema>`.
    """

    def __init__(self, inputs: Schema, outputs: Schema = None):
        if not isinstance(inputs, Schema):
            raise TypeError(
                "inputs must be mlflow.models.signature.Schema, got '{}'".format(type(inputs))
            )
        if outputs is not None and not isinstance(outputs, Schema):
            raise TypeError(
                "outputs must be either None or mlflow.models.signature.Schema, "
                "got '{}'".format(type(inputs))
            )
        self.inputs = inputs
        self.outputs = outputs

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize into a 'jsonable' dictionary.

        Input and output schema are represented as json strings. This is so that the
        representation is compact when embedded in a MLmofel yaml file.

        :return: dictionary representation with input and output schema represented as json strings.
        """

        return {
            "inputs": self.inputs.to_json(),
            "outputs": self.outputs.to_json() if self.outputs is not None else None,
        }

    @classmethod
    def from_dict(cls, signature_dict: Dict[str, Any]):
        """
        Deserialize from dictionary representation.

        :param signature_dict: Dictionary representation of model signature.
                               Expected dictionary format:
                               `{'inputs': <json string>, 'outputs': <json string>" }`

        :return: ModelSignature populated with the data form the dictionary.
        """
        inputs = Schema.from_json(signature_dict["inputs"])
        if "outputs" in signature_dict and signature_dict["outputs"] is not None:
            outputs = Schema.from_json(signature_dict["outputs"])
            return cls(inputs, outputs)
        else:
            return cls(inputs)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ModelSignature)
            and self.inputs == other.inputs
            and self.outputs == other.outputs
        )

    def __repr__(self) -> str:
        return (
            "inputs: \n"
            "  {}\n"
            "outputs: \n"
            "  {}\n".format(repr(self.inputs), repr(self.outputs))
        )


def infer_signature(
    model_input: Any, model_output: "MlflowInferableDataset" = None
) -> ModelSignature:
    """
    Infer an MLflow model signature from the training data (input) and model predictions (output).

    The signature represents model input and output as data frames with (optionally) named columns
    and data type specified as one of types defined in :py:class:`mlflow.types.DataType`.
    This method will raise an exception if the user data contains incompatible types or is not
    passed in one of the supported formats listed below.

    The input should be one of these:
      - pandas.DataFrame
      - pandas.Series
      - dictionary of { name -> numpy.ndarray}
      - numpy.ndarray
      - pyspark.sql.DataFrame
      - scipy.sparse.csr_matrix
      - scipy.sparse.csc_matrix

    The element types should be mappable to one of :py:class:`mlflow.types.DataType`.

    For pyspark.sql.DataFrame inputs, columns of type DateType and TimestampType are both inferred
    as type :py:data:`datetime <mlflow.types.DataType.datetime>`, which is coerced to
    TimestampType at inference.

    :param model_input: Valid input to the model. E.g. (a subset of) the training dataset.
    :param model_output: Valid model output. E.g. Model predictions for the (subset of) training
                         dataset.
    :return: ModelSignature
    """
    inputs = _infer_schema(model_input)
    outputs = _infer_schema(model_output) if model_output is not None else None
    return ModelSignature(inputs, outputs)
