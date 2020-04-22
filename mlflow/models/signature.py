"""
The :py:mod:`mlflow.models.signature` module provides an API for specification of model signature.

Model signature defines schema of model input and output. See :py:class:`mlflow.types.Schema` for
more details on Schema and data types.

Model signature can be infered from datasets specifying valid model input (e.g. the training
dataset) and valid model output (e.g. model predictions on the training dataset) by calling
:py:func:`mlflow.models.utils.infer_signature`.

"""
import json
from typing import Dict, Any

from mlflow.types.schema import Schema
from mlflow.types.utils import MlflowModelDataset, infer_schema


class ModelSignature(object):
    """
    ModelSignature specifies schema of model's inputs and outputs.

    The current supported schema for both the input and the output is a data-frame like schema
    defined as a list of column specification :py:class:`ColSpec`. Columns can be named and must
    specify their data type. Currently the list of supported types is limited to scalar data types
    defined in :py:class:`DataType` enum.

    ModelSignature can be inferred from training dataset and model predictions using
    :py:func:`mlflow.models.signature.infer_signature`, or alternatively constructed by hand by
    passing a lists of input and output column specifications.
    """

    def __init__(self, inputs: Schema, outputs: Schema = None):
        if not isinstance(inputs, Schema):
            raise TypeError("inputs must be mlflow.models.signature.Schema, got '{}'".format(
                type(inputs)))
        if outputs is not None and not isinstance(outputs, Schema):
            raise TypeError("outputs must be either None or mlflow.models.signature.Schema, "
                            "got '{}'".format(type(inputs)))
        self.inputs = inputs
        self.outputs = outputs

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize into a 'jsonable' dictionary.

        Input and output schema are represented as json strings. This is so that the
        representation is compact when embedded in a MLmofel yaml file.

        :return: dictionary representation with input and output shcema represented as json strings.
        """

        return {
            "inputs": self.inputs.to_json(),
            "outputs": self.outputs.to_json() if self.outputs is not None else None
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
        return self.inputs == other.inputs and self.outputs == other.outputs

    def __repr__(self) -> str:
        return json.dumps({"ModelSignature": self.to_dict()}, indent=2)


def infer_signature(model_input: MlflowModelDataset,
                    model_output: MlflowModelDataset = None) -> ModelSignature:
    """
    Infer an MLflow model signature from the training data (input) and model predictions (output).
    This method captures the column names and data types from the user data. The signature
    represents model input and output as dataframes with (optionally) named columns and data type
    specified as one of types defined in :py:class:`DataType`. This method will raise
    an exception if the user data contains incompatible types or is not passed in one of the
    supported formats (containers).

    The input should be one of these:
      - pandas.DataFrame
      - dictionary of { name -> numpy.ndarray}
      - numpy.ndarray
      - pyspark.sql.DataFrame

    The element types should be mappable to one of :py:class:`mlflow.models.signature.DataType`.

    NOTE: Multidimensional (>2d) arrays (aka tensors) are not supported at this time.

    :param model_input: Valid input to the model. E.g. (a subset of) the training dataset.
    :param model_output: Valid model output. E.g. Model predictions for the (subset of) training
                         dataset.
    :return: ModelSignature
    """
    inputs = infer_schema(model_input)
    outputs = infer_schema(model_output) if model_output is not None else None
    return ModelSignature(inputs, outputs)
