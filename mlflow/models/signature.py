"""
The :py:mod:`mlflow.models.signature` module provides an API for specification of model signature.

Model signature defines schema of model input and output. See :py:class:`mlflow.types.schema.Schema`
for more details on Schema and data types.
"""

import inspect
import logging
import re
from copy import deepcopy
from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, get_type_hints

import numpy as np
import pandas as pd

from mlflow import environment_variables
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _contains_params, _Example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri, _upload_artifact_to_uri
from mlflow.types.schema import ParamSchema, Schema, convert_dataclass_to_schema
from mlflow.types.utils import _infer_param_schema, _infer_schema, _infer_schema_from_type_hint
from mlflow.utils.uri import append_to_uri_path

# At runtime, we don't need  `pyspark.sql.dataframe`
if TYPE_CHECKING:
    try:
        import pyspark.sql.dataframe

        MlflowInferableDataset = Union[
            pd.DataFrame, np.ndarray, Dict[str, np.ndarray], pyspark.sql.dataframe.DataFrame
        ]
    except ImportError:
        MlflowInferableDataset = Union[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]

_logger = logging.getLogger(__name__)

_LOG_MODEL_INFER_SIGNATURE_WARNING_TEMPLATE = (
    "Failed to infer the model signature from the input example. Reason: %s. To see the full "
    "traceback, set the logging level to DEBUG via "
    '`logging.getLogger("mlflow").setLevel(logging.DEBUG)`. To disable automatic signature '
    "inference, set `signature` to `False` in your `log_model` or `save_model` call."
)


class ModelSignature:
    """
    ModelSignature specifies schema of model's inputs, outputs and params.

    ModelSignature can be :py:func:`inferred <mlflow.models.infer_signature>` from training
    dataset, model predictions using and params for inference, or constructed by hand by
    passing an input and output :py:class:`Schema <mlflow.types.Schema>`, and params
    :py:class:`ParamSchema <mlflow.types.ParamSchema>`.
    """

    def __init__(
        self,
        inputs: Union[Schema, dataclass] = None,
        outputs: Union[Schema, dataclass] = None,
        params: ParamSchema = None,
    ):
        if inputs and not isinstance(inputs, Schema) and not is_dataclass(inputs):
            raise TypeError(
                "inputs must be either None, mlflow.models.signature.Schema, or a dataclass,"
                f"got '{type(inputs).__name__}'"
            )
        if outputs and not isinstance(outputs, Schema) and not is_dataclass(outputs):
            raise TypeError(
                "outputs must be either None, mlflow.models.signature.Schema, or a dataclass,"
                f"got '{type(outputs).__name__}'"
            )
        if params and not isinstance(params, ParamSchema):
            raise TypeError(
                "If params are provided, they must by of type mlflow.models.signature.ParamSchema, "
                f"got '{type(params).__name__}'"
            )
        if all(x is None for x in [inputs, outputs, params]):
            raise ValueError("At least one of inputs, outputs or params must be provided")
        if is_dataclass(inputs):
            self.inputs = convert_dataclass_to_schema(inputs)
        else:
            self.inputs = inputs
        if is_dataclass(outputs):
            self.outputs = convert_dataclass_to_schema(outputs)
        else:
            self.outputs = outputs
        self.params = params

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize into a 'jsonable' dictionary.

        Input and output schema are represented as json strings. This is so that the
        representation is compact when embedded in an MLmodel yaml file.

        Returns:
            dictionary representation with input and output schema represented as json strings.
        """

        return {
            "inputs": self.inputs.to_json() if self.inputs else None,
            "outputs": self.outputs.to_json() if self.outputs else None,
            "params": self.params.to_json() if self.params else None,
        }

    @classmethod
    def from_dict(cls, signature_dict: Dict[str, Any]):
        """
        Deserialize from dictionary representation.

        Args:
            signature_dict: Dictionary representation of model signature.
                Expected dictionary format:
                `{'inputs': <json string>,
                'outputs': <json string>,
                'params': <json string>" }`

        Returns:
            ModelSignature populated with the data form the dictionary.
        """
        inputs = Schema.from_json(x) if (x := signature_dict.get("inputs")) else None
        outputs = Schema.from_json(x) if (x := signature_dict.get("outputs")) else None
        params = ParamSchema.from_json(x) if (x := signature_dict.get("params")) else None

        return cls(inputs, outputs, params)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ModelSignature)
            and self.inputs == other.inputs
            and self.outputs == other.outputs
            and self.params == other.params
        )

    def __repr__(self) -> str:
        return (
            "inputs: \n"
            f"  {self.inputs!r}\n"
            "outputs: \n"
            f"  {self.outputs!r}\n"
            "params: \n"
            f"  {self.params!r}\n"
        )


def infer_signature(
    model_input: Any = None,
    model_output: "MlflowInferableDataset" = None,
    params: Optional[Dict[str, Any]] = None,
) -> ModelSignature:
    """
    Infer an MLflow model signature from the training data (input), model predictions (output)
    and parameters (for inference).

    The signature represents model input and output as data frames with (optionally) named columns
    and data type specified as one of types defined in :py:class:`mlflow.types.DataType`. It also
    includes parameters schema for inference, .
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
      - dictionary / list of dictionaries of JSON-convertible types

    The element types should be mappable to one of :py:class:`mlflow.types.DataType`.

    For pyspark.sql.DataFrame inputs, columns of type DateType and TimestampType are both inferred
    as type :py:data:`datetime <mlflow.types.DataType.datetime>`, which is coerced to
    TimestampType at inference.

    Args:
      model_input: Valid input to the model. E.g. (a subset of) the training dataset.
      model_output: Valid model output. E.g. Model predictions for the (subset of) training
                    dataset.
      params: Valid parameters for inference. It should be a dictionary of parameters
              that can be set on the model during inference by passing `params` to pyfunc
              `predict` method.

              An example of valid parameters:

              .. code-block:: python

                    from mlflow.models import infer_signature
                    from mlflow.transformers import generate_signature_output

                    # Define parameters for inference
                    params = {
                        "num_beams": 5,
                        "max_length": 30,
                        "do_sample": True,
                        "remove_invalid_values": True,
                    }

                    # Infer the signature including parameters
                    signature = infer_signature(
                        data,
                        generate_signature_output(model, data),
                        params=params,
                    )

                    # Saving model with model signature
                    mlflow.transformers.save_model(
                        model,
                        path=model_path,
                        signature=signature,
                    )

                    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)

                    # Passing params to `predict` function directly
                    result = pyfunc_loaded.predict(data, params=params)

    Returns:
      ModelSignature
    """
    inputs = _infer_schema(model_input) if model_input is not None else None
    outputs = _infer_schema(model_output) if model_output is not None else None
    params = _infer_param_schema(params) if params else None
    return ModelSignature(inputs, outputs, params)


# `t\w*\.` matches the `typing` module or its alias
_LIST_OF_STRINGS_PATTERN = re.compile(r"^(t\w*\.)?list\[str\]$", re.IGNORECASE)


def _is_list_str(hint_str):
    return _LIST_OF_STRINGS_PATTERN.match(hint_str.replace(" ", "")) is not None


_LIST_OF_STR_DICT_PATTERN = re.compile(
    r"^(t\w*\.)?list\[(t\w*\.)?dict\[str,str\]\]$", re.IGNORECASE
)


def _is_list_of_string_dict(hint_str):
    return _LIST_OF_STR_DICT_PATTERN.match(hint_str.replace(" ", "")) is not None


def _infer_hint_from_str(hint_str):
    if _is_list_str(hint_str):
        return List[str]
    elif _is_list_of_string_dict(hint_str):
        return List[Dict[str, str]]
    else:
        return None


def _get_arg_names(f):
    return list(inspect.signature(f).parameters.keys())


class _TypeHints:
    def __init__(self, input_=None, output=None):
        self.input = input_
        self.output = output

    def __repr__(self):
        return f"<input: {self.input}, output: {self.output}>"


def _extract_type_hints(f, input_arg_index):
    """
    Extract type hints from a function.

    Args:
        f: Function to extract type hints from.
        input_arg_index: Index of the function argument that corresponds to the model input.

    Returns:
        A `_TypeHints` object containing the input and output type hints.
    """
    if not hasattr(f, "__annotations__") and hasattr(f, "__call__"):
        return _extract_type_hints(f.__call__, input_arg_index)

    if f.__annotations__ == {}:
        return _TypeHints()

    arg_names = _get_arg_names(f)
    if len(arg_names) - 1 < input_arg_index:
        raise MlflowException.invalid_parameter_value(
            f"The specified input argument index ({input_arg_index}) is out of range for the "
            "function signature: {}".format(input_arg_index, arg_names)
        )
    arg_name = _get_arg_names(f)[input_arg_index]
    try:
        hints = get_type_hints(f)
    except TypeError:
        # ---
        # from __future__ import annotations # postpones evaluation of 'list[str]'
        #
        # def f(x: list[str]) -> list[str]:
        #          ^^^^^^^^^ Evaluating this expression ('list[str]') results in a TypeError in
        #                    Python < 3.9 because the built-in list type is not subscriptable.
        #     return x
        # ---
        # Best effort to infer type hints from strings
        hints = {}
        for arg in [arg_name, "return"]:
            if hint_str := f.__annotations__.get(arg, None):
                if hint := _infer_hint_from_str(hint_str):
                    hints[arg] = hint
                else:
                    _logger.info("Unsupported type hint: %s, skipping schema inference", hint_str)
    except Exception as e:
        _logger.warning("Failed to extract type hints from function %s: %s", f.__name__, repr(e))
        return _TypeHints()

    return _TypeHints(hints.get(arg_name), hints.get("return"))


def _infer_signature_from_type_hints(func, input_arg_index, input_example=None):
    hints = _extract_type_hints(func, input_arg_index)
    if hints.input is None:
        return None

    params = None
    params_key = "params"
    if _contains_params(input_example):
        input_example, params = input_example

    input_schema = _infer_schema_from_type_hint(hints.input, input_example) if hints.input else None
    params_schema = _infer_param_schema(params) if params else None
    input_arg_name = _get_arg_names(func)[input_arg_index]
    if input_example:
        inputs = {input_arg_name: input_example}
        if params and params_key in inspect.signature(func).parameters:
            inputs[params_key] = params
        # This is for PythonModel's predict function
        if input_arg_index == 1:
            inputs["context"] = None
        output_example = func(**inputs)
    else:
        output_example = None
    output_schema = (
        _infer_schema_from_type_hint(hints.output, output_example) if hints.output else None
    )
    if not any([input_schema, output_schema, params_schema]):
        return None
    return ModelSignature(inputs=input_schema, outputs=output_schema, params=params_schema)


def _infer_signature_from_input_example(
    input_example: Optional[_Example], wrapped_model
) -> Optional[ModelSignature]:
    """
    Infer the signature from an example input and a PyFunc wrapped model. Catches all exceptions.

    Args:
        input_example: Saved _Example object that contains input example instance.
        wrapped_model: A PyFunc wrapped model which has a `predict` method.

    Returns:
        A `ModelSignature` object containing the inferred schema of both the model's inputs
        based on the `input_example` and the model's outputs based on the prediction from the
        `wrapped_model`.
    """
    from mlflow.pyfunc import _validate_prediction_input

    if input_example is None:
        return None

    try:
        # Copy the input example so that it is not mutated by predict()
        input_data = deepcopy(input_example.inference_data)
        params = input_example.inference_params

        input_schema = _infer_schema(input_data)
        params_schema = _infer_param_schema(params) if params else None
        # do the same validation as pyfunc predict to make sure the signature is correctly
        # applied to the model
        input_data, params = _validate_prediction_input(
            input_data, params, input_schema, params_schema
        )
        prediction = wrapped_model.predict(input_data, params=params)
        # For column-based inputs, 1D numpy arrays likely signify row-based predictions. Thus, we
        # convert them to a Pandas series for inferring as a single ColSpec Schema.
        if (
            not input_schema.is_tensor_spec()
            and isinstance(prediction, np.ndarray)
            and prediction.ndim == 1
        ):
            prediction = pd.Series(prediction)
        try:
            output_schema = _infer_schema(prediction)
        except Exception as e:
            _logger.warning(
                "Failed to infer model output schema from prediction "
                f"result {prediction}. Detailed exception: {e}"
            )
            signature = ModelSignature(inputs=input_schema, params=params_schema)
        else:
            signature = ModelSignature(input_schema, output_schema, params_schema)

        # try assign output schema if failing to infer it from prediction for langchain models
        if signature.outputs is None:
            try:
                from mlflow.langchain import _LangChainModelWrapper
                from mlflow.langchain.utils.chat import _ChatResponse
            except ImportError:
                pass
            else:
                if isinstance(wrapped_model, _LangChainModelWrapper) and isinstance(
                    prediction, _ChatResponse
                ):
                    signature.outputs = prediction.get_schema()

        return signature
    except Exception as e:
        if environment_variables._MLFLOW_TESTING.get():
            raise
        _logger.warning(_LOG_MODEL_INFER_SIGNATURE_WARNING_TEMPLATE, repr(e))
        _logger.debug("", exc_info=True)
        return None


def set_signature(
    model_uri: str,
    signature: ModelSignature,
):
    """
    Sets the model signature for specified model artifacts.

    The process involves downloading the MLmodel file in the model artifacts (if it's non-local),
    updating its model signature, and then overwriting the existing MLmodel file. Should the
    artifact repository associated with the model artifacts disallow overwriting, this function will
    fail.

    Furthermore, as model registry artifacts are read-only, model artifacts located in the
    model registry and represented by ``models:/`` URI schemes are not compatible with this API.
    To set a signature on a model version, first set the signature on the source model artifacts.
    Following this, generate a new model version using the updated model artifacts. For more
    information about setting signatures on model versions, see
    `this doc section <https://www.mlflow.org/docs/latest/models.html#set-signature-on-mv>`_.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``mlflow-artifacts:/path/to/model``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
            artifact-locations>`_.

            Please note that model URIs with the ``models:/`` scheme are not supported.

        signature: ModelSignature to set on the model.

    .. code-block:: python
        :caption: Example

        import mlflow
        from mlflow.models import set_signature, infer_signature

        # load model from run artifacts
        run_id = "96771d893a5e46159d9f3b49bf9013e2"
        artifact_path = "models"
        model_uri = f"runs:/{run_id}/{artifact_path}"
        model = mlflow.pyfunc.load_model(model_uri)

        # determine model signature
        test_df = ...
        predictions = model.predict(test_df)
        signature = infer_signature(test_df, predictions)

        # set the signature for the logged model
        set_signature(model_uri, signature)
    """
    assert isinstance(
        signature, ModelSignature
    ), "The signature argument must be a ModelSignature object"
    if ModelsArtifactRepository.is_models_uri(model_uri):
        raise MlflowException(
            f'Failed to set signature on "{model_uri}". '
            + "Model URIs with the `models:/` scheme are not supported.",
            INVALID_PARAMETER_VALUE,
        )
    try:
        resolved_uri = model_uri
        if RunsArtifactRepository.is_runs_uri(model_uri):
            resolved_uri = RunsArtifactRepository.get_underlying_uri(model_uri)
        ml_model_file = _download_artifact_from_uri(
            artifact_uri=append_to_uri_path(resolved_uri, MLMODEL_FILE_NAME)
        )
    except Exception as ex:
        raise MlflowException(
            f'Failed to download an "{MLMODEL_FILE_NAME}" model file from "{model_uri}"',
            RESOURCE_DOES_NOT_EXIST,
        ) from ex
    model_meta = Model.load(ml_model_file)
    model_meta.signature = signature
    model_meta.save(ml_model_file)
    _upload_artifact_to_uri(ml_model_file, resolved_uri)
