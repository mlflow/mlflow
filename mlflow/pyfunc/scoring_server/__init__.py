"""
Scoring server for python model format.
The passed int model is expected to have function:
   predict(pandas.Dataframe) -> pandas.DataFrame

Input, expected in text/csv or application/json format,
is parsed into pandas.DataFrame and passed to the model.

Defines four endpoints:
    /ping used for health check
    /health (same as /ping)
    /version used for getting the mlflow version
    /invocations used for scoring
"""

import asyncio
import inspect
import json
import logging
import os
import shlex
import sys
import traceback
from functools import wraps
from typing import Any, NamedTuple, Optional

from mlflow.environment_variables import (
    _MLFLOW_IS_IN_SERVING_ENVIRONMENT,
    MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT,
)

# NB: We need to be careful what we import form mlflow here. Scoring server is used from within
# model's conda environment. The version of mlflow doing the serving (outside) and the version of
# mlflow in the model's conda environment (inside) can differ. We should therefore keep mlflow
# dependencies to the minimum here.
# ALl of the mlflow dependencies below need to be backwards compatible.
from mlflow.exceptions import MlflowException
from mlflow.pyfunc.model import _log_warning_if_params_not_in_predict_signature
from mlflow.types import ParamSchema, Schema
from mlflow.utils import reraise
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.proto_json_utils import (
    MlflowInvalidInputException,
    NumpyEncoder,
    _get_jsonable_obj,
    dataframe_from_parsed_json,
    parse_tf_serving_input,
)
from mlflow.version import VERSION

try:
    from mlflow.pyfunc import PyFuncModel, load_model
except ImportError:
    from mlflow.pyfunc import load_pyfunc as load_model
from io import StringIO

from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.pyfunc.utils.serving_data_parser import is_unified_llm_input

_SERVER_MODEL_PATH = "__pyfunc_model_path__"
SERVING_MODEL_CONFIG = "SERVING_MODEL_CONFIG"

CONTENT_TYPE_CSV = "text/csv"
CONTENT_TYPE_JSON = "application/json"

CONTENT_TYPES = [
    CONTENT_TYPE_CSV,
    CONTENT_TYPE_JSON,
]

_logger = logging.getLogger(__name__)

DF_RECORDS = "dataframe_records"
DF_SPLIT = "dataframe_split"
INSTANCES = "instances"
INPUTS = "inputs"

SUPPORTED_FORMATS = {DF_RECORDS, DF_SPLIT, INSTANCES, INPUTS}
SERVING_PARAMS_KEY = "params"

REQUIRED_INPUT_FORMAT = (
    f"The input must be a JSON dictionary with exactly one of the input fields {SUPPORTED_FORMATS}"
)
SCORING_PROTOCOL_CHANGE_INFO = (
    "IMPORTANT: The MLflow Model scoring protocol has changed in MLflow version 2.0. If you are"
    " seeing this error, you are likely using an outdated scoring request format. To resolve the"
    " error, either update your request format or adjust your MLflow Model's requirements file to"
    " specify an older version of MLflow (for example, change the 'mlflow' requirement specifier"
    " to 'mlflow==1.30.0'). If you are making a request using the MLflow client"
    " (e.g. via `mlflow.pyfunc.spark_udf()`), upgrade your MLflow client to a version >= 2.0 in"
    " order to use the new request format. For more information about the updated MLflow"
    " Model scoring protocol in MLflow 2.0, see"
    " https://mlflow.org/docs/latest/models.html#deploy-mlflow-models."
)


def load_model_with_mlflow_config(model_uri):
    extra_kwargs = {}
    if model_config_json := os.environ.get(SERVING_MODEL_CONFIG):
        extra_kwargs["model_config"] = json.loads(model_config_json)

    return load_model(model_uri, **extra_kwargs)


# Keep this method to maintain compatibility with MLServer
# https://github.com/SeldonIO/MLServer/blob/caa173ab099a4ec002a7c252cbcc511646c261a6/runtimes/mlflow/mlserver_mlflow/runtime.py#L13C5-L13C31
def infer_and_parse_json_input(json_input, schema: Schema = None):
    """
    Args:
        json_input: A JSON-formatted string representation of TF serving input or a Pandas
                    DataFrame, or a stream containing such a string representation.
        schema: Optional schema specification to be used during parsing.
    """
    if isinstance(json_input, dict):
        decoded_input = json_input
    else:
        try:
            decoded_input = json.loads(json_input)
        except json.decoder.JSONDecodeError as ex:
            raise MlflowException(
                message=(
                    "Failed to parse input from JSON. Ensure that input is a valid JSON"
                    f" formatted string. Error: '{ex}'. Input: \n{json_input}\n"
                ),
                error_code=BAD_REQUEST,
            )
    if isinstance(decoded_input, dict):
        format_keys = set(decoded_input.keys()).intersection(SUPPORTED_FORMATS)
        if len(format_keys) != 1:
            message = f"Received dictionary with input fields: {list(decoded_input.keys())}"
            raise MlflowException(
                message=f"{REQUIRED_INPUT_FORMAT}. {message}. {SCORING_PROTOCOL_CHANGE_INFO}",
                error_code=BAD_REQUEST,
            )
        input_format = format_keys.pop()
        if input_format in (INSTANCES, INPUTS):
            return parse_tf_serving_input(decoded_input, schema=schema)

        elif input_format in (DF_SPLIT, DF_RECORDS):
            # NB: skip the dataframe_ prefix
            pandas_orient = input_format[10:]
            return dataframe_from_parsed_json(
                decoded_input[input_format], pandas_orient=pandas_orient, schema=schema
            )
    elif isinstance(decoded_input, list):
        message = "Received a list"
        raise MlflowException(
            message=f"{REQUIRED_INPUT_FORMAT}. {message}. {SCORING_PROTOCOL_CHANGE_INFO}",
            error_code=BAD_REQUEST,
        )
    else:
        message = f"Received unexpected input type '{type(decoded_input)}'"
        raise MlflowException(
            message=f"{REQUIRED_INPUT_FORMAT}. {message}.", error_code=BAD_REQUEST
        )


def _decode_json_input(json_input):
    """
    Args:
        json_input: A JSON-formatted string representation of TF serving input or a Pandas
                    DataFrame, or a stream containing such a string representation.

    Returns:
        A dictionary representation of the JSON input.
    """
    if isinstance(json_input, dict):
        return json_input

    try:
        decoded_input = json.loads(json_input)
    except json.decoder.JSONDecodeError as ex:
        raise MlflowInvalidInputException(
            "Ensure that input is a valid JSON formatted string. "
            f"Error: '{ex!r}'\nInput: \n{json_input}\n"
        ) from ex

    if isinstance(decoded_input, dict):
        return decoded_input
    if isinstance(decoded_input, list):
        raise MlflowInvalidInputException(f"{REQUIRED_INPUT_FORMAT}. Received a list.")

    raise MlflowInvalidInputException(
        f"{REQUIRED_INPUT_FORMAT}. Received unexpected input type '{type(decoded_input)}."
    )


def _split_data_and_params_for_llm_input(json_input, param_schema: Optional[ParamSchema]):
    data = {}
    params = {}
    schema_params = {param.name for param in param_schema.params} if param_schema else {}

    for key, value in json_input.items():
        # if the model defines a param schema, then we can add
        # it to the params dict. otherwise, add it to the data
        # dict to prevent it from being ignored at inference time
        if key in schema_params:
            params[key] = value
        else:
            data[key] = value

    return data, params


def _split_data_and_params(json_input):
    input_dict = _decode_json_input(json_input)
    data = {k: v for k, v in input_dict.items() if k in SUPPORTED_FORMATS}
    params = input_dict.pop(SERVING_PARAMS_KEY, None)
    return data, params


def infer_and_parse_data(data, schema: Schema = None):
    """
    Args:
        data: A dictionary representation of TF serving input or a Pandas
            DataFrame, or a stream containing such a string representation.
        schema: Optional schema specification to be used during parsing.
    """

    format_keys = set(data.keys()).intersection(SUPPORTED_FORMATS)
    if len(format_keys) != 1:
        message = f"Received dictionary with input fields: {list(data.keys())}"
        raise MlflowException(
            message=f"{REQUIRED_INPUT_FORMAT}. {message}. {SCORING_PROTOCOL_CHANGE_INFO}",
            error_code=BAD_REQUEST,
        )
    input_format = format_keys.pop()
    if input_format in (INSTANCES, INPUTS):
        return parse_tf_serving_input(data, schema=schema)

    if input_format in (DF_SPLIT, DF_RECORDS):
        pandas_orient = input_format[10:]  # skip the dataframe_ prefix
        return dataframe_from_parsed_json(
            data[input_format], pandas_orient=pandas_orient, schema=schema
        )


def parse_csv_input(csv_input, schema: Schema = None):
    """
    Args:
        csv_input: A CSV-formatted string representation of a Pandas DataFrame, or a stream
                   containing such a string representation.
        schema: Optional schema specification to be used during parsing.
    """
    import pandas as pd

    try:
        if schema is None:
            return pd.read_csv(csv_input)
        else:
            dtypes = dict(zip(schema.input_names(), schema.pandas_types()))
            return pd.read_csv(csv_input, dtype=dtypes)
    except Exception as e:
        _handle_serving_error(
            error_message=(
                "Failed to parse input as a Pandas DataFrame. Ensure that the input is"
                " a valid CSV-formatted Pandas DataFrame produced using the"
                f" `pandas.DataFrame.to_csv()` method. Error: '{e}'"
            ),
            error_code=BAD_REQUEST,
        )


def unwrapped_predictions_to_json(raw_predictions, output):
    predictions = _get_jsonable_obj(raw_predictions, pandas_orient="records")
    return json.dump(predictions, output, cls=NumpyEncoder)


def predictions_to_json(raw_predictions, output, metadata=None):
    if metadata and "predictions" in metadata:
        raise MlflowException(
            "metadata cannot contain 'predictions' key", error_code=INVALID_PARAMETER_VALUE
        )
    predictions = _get_jsonable_obj(raw_predictions, pandas_orient="records")
    return json.dump({"predictions": predictions, **(metadata or {})}, output, cls=NumpyEncoder)


def _handle_serving_error(error_message, error_code, include_traceback=True):
    """
    Logs information about an exception thrown by model inference code that is currently being
    handled and reraises it with the specified error message. The exception stack trace
    is also included in the reraised error message.

    Args:
        error_message: A message for the reraised exception.
        error_code: An appropriate error code for the reraised exception. This should be one of
            the codes listed in the `mlflow.protos.databricks_pb2` proto.
        include_traceback: Whether to include the current traceback in the returned error.
    """
    if include_traceback:
        traceback_buf = StringIO()
        traceback.print_exc(file=traceback_buf)
        traceback_str = traceback_buf.getvalue()
        e = MlflowException(message=error_message, error_code=error_code, stack_trace=traceback_str)
    else:
        e = MlflowException(message=error_message, error_code=error_code)
    reraise(MlflowException, e)


class InvocationsResponse(NamedTuple):
    response: str
    status: int
    mimetype: str


def invocations(data, content_type, model, input_schema):
    type_parts = list(map(str.strip, content_type.split(";")))
    mime_type = type_parts[0]
    parameter_value_pairs = type_parts[1:]
    parameter_values = {
        key: value for pair in parameter_value_pairs for key, _, value in [pair.partition("=")]
    }

    charset = parameter_values.get("charset", "utf-8").lower()
    if charset != "utf-8":
        return InvocationsResponse(
            response="The scoring server only supports UTF-8",
            status=415,
            mimetype="text/plain",
        )

    unexpected_content_parameters = set(parameter_values.keys()).difference({"charset"})
    if unexpected_content_parameters:
        return InvocationsResponse(
            response=(
                f"Unrecognized content type parameters: "
                f"{', '.join(unexpected_content_parameters)}. "
                f"{SCORING_PROTOCOL_CHANGE_INFO}"
            ),
            status=415,
            mimetype="text/plain",
        )

    # The traditional JSON request/response format, wraps the data with one of the supported keys
    # like "dataframe_split" and "predictions". For LLM use cases, we also support unwrapped JSON
    # payload, to provide unified prediction interface.
    should_parse_as_unified_llm_input = False

    if mime_type == CONTENT_TYPE_CSV:
        # Convert from CSV to pandas
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        csv_input = StringIO(data)
        data = parse_csv_input(csv_input=csv_input, schema=input_schema)
        params = None
    elif mime_type == CONTENT_TYPE_JSON:
        parsed_json_input = _parse_json_data(data, model.metadata, input_schema)
        data = parsed_json_input.data
        params = parsed_json_input.params
        should_parse_as_unified_llm_input = parsed_json_input.is_unified_llm_input
    else:
        return InvocationsResponse(
            response=(
                "This predictor only supports the following content types:"
                f" Types: {CONTENT_TYPES}."
                f" Got '{content_type}'."
            ),
            status=415,
            mimetype="text/plain",
        )

    # Do the prediction
    # NB: utils._validate_serving_input mimic the scoring process here to validate input_example
    # work for serving, so any changes here should be reflected there as well
    try:
        if "params" in inspect.signature(model.predict).parameters:
            raw_predictions = model.predict(data, params=params)
        else:
            _log_warning_if_params_not_in_predict_signature(_logger, params)
            raw_predictions = model.predict(data)
    except MlflowException as e:
        if "Failed to enforce schema" in e.message:
            _logger.warning(
                "If using `instances` as input key, we internally convert "
                "the data type from `records` (List[Dict]) type to "
                "`list` (Dict[str, List]) type if the data is a pandas "
                "dataframe representation. This might cause schema changes. "
                "Please use `inputs` to avoid this conversion.\n"
            )
        e.message = f"Failed to predict data '{data}'. \nError: {e.message}"
        raise e
    except Exception:
        raise MlflowException(
            message=(
                "Encountered an unexpected error while evaluating the model. Verify"
                " that the serialized input Dataframe is compatible with the model for"
                " inference."
            ),
            error_code=BAD_REQUEST,
            stack_trace=traceback.format_exc(),
        )
    result = StringIO()

    # if the data was formatted using the unified LLM format,
    # then return the data without the "predictions" key
    if should_parse_as_unified_llm_input:
        unwrapped_predictions_to_json(raw_predictions, result)
    else:
        predictions_to_json(raw_predictions, result)

    return InvocationsResponse(response=result.getvalue(), status=200, mimetype="application/json")


class ParsedJsonInput(NamedTuple):
    data: Any
    params: Optional[dict]
    is_unified_llm_input: bool


def _parse_json_data(data, metadata, input_schema):
    json_input = _decode_json_input(data)
    _is_unified_llm_input = is_unified_llm_input(json_input)
    # no data parsing for unified LLM input format
    if _is_unified_llm_input:
        # Unified LLM input format
        if hasattr(metadata, "get_params_schema"):
            params_schema = metadata.get_params_schema()
        else:
            params_schema = None
        data, params = _split_data_and_params_for_llm_input(json_input, params_schema)
    else:
        # Traditional json input format
        data, params = _split_data_and_params(data)
        # data only needs to be parsed if the model signature is not from type hint
        # default to True for backwards compatibility
        should_parse_data = (
            not metadata._is_signature_from_type_hint()
            if hasattr(metadata, "_is_signature_from_type_hint")
            else True
        )
        if should_parse_data:
            data = infer_and_parse_data(data, input_schema)
        else:
            if INPUTS not in data:
                raise MlflowException.invalid_parameter_value(
                    "Request payload must be a dictionary with 'inputs' key when "
                    f"the model contains a valid type hint. Found keys in payload: {data.keys()}."
                )
            data = data[INPUTS]
    return ParsedJsonInput(data, params, _is_unified_llm_input)


def _async_catch_mlflow_exception(func):
    from fastapi.responses import Response

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except MlflowException as e:
            return Response(
                content=e.serialize_as_json(),
                status_code=e.get_http_status_code(),
                media_type="application/json",
            )

    return wrapper


def init(model: PyFuncModel):
    """
    Initialize the server. Loads pyfunc model from the path.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    app = FastAPI()
    input_schema = model.metadata.get_input_schema()
    # set the environment variable to indicate that we are in a serving environment
    os.environ[_MLFLOW_IS_IN_SERVING_ENVIRONMENT.name] = "true"
    timeout = MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT.get()

    @app.middleware("http")
    async def timeout_middleware(request: Request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return Response(
                content="Request processing time exceeded limit",
                status_code=504,
                media_type="application/json",
            )

    @app.route("/ping", methods=["GET"])
    @app.route("/health", methods=["GET"])
    async def ping(request: Request):
        """
        Determine if the container is working and healthy.
        We declare it healthy if we can load the model successfully.
        """
        health = model is not None
        status = 200 if health else 404
        return Response(content="\n", status_code=status, media_type="application/json")

    @app.route("/version", methods=["GET"])
    async def version(request: Request):
        """
        Returns the current mlflow version.
        """
        return Response(content=VERSION, status_code=200, media_type="application/json")

    @app.route("/invocations", methods=["POST"])
    @_async_catch_mlflow_exception
    async def transformation(request: Request):
        """
        Do an inference on a single batch of data. In this sample server,
        we take data as CSV or json, convert it to a Pandas DataFrame or Numpy,
        generate predictions and convert them back to json.
        """

        data = await request.body()
        content_type = request.headers.get("content-type")
        # TODO: convert "invocations" to an async method to make internal logic fully non-blocking.
        result = await asyncio.to_thread(invocations, data, content_type, model, input_schema)

        return Response(
            content=result.response, status_code=result.status, media_type=result.mimetype
        )

    return app


def _predict(model_uri, input_path, output_path, content_type):
    from mlflow.pyfunc.utils.environment import _simulate_serving_environment

    with _simulate_serving_environment():
        pyfunc_model = load_model(model_uri)

        should_parse_as_unified_llm_input = False
        if content_type == "json":
            if input_path is None:
                input_str = sys.stdin.read()
            else:
                with open(input_path) as f:
                    input_str = f.read()
            parsed_json_input = _parse_json_data(
                data=input_str,
                metadata=pyfunc_model.metadata,
                input_schema=pyfunc_model.metadata.get_input_schema(),
            )
            df = parsed_json_input.data
            params = parsed_json_input.params
            should_parse_as_unified_llm_input = parsed_json_input.is_unified_llm_input
        elif content_type == "csv":
            df = (
                parse_csv_input(input_path)
                if input_path is not None
                else parse_csv_input(sys.stdin)
            )
            params = None
        else:
            raise Exception(f"Unknown content type '{content_type}'")

        if "params" in inspect.signature(pyfunc_model.predict).parameters:
            raw_predictions = pyfunc_model.predict(df, params=params)
        else:
            _log_warning_if_params_not_in_predict_signature(_logger, params)
            raw_predictions = pyfunc_model.predict(df)

        parse_output_func = (
            unwrapped_predictions_to_json
            if should_parse_as_unified_llm_input
            else predictions_to_json
        )

        if output_path is None:
            parse_output_func(raw_predictions, sys.stdout)
        else:
            with open(output_path, "w") as fout:
                parse_output_func(raw_predictions, fout)


def _serve(model_uri, port, host):
    pyfunc_model = load_model(model_uri)
    init(pyfunc_model).run(port=port, host=host)


def get_cmd(
    model_uri: str,
    port: Optional[int] = None,
    host: Optional[int] = None,
    timeout: Optional[int] = None,
    nworkers: Optional[int] = None,
) -> tuple[str, dict[str, str]]:
    local_uri = path_to_local_file_uri(model_uri)
    timeout = timeout or MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT.get()

    args = []
    if host:
        args.append(f"--host {shlex.quote(host)}")

    if port:
        args.append(f"--port {port}")

    if nworkers:
        args.append(f"--workers {nworkers}")

    command = f"uvicorn {' '.join(args)} mlflow.pyfunc.scoring_server.app:app"

    command_env = os.environ.copy()
    command_env[_SERVER_MODEL_PATH] = local_uri
    command_env[MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT.name] = str(timeout)

    return command, command_env
