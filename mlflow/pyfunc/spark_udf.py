import collections
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import functools
from typing import Union, Iterator, Tuple

import numpy as np
import pandas

import mlflow
from mlflow.environment_variables import MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils import find_free_port, check_port_connectivity
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.file_utils import get_or_create_tmp_dir, get_or_create_nfs_tmp_dir
from mlflow.utils.nfs_on_spark import get_nfs_cache_root_dir
from mlflow.utils.requirements_utils import _warn_dependency_requirement_mismatches
from mlflow.environment_variables import _MLFLOW_TESTING


_logger = logging.getLogger(__name__)


def _create_model_downloading_tmp_dir(should_use_nfs):
    if should_use_nfs:
        root_tmp_dir = get_or_create_nfs_tmp_dir()
    else:
        root_tmp_dir = get_or_create_tmp_dir()

    root_model_cache_dir = os.path.join(root_tmp_dir, "models")
    os.makedirs(root_model_cache_dir, exist_ok=True)

    tmp_model_dir = tempfile.mkdtemp(dir=root_model_cache_dir)
    # mkdtemp creates a directory with permission 0o700
    # change it to be 0o777 to ensure it can be seen in spark UDF
    os.chmod(tmp_model_dir, 0o777)
    return tmp_model_dir


_MLFLOW_SERVER_OUTPUT_TAIL_LINES_TO_KEEP = 200


def _parse_spark_datatype(datatype: str):
    from pyspark.sql.functions import udf

    return udf(lambda x: x, returnType=datatype).returnType


def spark_udf(spark, model_uri, result_type="double", env_manager=_EnvManager.LOCAL):
    """
    A Spark UDF that can be used to invoke the Python function formatted model.

    Parameters passed to the UDF are forwarded to the model as a DataFrame where the column names
    are ordinals (0, 1, ...). On some versions of Spark (3.0 and above), it is also possible to
    wrap the input in a struct. In that case, the data will be passed as a DataFrame with column
    names given by the struct definition (e.g. when invoked as my_udf(struct('x', 'y')), the model
    will get the data as a pandas DataFrame with 2 columns 'x' and 'y').

    If a model contains a signature with tensor spec inputs, you will need to pass a column of
    array type as a corresponding UDF argument. The column values of which must be one dimensional
    arrays. The UDF will reshape the column values to the required shape with 'C' order
    (i.e. read / write the elements using C-like index order) and cast the values as the required
    tensor spec type.

    If a model contains a signature, the UDF can be called without specifying column name
    arguments. In this case, the UDF will be called with column names from signature, so the
    evaluation dataframe's column names must match the model signature's column names.

    The predictions are filtered to contain only the columns that can be represented as the
    ``result_type``. If the ``result_type`` is string or array of strings, all predictions are
    converted to string. If the result type is not an array type, the left most column with
    matching type is returned.

    NOTE: Inputs of type ``pyspark.sql.types.DateType`` are not supported on earlier versions of
    Spark (2.4 and below).

    .. code-block:: python
        :caption: Example

        from pyspark.sql.functions import struct

        predict = mlflow.pyfunc.spark_udf(spark, "/my/local/model")
        df.withColumn("prediction", predict(struct("name", "age"))).show()

    :param spark: A SparkSession object.
    :param model_uri: The location, in URI format, of the MLflow model with the
                      :py:mod:`mlflow.pyfunc` flavor. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``
                      - ``mlflow-artifacts:/path/to/model``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.

    :param result_type: the return type of the user-defined function. The value can be either a
        ``pyspark.sql.types.DataType`` object or a DDL-formatted type string. Only a primitive
        type, an array ``pyspark.sql.types.ArrayType`` of primitive type, or a struct type
        containing fields of above 2 kinds of types are allowed.
        The following classes of result type are supported:

        - "int" or ``pyspark.sql.types.IntegerType``: The leftmost integer that can fit in an
          ``int32`` or an exception if there is none.

        - "long" or ``pyspark.sql.types.LongType``: The leftmost long integer that can fit in an
          ``int64`` or an exception if there is none.

        - ``ArrayType(IntegerType|LongType)``: All integer columns that can fit into the requested
          size.

        - "float" or ``pyspark.sql.types.FloatType``: The leftmost numeric result cast to
          ``float32`` or an exception if there is none.

        - "double" or ``pyspark.sql.types.DoubleType``: The leftmost numeric result cast to
          ``double`` or an exception if there is none.

        - ``ArrayType(FloatType|DoubleType)``: All numeric columns cast to the requested type or
          an exception if there are no numeric columns.

        - "string" or ``pyspark.sql.types.StringType``: The leftmost column converted to ``string``.

        - "boolean" or "bool" or ``pyspark.sql.types.BooleanType``: The leftmost column converted
          to ``bool`` or an exception if there is none.

        - ``ArrayType(StringType)``: All columns converted to ``string``.

        - "field1 FIELD1_TYPE, field2 FIELD2_TYPE, ...": A struct type containing multiple fields
          separated by comma, each field type must be one of types listed above.

    :param env_manager: The environment manager to use in order to create the python environment
                        for model inference. Note that environment is only restored in the context
                        of the PySpark UDF; the software environment outside of the UDF is
                        unaffected. Default value is ``local``, and the following values are
                        supported:

                         - ``virtualenv``: Use virtualenv to restore the python environment that
                           was used to train the model.
                         - ``conda``: (Recommended) Use Conda to restore the software environment
                           that was used to train the model.
                         - ``local``: Use the current Python environment for model inference, which
                           may differ from the environment used to train the model and may lead to
                           errors or invalid predictions.

    :return: Spark UDF that applies the model's ``predict`` method to the data and returns a
             type specified by ``result_type``, which by default is a double.
    """

    # Scope Spark import to this method so users don't need pyspark to use non-Spark-related
    # functionality.
    from mlflow.pyfunc.spark_model_cache import SparkModelCache
    from mlflow.utils._spark_utils import _SparkDirectoryDistributor
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import (
        ArrayType,
        DataType as SparkDataType,
        StructType as SparkStructType,
    )
    from pyspark.sql.types import (
        DoubleType,
        IntegerType,
        FloatType,
        LongType,
        StringType,
        BooleanType,
    )

    # Used in test to force install local version of mlflow when starting a model server
    mlflow_home = os.environ.get("MLFLOW_HOME")
    openai_env_vars = mlflow.openai._OpenAIEnvVar.read_environ()
    mlflow_testing = _MLFLOW_TESTING.get_raw()

    _EnvManager.validate(env_manager)

    # Check whether spark is in local or local-cluster mode
    # this case all executors and driver share the same filesystem
    is_spark_in_local_mode = spark.conf.get("spark.master").startswith("local")

    nfs_root_dir = get_nfs_cache_root_dir()
    should_use_nfs = nfs_root_dir is not None
    should_use_spark_to_broadcast_file = not (is_spark_in_local_mode or should_use_nfs)

    result_type = "boolean" if result_type == "bool" else result_type

    if not isinstance(result_type, SparkDataType):
        result_type = _parse_spark_datatype(result_type)

    elem_type = result_type
    if isinstance(elem_type, ArrayType):
        elem_type = elem_type.elementType

    supported_types = [
        IntegerType,
        LongType,
        FloatType,
        DoubleType,
        StringType,
        BooleanType,
        SparkStructType,
    ]

    if not any(isinstance(elem_type, x) for x in supported_types):
        raise MlflowException(
            message="Invalid result_type '{}'. Result type can only be one of or an array of one "
            "of the following types: {}".format(str(elem_type), str(supported_types)),
            error_code=INVALID_PARAMETER_VALUE,
        )

    local_model_path = _download_artifact_from_uri(
        artifact_uri=model_uri,
        output_path=_create_model_downloading_tmp_dir(should_use_nfs),
    )

    if env_manager == _EnvManager.LOCAL:
        # Assume spark executor python environment is the same with spark driver side.
        _warn_dependency_requirement_mismatches(local_model_path)
        _logger.warning(
            'Calling `spark_udf()` with `env_manager="local"` does not recreate the same '
            "environment that was used during training, which may lead to errors or inaccurate "
            'predictions. We recommend specifying `env_manager="conda"`, which automatically '
            "recreates the environment that was used to train the model and performs inference "
            "in the recreated environment."
        )
    else:
        _logger.info(
            f"This UDF will use {env_manager} to recreate the model's software environment for "
            "inference. This may take extra time during execution."
        )
        if not sys.platform.startswith("linux"):
            # TODO: support killing mlflow server launched in UDF task when spark job canceled
            #  for non-linux system.
            #  https://stackoverflow.com/questions/53208/how-do-i-automatically-destroy-child-processes-in-windows
            _logger.warning(
                "In order to run inference code in restored python environment, PySpark UDF "
                "processes spawn MLflow Model servers as child processes. Due to system "
                "limitations with handling SIGKILL signals, these MLflow Model server child "
                "processes cannot be cleaned up if the Spark Job is canceled."
            )
    pyfunc_backend = get_flavor_backend(
        local_model_path,
        env_manager=env_manager,
        install_mlflow=os.environ.get("MLFLOW_HOME") is not None,
        create_env_root_dir=True,
    )
    if not should_use_spark_to_broadcast_file:
        # Prepare restored environment in driver side if possible.
        # Note: In databricks runtime, because databricks notebook cell output cannot capture
        # child process output, so that set capture_output to be True so that when `conda prepare
        # env` command failed, the exception message will include command stdout/stderr output.
        # Otherwise user have to check cluster driver log to find command stdout/stderr output.
        # In non-databricks runtime, set capture_output to be False, because the benefit of
        # "capture_output=False" is the output will be printed immediately, otherwise you have
        # to wait conda command fail and suddenly get all output printed (included in error
        # message).
        if env_manager != _EnvManager.LOCAL:
            pyfunc_backend.prepare_env(
                model_uri=local_model_path, capture_output=is_in_databricks_runtime()
            )
    else:
        # Broadcast local model directory to remote worker if needed.
        archive_path = SparkModelCache.add_local_model(spark, local_model_path)

    model_metadata = Model.load(os.path.join(local_model_path, MLMODEL_FILE_NAME))

    def _predict_row_batch(predict_fn, args):
        input_schema = model_metadata.get_input_schema()
        pdf = None

        for x in args:
            if type(x) == pandas.DataFrame:
                if len(args) != 1:
                    raise Exception(
                        "If passing a StructType column, there should be only one "
                        "input column, but got %d" % len(args)
                    )
                pdf = x
        if pdf is None:
            args = list(args)
            if input_schema is None:
                names = [str(i) for i in range(len(args))]
            else:
                names = input_schema.input_names()
                required_names = input_schema.required_input_names()
                if len(args) > len(names):
                    args = args[: len(names)]
                if len(args) < len(required_names):
                    raise MlflowException(
                        "Model input is missing required columns. Expected {} required"
                        " input columns {}, but the model received only {} unnamed input columns"
                        " (Since the columns were passed unnamed they are expected to be in"
                        " the order specified by the schema).".format(len(names), names, len(args))
                    )
            pdf = pandas.DataFrame(data={names[i]: x for i, x in enumerate(args)}, columns=names)

        result = predict_fn(pdf)

        if isinstance(result, dict):
            result = {k: list(v) for k, v in result.items()}

        if not isinstance(result, pandas.DataFrame):
            result = pandas.DataFrame(data=result)

        spark_primitive_type_to_np_type = {
            IntegerType: np.int32,
            LongType: np.int64,
            FloatType: np.float32,
            DoubleType: np.float64,
            BooleanType: np.bool_,
            StringType: np.str_,
        }

        if isinstance(result_type, SparkStructType):
            result_dict = {}
            for field_name in result_type.fieldNames():
                field_type = result_type[field_name].dataType
                field_values = result[field_name]

                if type(field_type) in spark_primitive_type_to_np_type:
                    np_type = spark_primitive_type_to_np_type[type(field_type)]
                    field_values = field_values.astype(np_type)

                elif type(field_type) == ArrayType:
                    elem_type = field_type.elementType
                    if type(elem_type) not in spark_primitive_type_to_np_type:
                        raise MlflowException(
                            "Unsupported array type field with element type "
                            f"{elem_type.simpleString()} in struct type.",
                            error_code=INVALID_PARAMETER_VALUE,
                        )
                    np_type = spark_primitive_type_to_np_type[type(elem_type)]

                    field_values = [
                        np.array(v, dtype=np.dtype(type(elem_type))) for v in field_values
                    ]

                else:
                    raise MlflowException(
                        f"Unsupported field type {field_type.simpleString()} in struct type.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )

                result_dict[field_name] = field_values

            return pandas.DataFrame(result_dict)

        elem_type = result_type.elementType if isinstance(result_type, ArrayType) else result_type

        if type(elem_type) == IntegerType:
            result = result.select_dtypes(
                [np.byte, np.ubyte, np.short, np.ushort, np.int32]
            ).astype(np.int32)

        elif type(elem_type) == LongType:
            result = result.select_dtypes([np.byte, np.ubyte, np.short, np.ushort, int]).astype(
                np.int64
            )

        elif type(elem_type) == FloatType:
            result = result.select_dtypes(include=(np.number,)).astype(np.float32)

        elif type(elem_type) == DoubleType:
            result = result.select_dtypes(include=(np.number,)).astype(np.float64)

        elif type(elem_type) == BooleanType:
            result = result.select_dtypes([bool, np.bool_]).astype(bool)

        if len(result.columns) == 0:
            raise MlflowException(
                message="The model did not produce any values compatible with the requested "
                "type '{}'. Consider requesting udf with StringType or "
                "Arraytype(StringType).".format(str(elem_type)),
                error_code=INVALID_PARAMETER_VALUE,
            )

        if type(elem_type) == StringType:
            result = result.applymap(str)

        if type(result_type) == ArrayType:
            return pandas.Series(result.to_numpy().tolist())
        else:
            return result[result.columns[0]]

    result_type_hint = (
        pandas.DataFrame if isinstance(result_type, SparkStructType) else pandas.Series
    )

    @pandas_udf(result_type)
    def udf(
        iterator: Iterator[Tuple[Union[pandas.Series, pandas.DataFrame], ...]]
    ) -> Iterator[result_type_hint]:
        # importing here to prevent circular import
        from mlflow.pyfunc.scoring_server.client import (
            ScoringServerClient,
            StdinScoringServerClient,
        )

        # Note: this is a pandas udf function in iteration style, which takes an iterator of
        # tuple of pandas.Series and outputs an iterator of pandas.Series.
        if mlflow_home is not None:
            os.environ["MLFLOW_HOME"] = mlflow_home
        if openai_env_vars:
            os.environ.update(openai_env_vars)
        if mlflow_testing:
            _MLFLOW_TESTING.set(mlflow_testing)
        scoring_server_proc = None

        if env_manager != _EnvManager.LOCAL:
            if should_use_spark_to_broadcast_file:
                local_model_path_on_executor = _SparkDirectoryDistributor.get_or_extract(
                    archive_path
                )
                # Call "prepare_env" in advance in order to reduce scoring server launch time.
                # So that we can use a shorter timeout when call `client.wait_server_ready`,
                # otherwise we have to set a long timeout for `client.wait_server_ready` time,
                # this prevents spark UDF task failing fast if other exception raised when scoring
                # server launching.
                # Set "capture_output" so that if "conda env create" command failed, the command
                # stdout/stderr output will be attached to the exception message and included in
                # driver side exception.
                pyfunc_backend.prepare_env(
                    model_uri=local_model_path_on_executor, capture_output=True
                )
            else:
                local_model_path_on_executor = None

            if check_port_connectivity():
                # launch scoring server
                server_port = find_free_port()
                host = "127.0.0.1"
                scoring_server_proc = pyfunc_backend.serve(
                    model_uri=local_model_path_on_executor or local_model_path,
                    port=server_port,
                    host=host,
                    timeout=MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT.get(),
                    enable_mlserver=False,
                    synchronous=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )

                client = ScoringServerClient(host, server_port)
            else:
                scoring_server_proc = pyfunc_backend.serve_stdin(
                    model_uri=local_model_path_on_executor or local_model_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                client = StdinScoringServerClient(scoring_server_proc)

            _logger.info("Using %s", client.__class__.__name__)

            server_tail_logs = collections.deque(maxlen=_MLFLOW_SERVER_OUTPUT_TAIL_LINES_TO_KEEP)

            def server_redirect_log_thread_func(child_stdout):
                for line in child_stdout:
                    decoded = line.decode() if isinstance(line, bytes) else line
                    server_tail_logs.append(decoded)
                    sys.stdout.write("[model server] " + decoded)

            server_redirect_log_thread = threading.Thread(
                target=server_redirect_log_thread_func,
                args=(scoring_server_proc.stdout,),
            )
            server_redirect_log_thread.setDaemon(True)
            server_redirect_log_thread.start()

            try:
                client.wait_server_ready(timeout=90, scoring_server_proc=scoring_server_proc)
            except Exception as e:
                err_msg = "During spark UDF task execution, mlflow model server failed to launch. "
                if len(server_tail_logs) == _MLFLOW_SERVER_OUTPUT_TAIL_LINES_TO_KEEP:
                    err_msg += (
                        f"Last {_MLFLOW_SERVER_OUTPUT_TAIL_LINES_TO_KEEP} "
                        "lines of MLflow model server output:\n"
                    )
                else:
                    err_msg += "MLflow model server output:\n"
                err_msg += "".join(server_tail_logs)
                raise MlflowException(err_msg) from e

            def batch_predict_fn(pdf):
                return client.invoke(pdf).get_predictions()

        elif env_manager == _EnvManager.LOCAL:
            if should_use_spark_to_broadcast_file:
                loaded_model, _ = SparkModelCache.get_or_load(archive_path)
            else:
                loaded_model = mlflow.pyfunc.load_model(local_model_path)

            def batch_predict_fn(pdf):
                return loaded_model.predict(pdf)

        try:
            for input_batch in iterator:
                # If the UDF is called with only multiple arguments,
                # the `input_batch` is a tuple which composes of several pd.Series/pd.DataFrame
                # objects.
                # If the UDF is called with only one argument,
                # the `input_batch` instance will be an instance of `pd.Series`/`pd.DataFrame`,
                if isinstance(input_batch, (pandas.Series, pandas.DataFrame)):
                    # UDF is called with only one argument
                    row_batch_args = (input_batch,)
                else:
                    row_batch_args = input_batch

                if len(row_batch_args[0]) > 0:
                    yield _predict_row_batch(batch_predict_fn, row_batch_args)
        finally:
            if scoring_server_proc is not None:
                os.kill(scoring_server_proc.pid, signal.SIGTERM)

    udf.metadata = model_metadata

    @functools.wraps(udf)
    def udf_with_default_cols(*args):
        if len(args) == 0:
            input_schema = model_metadata.get_input_schema()
            if input_schema and len(input_schema.optional_input_names()) > 0:
                raise MlflowException(
                    message="Cannot apply UDF without column names specified when"
                    " model signature contains optional columns.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            if input_schema and len(input_schema.inputs) > 0:
                if input_schema.has_input_names():
                    input_names = input_schema.input_names()
                    return udf(*input_names)
                else:
                    raise MlflowException(
                        message="Cannot apply udf because no column names specified. The udf "
                        "expects {} columns with types: {}. Input column names could not be "
                        "inferred from the model signature (column names not found).".format(
                            len(input_schema.inputs),
                            input_schema.inputs,
                        ),
                        error_code=INVALID_PARAMETER_VALUE,
                    )
            else:
                raise MlflowException(
                    "Attempting to apply udf on zero columns because no column names were "
                    "specified as arguments or inferred from the model signature.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            return udf(*args)

    return udf_with_default_cols
