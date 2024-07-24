import json
import logging
import math
import struct
import sys

from packaging.version import Version

import mlflow
from mlflow.entities import RunTag
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils import insecure_hash
from mlflow.utils.string_utils import generate_feature_name_if_not_string

try:
    # `numpy` and `pandas` are not required for `mlflow-skinny`.
    import numpy as np
    import pandas as pd
except ImportError:
    pass

_logger = logging.getLogger(__name__)


def _hash_uint64_ndarray_as_bytes(array):
    assert len(array.shape) == 1
    # see struct pack format string https://docs.python.org/3/library/struct.html#format-strings
    return struct.pack(f">{array.size}Q", *array)


def _is_empty_list_or_array(data):
    if isinstance(data, list):
        return len(data) == 0
    elif isinstance(data, np.ndarray):
        return data.size == 0
    return False


def _is_array_has_dict(nd_array):
    if _is_empty_list_or_array(nd_array):
        return False

    # It is less likely the array or list contains heterogeneous elements, so just checking the
    # first element to avoid performance overhead.
    elm = nd_array.item(0)
    if isinstance(elm, (list, np.ndarray)):
        return _is_array_has_dict(elm)
    elif isinstance(elm, dict):
        return True

    return False


def _hash_array_of_dict_as_bytes(data):
    # NB: If an array or list contains dictionary element, it can't be hashed with
    # pandas.util.hash_array. Hence we need to manually hash the elements here. This is
    # particularly for the LLM use case where the input can be a list of dictionary
    # (chat/completion payloads), so doesn't handle more complex case like nested lists.
    result = b""
    for elm in data:
        if isinstance(elm, (list, np.ndarray)):
            result += _hash_array_of_dict_as_bytes(elm)
        elif isinstance(elm, dict):
            result += _hash_dict_as_bytes(elm)
        else:
            result += _hash_data_as_bytes(elm)
    return result


def _hash_ndarray_as_bytes(nd_array):
    if not isinstance(nd_array, np.ndarray):
        nd_array = np.array(nd_array)

    if _is_array_has_dict(nd_array):
        return _hash_array_of_dict_as_bytes(nd_array)

    return _hash_uint64_ndarray_as_bytes(
        pd.util.hash_array(nd_array.flatten(order="C"))
    ) + _hash_uint64_ndarray_as_bytes(np.array(nd_array.shape, dtype="uint64"))


def _hash_data_as_bytes(data):
    try:
        if isinstance(data, (list, np.ndarray)):
            return _hash_ndarray_as_bytes(data)
        if isinstance(data, dict):
            return _hash_dict_as_bytes(data)
        if np.isscalar(data):
            return _hash_uint64_ndarray_as_bytes(pd.util.hash_array(np.array([data])))
    finally:
        return b""  # Skip unsupported types by returning an empty byte string


def _hash_dict_as_bytes(data_dict):
    result = _hash_ndarray_as_bytes(list(data_dict.keys()))
    try:
        result += _hash_ndarray_as_bytes(list(data_dict.values()))
    # If the values containing non-hashable objects, we will hash the values recursively.
    except Exception:
        for value in data_dict.values():
            result += _hash_data_as_bytes(value)
    return result


def _hash_array_like_obj_as_bytes(data):
    """
    Helper method to convert pandas dataframe/numpy array/list into bytes for
    MD5 calculation purpose.
    """
    if isinstance(data, pd.DataFrame):
        # add checking `'pyspark' in sys.modules` to avoid importing pyspark when user
        # run code not related to pyspark.
        if "pyspark" in sys.modules:
            from pyspark.ml.linalg import Vector as spark_vector_type
        else:
            spark_vector_type = None

        def _hash_array_like_element_as_bytes(v):
            if spark_vector_type is not None:
                if isinstance(v, spark_vector_type):
                    return _hash_ndarray_as_bytes(v.toArray())
            if isinstance(v, (dict, list, np.ndarray)):
                return _hash_data_as_bytes(v)

            try:
                # Attempt to hash the value, if it fails, return an empty byte string
                pd.util.hash_array(np.array([v]))
                return v
            except TypeError:
                return b""  # Skip unhashable types by returning an empty byte string

        if Version(pd.__version__) >= Version("2.1.0"):
            data = data.map(_hash_array_like_element_as_bytes)
        else:
            data = data.applymap(_hash_array_like_element_as_bytes)
        return _hash_uint64_ndarray_as_bytes(pd.util.hash_pandas_object(data))
    elif isinstance(data, np.ndarray) and len(data) > 0 and isinstance(data[0], list):
        # convert numpy array of lists into numpy array of the string representation of the lists
        # because lists are not hashable
        hashable = np.array(str(val) for val in data)
        return _hash_ndarray_as_bytes(hashable)
    elif isinstance(data, np.ndarray) and len(data) > 0 and isinstance(data[0], np.ndarray):
        # convert numpy array of numpy arrays into 2d numpy arrays
        # because numpy array of numpy arrays are not hashable
        hashable = np.array(data.tolist())
        return _hash_ndarray_as_bytes(hashable)
    elif isinstance(data, np.ndarray):
        return _hash_ndarray_as_bytes(data)
    elif isinstance(data, list):
        return _hash_ndarray_as_bytes(np.array(data))
    else:
        raise ValueError("Unsupported data type.")


def _gen_md5_for_arraylike_obj(md5_gen, data):
    """
    Helper method to generate MD5 hash array-like object, the MD5 will calculate over:
     - array length
     - first NUM_SAMPLE_ROWS_FOR_HASH rows content
     - last NUM_SAMPLE_ROWS_FOR_HASH rows content
    """
    len_bytes = _hash_uint64_ndarray_as_bytes(np.array([len(data)], dtype="uint64"))
    md5_gen.update(len_bytes)
    if len(data) < EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH * 2:
        md5_gen.update(_hash_array_like_obj_as_bytes(data))
    else:
        if isinstance(data, pd.DataFrame):
            # Access rows of pandas Df with iloc
            head_rows = data.iloc[: EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH]
            tail_rows = data.iloc[-EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH :]
        else:
            head_rows = data[: EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH]
            tail_rows = data[-EvaluationDataset.NUM_SAMPLE_ROWS_FOR_HASH :]
        md5_gen.update(_hash_array_like_obj_as_bytes(head_rows))
        md5_gen.update(_hash_array_like_obj_as_bytes(tail_rows))


def convert_data_to_mlflow_dataset(data, targets=None, predictions=None):
    """Convert input data to mlflow dataset."""
    supported_dataframe_types = [pd.DataFrame]
    if "pyspark" in sys.modules:
        from pyspark.sql import DataFrame as SparkDataFrame

        supported_dataframe_types.append(SparkDataFrame)

    if predictions is not None:
        _validate_dataset_type_supports_predictions(
            data=data, supported_predictions_dataset_types=supported_dataframe_types
        )

    if isinstance(data, list):
        # If the list is flat, we assume each element is an independent sample.
        if not isinstance(data[0], (list, np.ndarray)):
            data = [[elm] for elm in data]

        return mlflow.data.from_numpy(
            np.array(data), targets=np.array(targets) if targets else None
        )
    elif isinstance(data, np.ndarray):
        return mlflow.data.from_numpy(data, targets=targets)
    elif isinstance(data, pd.DataFrame):
        return mlflow.data.from_pandas(df=data, targets=targets, predictions=predictions)
    elif "pyspark" in sys.modules and isinstance(data, SparkDataFrame):
        return mlflow.data.from_spark(df=data, targets=targets, predictions=predictions)
    else:
        # Cannot convert to mlflow dataset, return original data.
        _logger.info(
            "Cannot convert input data to `evaluate()` to an mlflow dataset, input must be a list, "
            f"a numpy array, a panda Dataframe or a spark Dataframe, but received {type(data)}."
        )
        return data


def _validate_dataset_type_supports_predictions(data, supported_predictions_dataset_types):
    """
    Validate that the dataset type supports a user-specified "predictions" column.
    """
    if not any(isinstance(data, sdt) for sdt in supported_predictions_dataset_types):
        raise MlflowException(
            message=(
                "If predictions is specified, data must be one of the following types, or an"
                " MLflow Dataset that represents one of the following types:"
                f" {supported_predictions_dataset_types}."
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )


class EvaluationDataset:
    """
    An input dataset for model evaluation. This is intended for use with the
    :py:func:`mlflow.models.evaluate()`
    API.
    """

    NUM_SAMPLE_ROWS_FOR_HASH = 5
    SPARK_DATAFRAME_LIMIT = 10000

    def __init__(
        self,
        data,
        *,
        targets=None,
        name=None,
        path=None,
        feature_names=None,
        predictions=None,
    ):
        """
        The values of the constructor arguments comes from the `evaluate` call.
        """
        if name is not None and '"' in name:
            raise MlflowException(
                message=f'Dataset name cannot include a double quote (") but got {name}',
                error_code=INVALID_PARAMETER_VALUE,
            )
        if path is not None and '"' in path:
            raise MlflowException(
                message=f'Dataset path cannot include a double quote (") but got {path}',
                error_code=INVALID_PARAMETER_VALUE,
            )

        self._user_specified_name = name
        self._path = path
        self._hash = None
        self._supported_dataframe_types = (pd.DataFrame,)
        self._spark_df_type = None
        self._labels_data = None
        self._targets_name = None
        self._has_targets = False
        self._predictions_data = None
        self._predictions_name = None
        self._has_predictions = predictions is not None

        try:
            # add checking `'pyspark' in sys.modules` to avoid importing pyspark when user
            # run code not related to pyspark.
            if "pyspark" in sys.modules:
                from pyspark.sql import DataFrame as SparkDataFrame

                self._supported_dataframe_types = (pd.DataFrame, SparkDataFrame)
                self._spark_df_type = SparkDataFrame
        except ImportError:
            pass

        if feature_names is not None and len(set(feature_names)) < len(list(feature_names)):
            raise MlflowException(
                message="`feature_names` argument must be a list containing unique feature names.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if self._has_predictions:
            _validate_dataset_type_supports_predictions(
                data=data,
                supported_predictions_dataset_types=self._supported_dataframe_types,
            )

        has_targets = targets is not None
        if has_targets:
            self._has_targets = True
        if isinstance(data, (np.ndarray, list)):
            if has_targets and not isinstance(targets, (np.ndarray, list)):
                raise MlflowException(
                    message="If data is a numpy array or list of evaluation features, "
                    "`targets` argument must be a numpy array or list of evaluation labels.",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            shape_message = (
                "If the `data` argument is a numpy array, it must be a 2-dimensional "
                "array, with the second dimension representing the number of features. If the "
                "`data` argument is a list, each of its elements must be a feature array of "
                "the numpy array or list, and all elements must have the same length."
            )

            if isinstance(data, list):
                try:
                    data = np.array(data)
                except ValueError as e:
                    raise MlflowException(
                        message=shape_message, error_code=INVALID_PARAMETER_VALUE
                    ) from e

            if len(data.shape) != 2:
                raise MlflowException(
                    message=shape_message,
                    error_code=INVALID_PARAMETER_VALUE,
                )

            self._features_data = data
            if has_targets:
                self._labels_data = (
                    targets if isinstance(targets, np.ndarray) else np.array(targets)
                )

                if len(self._features_data) != len(self._labels_data):
                    raise MlflowException(
                        message="The input features example rows must be the same length "
                        "with labels array.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )

            num_features = data.shape[1]

            if feature_names is not None:
                feature_names = list(feature_names)
                if num_features != len(feature_names):
                    raise MlflowException(
                        message="feature name list must be the same length with feature data.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                self._feature_names = feature_names
            else:
                self._feature_names = [
                    f"feature_{str(i + 1).zfill(math.ceil(math.log10(num_features + 1)))}"
                    for i in range(num_features)
                ]
        elif isinstance(data, self._supported_dataframe_types):
            if has_targets and not isinstance(targets, str):
                raise MlflowException(
                    message="If data is a Pandas DataFrame or Spark DataFrame, `targets` argument "
                    "must be the name of the column which contains evaluation labels in the `data` "
                    "dataframe.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            if self._spark_df_type and isinstance(data, self._spark_df_type):
                if data.count() > EvaluationDataset.SPARK_DATAFRAME_LIMIT:
                    _logger.warning(
                        "Specified Spark DataFrame is too large for model evaluation. Only "
                        f"the first {EvaluationDataset.SPARK_DATAFRAME_LIMIT} rows will be used. "
                        "If you want evaluate on the whole spark dataframe, please manually call "
                        "`spark_dataframe.toPandas()`."
                    )
                data = data.limit(EvaluationDataset.SPARK_DATAFRAME_LIMIT).toPandas()

            if has_targets:
                self._labels_data = data[targets].to_numpy()
                self._targets_name = targets

            if self._has_predictions:
                self._predictions_data = data[predictions].to_numpy()
                self._predictions_name = predictions

            if feature_names is not None:
                self._features_data = data[list(feature_names)]
                self._feature_names = feature_names
            else:
                features_data = data

                if has_targets:
                    features_data = features_data.drop(targets, axis=1, inplace=False)

                if self._has_predictions:
                    features_data = features_data.drop(predictions, axis=1, inplace=False)

                self._features_data = features_data
                self._feature_names = [
                    generate_feature_name_if_not_string(c) for c in self._features_data.columns
                ]
        else:
            raise MlflowException(
                message="The data argument must be a numpy array, a list or a Pandas DataFrame, or "
                "spark DataFrame if pyspark package installed.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # generate dataset hash
        md5_gen = insecure_hash.md5()
        _gen_md5_for_arraylike_obj(md5_gen, self._features_data)
        if self._labels_data is not None:
            _gen_md5_for_arraylike_obj(md5_gen, self._labels_data)
        if self._predictions_data is not None:
            _gen_md5_for_arraylike_obj(md5_gen, self._predictions_data)
        md5_gen.update(",".join(list(map(str, self._feature_names))).encode("UTF-8"))

        self._hash = md5_gen.hexdigest()

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def features_data(self):
        """
        return features data as a numpy array or a pandas DataFrame.
        """
        return self._features_data

    @property
    def labels_data(self):
        """
        return labels data as a numpy array
        """
        return self._labels_data

    @property
    def has_targets(self):
        """
        Returns True if the dataset has targets, False otherwise.
        """
        return self._has_targets

    @property
    def targets_name(self):
        """
        return targets name
        """
        return self._targets_name

    @property
    def predictions_data(self):
        """
        return labels data as a numpy array
        """
        return self._predictions_data

    @property
    def has_predictions(self):
        """
        Returns True if the dataset has targets, False otherwise.
        """
        return self._has_predictions

    @property
    def predictions_name(self):
        """
        return predictions name
        """
        return self._predictions_name

    @property
    def name(self):
        """
        Dataset name, which is specified dataset name or the dataset hash if user don't specify
        name.
        """
        return self._user_specified_name if self._user_specified_name is not None else self.hash

    @property
    def path(self):
        """
        Dataset path
        """
        return self._path

    @property
    def hash(self):
        """
        Dataset hash, includes hash on first 20 rows and last 20 rows.
        """
        return self._hash

    @property
    def _metadata(self):
        """
        Return dataset metadata containing name, hash, and optional path.
        """
        metadata = {
            "name": self.name,
            "hash": self.hash,
        }
        if self.path is not None:
            metadata["path"] = self.path
        return metadata

    def _log_dataset_tag(self, client, run_id, model_uuid):
        """
        Log dataset metadata as a tag "mlflow.datasets", if the tag already exists, it will
        append current dataset metadata into existing tag content.
        """
        existing_dataset_metadata_str = client.get_run(run_id).data.tags.get(
            "mlflow.datasets", "[]"
        )
        dataset_metadata_list = json.loads(existing_dataset_metadata_str)

        for metadata in dataset_metadata_list:
            if (
                metadata["hash"] == self.hash
                and metadata["name"] == self.name
                and metadata["model"] == model_uuid
            ):
                break
        else:
            dataset_metadata_list.append({**self._metadata, "model": model_uuid})

        dataset_metadata_str = json.dumps(dataset_metadata_list, separators=(",", ":"))
        client.log_batch(
            run_id,
            tags=[RunTag("mlflow.datasets", dataset_metadata_str)],
        )

    def __hash__(self):
        return hash(self.hash)

    def __eq__(self, other):
        if not isinstance(other, EvaluationDataset):
            return False

        if isinstance(self._features_data, np.ndarray):
            is_features_data_equal = np.array_equal(self._features_data, other._features_data)
        else:
            is_features_data_equal = self._features_data.equals(other._features_data)

        return (
            is_features_data_equal
            and np.array_equal(self._labels_data, other._labels_data)
            and self.name == other.name
            and self.path == other.path
            and self._feature_names == other._feature_names
        )
