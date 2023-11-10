from typing import Any, List

from packaging.version import Version

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils import insecure_hash

MAX_ROWS = 10000


def compute_pandas_digest(df) -> str:
    """
    Computes a digest for the given Pandas DataFrame.

    :param df: A Pandas DataFrame.
    :return: A string digest.
    """
    import numpy as np
    import pandas as pd

    # trim to max rows
    trimmed_df = df.head(MAX_ROWS)

    # keep string and number columns, drop other column types
    string_columns = trimmed_df.columns[(df.applymap(type) == str).all(0)]
    numeric_columns = trimmed_df.select_dtypes(include=[np.number]).columns

    desired_columns = string_columns.union(numeric_columns)
    trimmed_df = trimmed_df[desired_columns]

    return get_normalized_md5_digest(
        [
            pd.util.hash_pandas_object(trimmed_df).values,
            np.int64(len(df)),
        ]
        + [str(x).encode() for x in df.columns]
    )


def compute_numpy_digest(features, targets=None) -> str:
    """
    Computes a digest for the given numpy array.

    :param features: A numpy array containing dataset features.
    :param targets: A numpy array containing dataset targets. Optional.
    :return: A string digest.
    """
    import numpy as np
    import pandas as pd

    hashable_elements = []

    def hash_array(array):
        flattened_array = array.flatten()
        trimmed_array = flattened_array[0:MAX_ROWS]
        try:
            hashable_elements.append(pd.util.hash_array(trimmed_array))
        except TypeError:
            hashable_elements.append(np.int64(trimmed_array.size))

        # hash full array dimensions
        for x in array.shape:
            hashable_elements.append(np.int64(x))

    def hash_dict_of_arrays(array_dict):
        for key in sorted(array_dict.keys()):
            hash_array(array_dict[key])

    for item in [features, targets]:
        if item is None:
            continue
        if isinstance(item, dict):
            hash_dict_of_arrays(item)
        else:
            hash_array(item)

    return get_normalized_md5_digest(hashable_elements)


def compute_tensorflow_dataset_digest(dataset, targets=None) -> str:
    """
    Computes a digest for the given Tensorflow dataset.

    :param dataset: A Tensorflow dataset.
    :return: A string digest.
    """
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    hashable_elements = []

    def hash_tf_dataset_iterator_element(element):
        if element is None:
            return
        flat_element = tf.nest.flatten(element)
        flattened_array = np.concatenate([x.flatten() for x in flat_element])
        trimmed_array = flattened_array[0:MAX_ROWS]
        try:
            hashable_elements.append(pd.util.hash_array(trimmed_array))
        except TypeError:
            hashable_elements.append(np.int64(trimmed_array.size))

    for element in dataset.as_numpy_iterator():
        hash_tf_dataset_iterator_element(element)
    if targets is not None:
        for element in targets.as_numpy_iterator():
            hash_tf_dataset_iterator_element(element)

    return get_normalized_md5_digest(hashable_elements)


def compute_tensor_digest(tensor_data, tensor_targets) -> str:
    """
    Computes a digest for the given Tensorflow tensor.

    :param tensor: A Tensorflow tensor.
    :return: A string digest.
    """
    if tensor_targets is None:
        return compute_numpy_digest(tensor_data.numpy())
    else:
        return compute_numpy_digest(tensor_data.numpy(), tensor_targets.numpy())


def compute_spark_df_digest(df) -> str:
    """
    Computes a digest for the given Spark DataFrame. Retrieve a semantic hash of the
    DataFrame's logical plan, which is much more efficient and deterministic than hashing
    DataFrame records

    :param df: A Spark DataFrame.
    :return: A string digest.
    """

    import numpy as np
    import pyspark

    # Spark 3.1.0+ has a semanticHash() method on DataFrame
    if Version(pyspark.__version__) >= Version("3.1.0"):
        semantic_hash = df.semanticHash()
    else:
        semantic_hash = df._jdf.queryExecution().analyzed().semanticHash()
    return get_normalized_md5_digest([np.int64(semantic_hash)])


def get_normalized_md5_digest(elements: List[Any]) -> str:
    """
    Computes a normalized digest for a list of hashable elements.

    :param elements: A list of hashable elements for inclusion in the md5 digest.
    :return: An 8-character, truncated md5 digest.
    """

    if not elements:
        raise MlflowException(
            "No hashable elements were provided for md5 digest creation",
            INVALID_PARAMETER_VALUE,
        )

    md5 = insecure_hash.md5()
    for element in elements:
        md5.update(element)

    return md5.hexdigest()[:8]
