import pandas as pd
import random
from typing import Any
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StructField, StructType, LongType, ArrayType, MapType, StringType
from pyspark.sql.pandas.functions import pandas_udf, PandasUDFType

spark = SparkSession.builder.getOrCreate()


@pandas_udf(ArrayType(MapType(StringType(), LongType())))
def array_map(s: pd.Series) -> pd.Series:
    return pd.Series(
        [[{x: idx}] for idx, x in enumerate(s)],
    )


df = spark.createDataFrame([("Foo",), ("Bar",)], ("name",))
df.select(array_map("name")).show()

# 23/03/27 13:19:01 WARN TaskSetManager: Lost task 4.0 in stage 5.0 (TID 25) (192.168.0.177 executor driver): TaskKilled (Stage cancelled)
# Traceback (most recent call last):
#   File "example.py", line 20, in <module>
#   File "/home/haru/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/pyspark/sql/dataframe.py", line 606, in show
#     print(self._jdf.showString(n, 20, vertical))
#   File "/home/haru/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/py4j/java_gateway.py", line 1321, in __call__
#     return_value = get_return_value(
#   File "/home/haru/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/pyspark/sql/utils.py", line 196, in deco
#     raise converted from None
# pyspark.sql.utils.PythonException:
#   An exception was thrown from the Python worker. Please see the stack trace below.
# Traceback (most recent call last):
#   File "pyarrow/array.pxi", line 1044, in pyarrow.lib.Array.from_pandas
#   File "pyarrow/array.pxi", line 316, in pyarrow.lib.array
#   File "pyarrow/array.pxi", line 83, in pyarrow.lib._ndarray_to_array
#   File "pyarrow/error.pxi", line 123, in pyarrow.lib.check_status
# pyarrow.lib.ArrowTypeError: Could not convert {'Bar': 0} with type dict: was not a sequence or recognized null for conversion to list type
#
# Related to https://github.com/apache/arrow/issues/33928?


@pandas_udf(
    ArrayType(
        StructType(
            [
                StructField("a", StringType(), nullable=False),
            ]
        )
    )
)
def array_map(s: pd.Series) -> pd.Series:
    return pd.Series(
        [[{"a": idx}] for idx, x in enumerate(s)],
    )


# Traceback (most recent call last):
#   File "/home/haru/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/pyspark/sql/udf.py", line 141, in returnType
#     to_arrow_type(self._returnType_placeholder)
#   File "/home/haru/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/pyspark/sql/pandas/types.py", line 90, in to_arrow_type
#     raise TypeError("Unsupported type in conversion to Arrow: " + str(dt))
# TypeError: Unsupported type in conversion to Arrow: ArrayType(StructType([StructField('a', StringType(), False)]), True)

# During handling of the above exception, another exception occurred:

# Traceback (most recent call last):
#   File "example.py", line 20, in <module>
#     def array_map(s: pd.Series) -> pd.Series:
#   File "/home/haru/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/pyspark/sql/pandas/functions.py", line 450, in _create_pandas_udf
#     return _create_udf(f, returnType, evalType)
#   File "/home/haru/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/pyspark/sql/udf.py", line 74, in _create_udf
#     return udf_obj._wrapped()
#   File "/home/haru/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/pyspark/sql/udf.py", line 286, in _wrapped
#     wrapper.returnType = self.returnType  # type: ignore[attr-defined]
#   File "/home/haru/miniconda3/envs/mlflow-dev-env/lib/python3.8/site-packages/pyspark/sql/udf.py", line 143, in returnType
#     raise NotImplementedError(
# NotImplementedError: Invalid return type with scalar Pandas UDFs: ArrayType(StructType([StructField('a', StringType(), False)]), True) is not supported
