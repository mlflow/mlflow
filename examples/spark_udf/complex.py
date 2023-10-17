import random
from typing import Iterator, Tuple, Union

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import pandas_udf

import mlflow


class MyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return [random.random()] * len(model_input)


def main():
    with SparkSession.builder.getOrCreate() as spark:
        df = spark.createDataFrame(
            [
                (
                    "a",
                    [0],
                    {"bool": True, "float": 0.0},
                    [{"double": 0.1}],
                )
            ],
            schema=T.StructType(
                [
                    T.StructField(
                        "str",
                        T.StringType(),
                    ),
                    T.StructField(
                        "arr",
                        T.ArrayType(T.IntegerType()),
                    ),
                    T.StructField(
                        "obj",
                        T.StructType(
                            [
                                T.StructField("bool", T.BooleanType()),
                                T.StructField("float", T.FloatType()),
                            ]
                        ),
                    ),
                    T.StructField(
                        "obj_arr",
                        T.ArrayType(
                            T.StructType(
                                [
                                    T.StructField("double", T.DoubleType()),
                                ]
                            )
                        ),
                    ),
                ]
            ),
        )
        df.printSchema()
        df.show()

        @pandas_udf("long")
        def inspect(
            iterator: Iterator[Tuple[Union[pd.Series, pd.DataFrame], ...]]
        ) -> Iterator[pd.Series]:
            for args in iterator:
                for arg in args:
                    print("-" * 10)
                    print(arg)
                    print(type(arg))

                yield pd.Series([random.random()])

        df.withColumn("output", inspect("str", "arr", "obj", "obj_arr")).show()

        with mlflow.start_run():
            model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=MyModel(),
                signature=mlflow.models.infer_signature(df),
            )

        udf = mlflow.pyfunc.spark_udf(
            spark=spark, model_uri=model_info.model_uri, result_type="string"
        )
        df.withColumn("output", udf("str", "arr", "obj", "obj_arr")).show()
        pass


if __name__ == "__main__":
    main()
