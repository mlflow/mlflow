from pyspark.sql import SparkSession
from pyspark.sql import types as T

import mlflow


class MyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return [str(" | ".join(map(str, row))) for _, row in model_input.iterrows()]


def main():
    with SparkSession.builder.getOrCreate() as spark:
        df = spark.createDataFrame(
            [
                (
                    "a",
                    [0],
                    {"bool": True},
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

        with mlflow.start_run():
            model_info = mlflow.pyfunc.log_model(
                name="model",
                python_model=MyModel(),
                signature=mlflow.models.infer_signature(df),
            )

        udf = mlflow.pyfunc.spark_udf(
            spark=spark,
            model_uri=model_info.model_uri,
            result_type="string",
        )
        df.withColumn("output", udf("str", "arr", "obj", "obj_arr")).show()


if __name__ == "__main__":
    main()
