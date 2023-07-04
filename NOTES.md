## Set up

```sh
wget https://dlcdn.apache.org/spark/spark-3.4.1/spark-3.4.1-bin-hadoop3.tgz
mkdir spark-bin
tar -xvf spark-3.4.1-bin-hadoop3.tgz --directory spark-bin
rm spark-3.4.1-bin-hadoop3.tgz
./spark-bin/spark-3.4.1-bin-hadoop3/sbin/start-connect-server.sh --packages org.apache.spark:spark-connect_2.12:3.4.0
export SPARK_REMOTE="sc://localhost"
```

```sh
python example.py
```

```
/Users/harutakakawamura/.pyenv/versions/miniconda3-4.7.12/envs/mlflow-dev-env/lib/python3.8/site-packages/_distutils_hack/__init__.py:33: UserWarning: Setuptools is replacing distutils.
  warnings.warn("Setuptools is replacing distutils.")
Traceback (most recent call last):
  File "example.py", line 23, in <module>
    pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_info.model_uri, env_manager="conda")
  File "/Users/harutakakawamura/Desktop/repositories/mlflow/mlflow/pyfunc/__init__.py", line 973, in spark_udf
    result_type = _parse_datatype_string(result_type)
  File "/Users/harutakakawamura/.pyenv/versions/miniconda3-4.7.12/envs/mlflow-dev-env/lib/python3.8/site-packages/pyspark/sql/types.py", line 1212, in _parse_datatype_string
    sc = get_active_spark_context()
  File "/Users/harutakakawamura/.pyenv/versions/miniconda3-4.7.12/envs/mlflow-dev-env/lib/python3.8/site-packages/pyspark/sql/utils.py", line 202, in get_active_spark_context
    raise RuntimeError("SparkContext or SparkSession should be created first.")
RuntimeError: SparkContext or SparkSession should be created first.
```
