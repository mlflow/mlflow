## MLflow automatic Logging with SynapseML

[MLflow automatic logging](https://www.mlflow.org/docs/latest/tracking.html#automatic-logging) allows you to log metrics, parameters, and models without the need for explicit log statements.
SynapseML supports autologging for every model in the library.

Install SynapseML library following this [guidance](https://microsoft.github.io/SynapseML/docs/getting_started/installation/)

Default mlflow [log_model_allowlist file](https://github.com/mlflow/mlflow/blob/master/mlflow/pyspark/ml/log_model_allowlist.txt) already includes some SynapseML models. To enable more models, you could use `mlflow.pyspark.ml.autolog(log_model_allowlist=YOUR_SET_OF_MODELS)` function, or follow the below guidance by specifying a link to the file and update spark configuration.

To enable autologging with your custom log_model_allowlist file:

1. Put your customized log_model_allowlist file at a place that your code has access to. ([SynapseML official log_model_allowlist file](https://mmlspark.blob.core.windows.net/publicwasb/log_model_allowlist.txt))
   For example:

- In Synapse `wasb://<containername>@<accountname>.blob.core.windows.net/PATH_TO_YOUR/log_model_allowlist.txt`
- In Databricks `/dbfs/FileStore/PATH_TO_YOUR/log_model_allowlist.txt`.

2. Set spark configuration `spark.mlflow.pysparkml.autolog.logModelAllowlistFile` to the path of your `log_model_allowlist.txt` file.
3. Call `mlflow.pyspark.ml.autolog()` before your training code to enable autologging for all supported models.

Note:

If you want to support autologging of PySpark models not present in the log_model_allowlist file, you can add such models to the file.

## Configuration process in Databricks as an example

1. Install latest MLflow via `%pip install mlflow -u`
2. Upload your customized `log_model_allowlist.txt` file to dbfs by clicking File/Upload Data button on Databricks UI.
3. Set Cluster Spark configuration following [this documentation](https://docs.microsoft.com/en-us/azure/databricks/clusters/configure#spark-configuration)

```
spark.mlflow.pysparkml.autolog.logModelAllowlistFile /dbfs/FileStore/PATH_TO_YOUR/log_model_allowlist.txt
```

4. Run the following line before your training code executes.

```python
import mlflow

mlflow.pyspark.ml.autolog()
```

You can customize how autologging works by supplying appropriate [parameters](https://www.mlflow.org/docs/latest/python_api/mlflow.pyspark.ml.html#mlflow.pyspark.ml.autolog).

5. To find your experiment's results via the `Experiments` tab of the MLflow UI.
   <img src="https://mmlspark.blob.core.windows.net/graphics/adb_experiments.png" width="1200" />

## Example for ConditionalKNNModel

```python
from pyspark.ml.linalg import Vectors
from synapse.ml.nn import ConditionalKNN

df = spark.createDataFrame(
    [
        (Vectors.dense(2.0, 2.0, 2.0), "foo", 1),
        (Vectors.dense(2.0, 2.0, 4.0), "foo", 3),
        (Vectors.dense(2.0, 2.0, 6.0), "foo", 4),
        (Vectors.dense(2.0, 2.0, 8.0), "foo", 3),
        (Vectors.dense(2.0, 2.0, 10.0), "foo", 1),
        (Vectors.dense(2.0, 2.0, 12.0), "foo", 2),
        (Vectors.dense(2.0, 2.0, 14.0), "foo", 0),
        (Vectors.dense(2.0, 2.0, 16.0), "foo", 1),
        (Vectors.dense(2.0, 2.0, 18.0), "foo", 3),
        (Vectors.dense(2.0, 2.0, 20.0), "foo", 0),
        (Vectors.dense(2.0, 4.0, 2.0), "foo", 2),
        (Vectors.dense(2.0, 4.0, 4.0), "foo", 4),
        (Vectors.dense(2.0, 4.0, 6.0), "foo", 2),
        (Vectors.dense(2.0, 4.0, 8.0), "foo", 2),
        (Vectors.dense(2.0, 4.0, 10.0), "foo", 4),
        (Vectors.dense(2.0, 4.0, 12.0), "foo", 3),
        (Vectors.dense(2.0, 4.0, 14.0), "foo", 2),
        (Vectors.dense(2.0, 4.0, 16.0), "foo", 1),
        (Vectors.dense(2.0, 4.0, 18.0), "foo", 4),
        (Vectors.dense(2.0, 4.0, 20.0), "foo", 4),
    ],
    ["features", "values", "labels"],
)

cnn = ConditionalKNN().setOutputCol("prediction")
cnnm = cnn.fit(df)

test_df = spark.createDataFrame(
    [
        (Vectors.dense(2.0, 2.0, 2.0), "foo", 1, [0, 1]),
        (Vectors.dense(2.0, 2.0, 4.0), "foo", 4, [0, 1]),
        (Vectors.dense(2.0, 2.0, 6.0), "foo", 2, [0, 1]),
        (Vectors.dense(2.0, 2.0, 8.0), "foo", 4, [0, 1]),
        (Vectors.dense(2.0, 2.0, 10.0), "foo", 4, [0, 1]),
    ],
    ["features", "values", "labels", "conditioner"],
)

display(cnnm.transform(test_df))
```

This code should log one run with a ConditionalKNNModel artifact and its parameters.
<img src="https://mmlspark.blob.core.windows.net/graphics/autologgingRunSample.png" width="1200" />
