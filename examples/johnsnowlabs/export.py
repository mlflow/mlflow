import json
import os

import pandas as pd
from johnsnowlabs import nlp

import mlflow
from mlflow.pyfunc import spark_udf

# 1) Write your raw license.json string into the 'JOHNSNOWLABS_LICENSE_JSON' env variable for MLflow
creds = {
    "AWS_ACCESS_KEY_ID": "...",
    "AWS_SECRET_ACCESS_KEY": "...",
    "SPARK_NLP_LICENSE": "...",
    "SECRET": "...",
}
os.environ["JOHNSNOWLABS_LICENSE_JSON"] = json.dumps(creds)

# 2) Install enterprise libraries
nlp.install()
# 3) Start a Spark session with enterprise libraries
spark = nlp.start()

# 4) Load a model and test it
nlu_model = "en.classify.bert_sequence.covid_sentiment"
model_save_path = "my_model"
johnsnowlabs_model = nlp.load(nlu_model)
johnsnowlabs_model.predict(["I hate COVID,", "I love COVID"])

# 5) Export model with pyfunc and johnsnowlabs flavors
with mlflow.start_run():
    model_info = mlflow.johnsnowlabs.log_model(johnsnowlabs_model, name=model_save_path)

# 6) Load model with johnsnowlabs flavor
mlflow.johnsnowlabs.load_model(model_info.model_uri)

# 7) Load model with pyfunc flavor
mlflow.pyfunc.load_model(model_save_path)

pandas_df = pd.DataFrame({"text": ["Hello World"]})
spark_df = spark.createDataFrame(pandas_df).coalesce(1)
pyfunc_udf = spark_udf(
    spark=spark,
    model_uri=model_save_path,
    env_manager="virtualenv",
    result_type="string",
)
new_df = spark_df.withColumn("prediction", pyfunc_udf(*pandas_df.columns))

# 9) You can now use the mlflow models serve command to serve the model see next section

# 10)  You can also use x command to deploy model inside of a container see next section
