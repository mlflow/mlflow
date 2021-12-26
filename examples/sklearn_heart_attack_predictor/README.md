# Scikit-learn Heart Attack Prediction Example

This example trains an sklearn regression model for predicting heart attack posibilities.

## To run the project locally

1. clone the repo
2. run the train.py , to train the Heart Attack Prediction model

   ```python
   python3 train.py
   ```

3. To log the model in MLflow run the log_model.py script

   ```python
   python3 log_model.py
   ```

   It will create a local sqlite database (mlruns.db) and a artifact folder (mlruns).

4. Now serve the model using mlflow serve command

   ```
   mlflow models serve -m models:/   heart-attack-prediction-model/production -p 2000 --no-conda
   ```

5. Finally you can hit the curl command and get the prediction

   ```
   curl http://127.0.0.1:2000/invocations -H    'Content-Type: application/json' -d '{"columns":   ["age", "sex", "cp", "trtbps", "chol", "fbs",    "restecg", "thalachh", "exng", "oldpeak", "slp",    "caa", "thall"],"data": [[0.625000,1.0,3,0.716981,0.   369863,0,0,0.671756,0,0.032258,1,0,3]]}'
   ```
   
   ```
   curl http://127.0.0.1:2000/invocations -H    "Content-Type: application/json" -d '{"instances":[   {"age":0.625000, "sex":1, "cp":3, "trtbps":0.716981,    "chol":0.369863, "fbs":0, "restecg":0, "thalachh":0.   671756, "exng":0, "oldpeak":0.032258, "slp":1,    "caa":0, "thall":3}]}'
   ```

#
#

**Note :** 

*  We have used signature to put validation in my input
*  We have put the model into production through code but we can do it manually through mlflow ui as well