# Basic authentication example

This example demonstrates the authentication and authorization feature of MLflow.

To run this example,

1. Start the tracking server
   ```shell
   mlflow ui --app-name=basic-auth
   ```
2. Go to `http://localhost:5000/signup` and register two users:
   - `(user_a, password_a)`
   - `(user_b, password_b)`
3. Run the script
   ```shell
   python auth.py
   ```
   Expected output:
   ```
   2023/05/02 14:03:58 INFO mlflow.tracking.fluent: Experiment with name 'experiment_a' does not exist. Creating a new experiment.
   {}
   API request to endpoint /api/2.0/mlflow/runs/create failed with error code 403 != 200. Response body: 'Permission denied'
   ```
