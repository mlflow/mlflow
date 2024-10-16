from mlflow import MlflowClient

client = MlflowClient()
client.create_registered_model("sk-learn-random-forest-reg-model-2")

result = client.create_model_version(
    name="sk-learn-random-forest-reg-model-2",
    source="mlruns/0/d16076a3ec534311817565e6527539c0/artifacts/sklearn-model",
    run_id="d16076a3ec534311817565e6527539c0",
    tags={"version": "1.0.0"},
)

result2 = client.create_model_version(
    name="sk-learn-random-forest-reg-model-2",
    source="mlruns/0/d16076a3ec534311817565e6527539c0/artifacts/sklearn-model",
    run_id="asdf",
    tags={"version": "2.0.0"},
)

result3 = client.create_model_version(
    name="sk-learn-random-forest-reg-model-2",
    source="mlruns/0/d16076a3ec534311817565e6527539c0/artifacts/sklearn-model",
    run_id="asdfasdf",
    tags={"version": "3.0.0"},
)

print(result)
print(result2)
print(result3)