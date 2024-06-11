import mlflow

# Use the Unity Catalog model registry
mlflow.set_registry_uri("databricks-uc")

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "What is Retrieval-augmented Generation?",
        }
    ]
}

# Specify the path to the chain notebook
chain_notebook_path = "chain_as_code.py"

print(f"Chain notebook path: {chain_notebook_path}")

print("Logging model model as code using Langchain log model API")
with mlflow.start_run():
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=chain_notebook_path,
        artifact_path="chain",
        input_example=input_example,
    )

print("Loading model using Langchain load model API")
model = mlflow.langchain.load_model(logged_chain_info.model_uri)
output = model.invoke(input_example)
print(f"Output: {output}")

print("Loading model using Pyfunc load model API")
pyfunc_model = mlflow.pyfunc.load_model(logged_chain_info.model_uri)
output = pyfunc_model.predict([input_example])
print(f"Output: {output}")
