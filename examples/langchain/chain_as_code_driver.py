# This is an example for logging a Langchain model from code using the
# mlflow.langchain.log_model API. When a path to a valid Python script is submitted to the
# lc_model argument, the model code itself is serialized instead of the model object.
# Within the targeted script, the model implementation must be defined and set by
# using the mlflow.models.set_model API.

import mlflow

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "What is Retrieval-augmented Generation?",
        }
    ]
}

# Specify the path to the chain notebook
chain_path = "chain_as_code.py"

print(f"Chain path: {chain_path}")

print("Logging model as code using Langchain log model API")
with mlflow.start_run():
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=chain_path,
        name="chain",
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
