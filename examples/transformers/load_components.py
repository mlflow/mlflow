import transformers

import mlflow

pipeline = transformers.pipeline(
    task="fill-mask",
    model=transformers.AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased"),
    tokenizer=transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased"),
)

with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=pipeline,
        name="mask_filler",
        input_example="MLflow is [MASK]!",
    )

components = mlflow.transformers.load_model(model_info.model_uri, return_type="components")

for key, value in components.items():
    print(f"{key} -> {type(value).__name__}")

response = pipeline("MLflow is [MASK]!")

print(response)

reconstructed_pipeline = transformers.pipeline(**components)

reconstructed_response = reconstructed_pipeline("Transformers is [MASK]!")

print(reconstructed_response)
