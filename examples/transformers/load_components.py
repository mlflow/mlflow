import transformers

import mlflow

translation_pipeline = transformers.pipeline(
    task="translation_en_to_fr",
    model=transformers.T5ForConditionalGeneration.from_pretrained("t5-small"),
    tokenizer=transformers.T5TokenizerFast.from_pretrained("t5-small", model_max_length=100),
)

signature = mlflow.models.infer_signature(
    "Hi there, chatbot!",
    mlflow.transformers.generate_signature_output(translation_pipeline, "Hi there, chatbot!"),
)

with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=translation_pipeline,
        name="french_translator",
        signature=signature,
    )

translation_components = mlflow.transformers.load_model(
    model_info.model_uri, return_type="components"
)

for key, value in translation_components.items():
    print(f"{key} -> {type(value).__name__}")

response = translation_pipeline("MLflow is great!")

print(response)

reconstructed_pipeline = transformers.pipeline(**translation_components)

reconstructed_response = reconstructed_pipeline(
    "transformers makes using Deep Learning models easy and fun!"
)

print(reconstructed_response)
