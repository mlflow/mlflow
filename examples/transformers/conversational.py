import transformers

import mlflow

conversational_pipeline = transformers.pipeline(model="microsoft/DialoGPT-medium")

signature = mlflow.models.infer_signature(
    "Hi there, chatbot!",
    mlflow.transformers.generate_signature_output(conversational_pipeline, "Hi there, chatbot!"),
)

with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=conversational_pipeline,
        name="chatbot",
        task="conversational",
        signature=signature,
        input_example="A clever and witty question",
    )

# Load the conversational pipeline as an interactive chatbot

chatbot = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

first = chatbot.predict("What is the best way to get to Antarctica?")

print(f"Response: {first}")

second = chatbot.predict("What kind of boat should I use?")

print(f"Response: {second}")
