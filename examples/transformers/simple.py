import transformers

import mlflow

task = "text2text-generation"

generation_pipeline = transformers.pipeline(
    task=task,
    model="declare-lab/flan-alpaca-base",
)

input_example = ["prompt 1", "prompt 2", "prompt 3"]

parameters = {"max_length": 512, "do_sample": True}

signature = mlflow.models.infer_signature(
    input_example,
    mlflow.transformers.generate_signature_output(generation_pipeline, input_example),
    parameters,
)

with mlflow.start_run() as run:
    model_info = mlflow.transformers.log_model(
        transformers_model=generation_pipeline,
        artifact_path="text_generator",
        input_example=["prompt 1", "prompt 2", "prompt 3"],
        signature=signature,
    )

sentence_generator = mlflow.pyfunc.load_model(model_info.model_uri)

print(
    sentence_generator.predict(
        ["tell me a story about rocks", "Tell me a joke about a dog that likes spaghetti"],
        # pass in additional parameters applied to the pipeline during inference
        params=parameters,
    )
)
