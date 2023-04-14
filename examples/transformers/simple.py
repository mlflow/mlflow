import transformers
import mlflow

task = "text2text-generation"
architecture = "mrm8488/t5-base-finetuned-common_gen"
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(architecture)
tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)

generation_pipeline = transformers.pipeline(
    task=task,
    tokenizer=tokenizer,
    model=model,
)

with mlflow.start_run() as run:
    mlflow.transformers.log_model(
        transformers_model=generation_pipeline,
        artifact_path="sentence_builder",
        input_example="keyboard engineer model data science",
    )

model_uri = f"runs:/{run.info.run_id}/sentence_builder"

sentence_generator = mlflow.pyfunc.load_model(model_uri)

print(sentence_generator.predict(["pack howl moon wolf", "brush easel paint landscape"]))
