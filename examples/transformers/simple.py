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
    model_info = mlflow.transformers.log_model(
        transformers_model=generation_pipeline,
        artifact_path="sentence_builder",
        input_example="keyboard engineer model data science",
    )

sentence_generator = mlflow.pyfunc.load_model(model_info.model_uri)

print(sentence_generator.predict(["pack howl moon wolf", "brush easel paint landscape"]))
