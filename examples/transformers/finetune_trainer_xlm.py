import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.integrations import MLflowCallback
import mlflow


class CustomMlflowCallback(MLflowCallback):
    def __init__(self, task):
        super().__init__()
        self.task = task

    def on_save(self, args, state, control, model=None, tokenizer=None, train_dataloader=None, **kwargs):
        print("logging model")
        print(tokenizer)
        mlflow.transformers.log_model(model, artifact_path="model_artifact", tokenizer=tokenizer, task=self.task)
        model_uri = mlflow.get_artifact_uri("transformers_model")
        print(model_uri)


dataset = load_dataset("yelp_review_full")
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", TOKENIZERS_PARALLELISM=False)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(20))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))


model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=5)

metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", logging_strategy="epoch", save_strategy="epoch")


trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.remove_callback(MLflowCallback)
trainer.add_callback(CustomMlflowCallback(task="text-classification"))
trainer.train()

model_uri = mlflow.get_artifact_uri("model_artifact")
print(model_uri)
loaded_model = mlflow.transformers.load_model(model_uri)
print(loaded_model(["All is well"]))