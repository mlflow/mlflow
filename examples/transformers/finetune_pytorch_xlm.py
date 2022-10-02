import torch
from datasets import load_dataset, load_metric
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, get_scheduler

import mlflow

dataset = load_dataset("yelp_review_full")

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")


small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))


train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)


model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=5)


optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)

lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


with mlflow.start_run():

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    mlflow.transformers.log_model(model, "bert_artifact", task="text-classification", tokenizer=tokenizer)

    metric = load_metric("accuracy")
    model.eval()

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    print(metric.compute())

    model_uri = mlflow.get_artifact_uri("bert_artifact")
    loaded_model = mlflow.transformers.load_model(model_uri)
    print(loaded_model(["All is well"]))