import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TFAutoModelForSequenceClassification

import mlflow

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(16))
small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(16))

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

tf_train_dataset = small_train_dataset.to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)

tf_validation_dataset = small_eval_dataset.to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=8,
)


model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


with mlflow.start_run():
    model.fit(
        tf_train_dataset,
        validation_data=tf_validation_dataset,
        epochs=1
    )
    mlflow.transformers.log_model(model, "bert_artifact", task="text-classification", tokenizer=tokenizer)

    model_uri = mlflow.get_artifact_uri("bert_artifact")
    loaded_model = mlflow.transformers.load_model(model_uri)
    print(loaded_model(["All is well"]))
