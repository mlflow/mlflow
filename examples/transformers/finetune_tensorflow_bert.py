import logging

from datasets import load_dataset


dataset = load_dataset("codyburker/yelp_review_sampled")
#dataset[100]

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", TOKENIZERS_PARALLELISM=False)



def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(20))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))

from transformers import DefaultDataCollator

data_collator = DefaultDataCollator(return_tensors="tf")

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

import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification

model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy(),
)
import mlflow


with mlflow.start_run():
    #model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=1)
    mlflow.transformers.log_model(model, "bert_artifact", task="text-classification", tokenizer=tokenizer)

    model_uri = mlflow.get_artifact_uri("bert_artifact")
    print(model_uri)
    loaded_model = mlflow.transformers.load_model(model_uri)
    print(loaded_model(["All is well"]))
