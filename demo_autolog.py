"""
Demo script to test transformers autologging with MLflow UI.
Run this, then open the MLflow UI to see logged parameters, metrics, and model.
"""
import mlflow
import mlflow.transformers
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)
import torch
import tempfile

# Enable autologging
mlflow.transformers.autolog(log_models=True, log_input_examples=True)

# Set experiment
mlflow.set_experiment("transformers_autolog_demo")

print("Loading model and tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

# Create a small dataset
train_texts = [
    "I love this product!",
    "This is terrible.",
    "Great experience!",
    "Awful service.",
]
train_labels = [1, 0, 1, 0]

train_encodings = tokenizer(train_texts, truncation=True, padding=True)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SimpleDataset(train_encodings, train_labels)

with tempfile.TemporaryDirectory() as tmpdir:
    training_args = TrainingArguments(
        output_dir=tmpdir,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Starting training with MLflow autologging...")
    trainer.train()
    
mlflow.flush_async_logging()
print("\n✅ Training complete!")
print("\nTo view results in MLflow UI, run:")
print("   mlflow ui --port 5000")
print("\nThen open: http://localhost:5000")
