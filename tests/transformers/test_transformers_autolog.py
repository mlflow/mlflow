import pytest

import mlflow

from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer, sample_dataset



def test_with_autolog():
    mlflow.autolog()

    dataset = load_dataset("sst2")

    train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=8)
    eval_dataset = dataset["validation"]

    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

    # Create trainer
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss_class=CosineSimilarityLoss,
        metric="accuracy",
        batch_size=16,
        num_iterations=20,  # The number of text pairs to generate for contrastive learning
        num_epochs=1,  # The number of epochs to use for contrastive learning
        column_mapping={"sentence": "text", "label": "label"}
        # Map dataset columns to text/label expected by trainer
    )

    # Train and evaluate
    trainer.train()
    metrics = trainer.evaluate()

    print(metrics)

    # Run inference
    preds = trainer.model(["i loved the spiderman movie!", "pineapple on pizza is the worst 🤮"])

    print(preds)


