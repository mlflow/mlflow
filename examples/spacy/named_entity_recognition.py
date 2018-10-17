#!/usr/bin/env python
# coding: utf8
"""Example of training spaCy's named entity recognizer, starting off with an
existing model or a blank model.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import click
import random
from pathlib import Path

import mlflow
import mlflow.spacy
import spacy
from spacy.util import minibatch, compounding


# Training data
TRAIN_DATA = [
    ("Who is Shaka Khan?", {
        "entities": [(7, 17, "PERSON")]
    }),
    ("I like London and Berlin.", {
        "entities": [(7, 13, "LOC"), (18, 24, "LOC")]
    })
]


@click.command()
@click.option("--model", "-m", help="Model name. Defaults to blank 'en' model",
              type=str, default=None)
@click.option("--n_iter", "-n", help="Number of training iterations", type=int, default=20)
def main(model, n_iter):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # Create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # Otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # Add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        with mlflow.start_run():
            optimizer = nlp.begin_training()
            mlflow.log_param("n_iter", n_iter)
            for i in range(n_iter):
                random.shuffle(TRAIN_DATA)
                losses = {}
                # Batch up the examples using spaCy's minibatch
                batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(
                        texts,  # batch of texts
                        annotations,  # batch of annotations
                        drop=0.5,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)
                print("Losses", losses)
                mlflow.log_metric("loss", losses["ner"])

            # Test the trained model
            for text, _ in TRAIN_DATA:
                doc = nlp(text)
                print("------")
                print(text)
                for ent in doc.ents:
                    print(ent.label_, ent.text)

            mlflow.spacy.log_model(nlp, "model")


if __name__ == "__main__":
    main()
