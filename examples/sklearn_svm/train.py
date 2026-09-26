#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import plot_top_bars, plot_coef_hist, top_terms_from_svm


def load_sst2():
    ds = load_dataset("glue", "sst2")

    train_ds = [ex["sentence"] for ex in ds["train"]]
    train_labels = np.array([ex["label"] for ex in ds["train"]])
    class_names = ['negative', 'positive']

    return train_ds, train_labels, class_names

def run(args):
    mlflow.set_experiment(args.experiment_name)

    texts, labels, class_names = load_sst2()

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, args.max_ngram),
                    max_features=None if args.max_features in (None, 0) else int(args.max_features),
                    min_df=args.min_df,
                    )),
        ("clf", LinearSVC(C=args.C, random_state=args.seed)),
    ])

    with mlflow.start_run(run_name=args.run_name):
        pipe.fit(texts, labels)

        explain_dir = Path("artifacts") / "explainability"
        explain_dir.mkdir(parents=True, exist_ok=True)
        
        # getting the top freq items and predicting sentiment proba
        weights_df = top_terms_from_svm(pipe, args.top_k, class_names)
        weights_to_csv_dir = explain_dir / "feature_weights.csv"
        weights_df.to_csv(weights_to_csv_dir, index=False)

        # Plotting histogram
        pos_dir = explain_dir / "top_pos_terms.png"
        neg_dir = explain_dir / "top_neg_terms.png"

        plot_top_bars(weights_df, "+", pos_dir, "Top Positive Terms (SST2)")
        plot_top_bars(weights_df, "-", neg_dir, "Top Negative Terms (SST2)")

        mlflow.log_artifact(str(pos_dir), artifact_path="explainability")
        mlflow.log_artifact(str(neg_dir), artifact_path="explainability")

        #coef hist plt
        coef_dir = explain_dir / "coef_distribution.png"
        coefs = pipe.named_steps["clf"].coef_.ravel()
        plot_coef_hist(coefs, coef_dir)
        mlflow.log_artifact(str(coef_dir), artifact_path="explainability")

    print("Process Done. Open MLFlow UI and navigate to 'Artifacts -> explainability'")



# --- CLI ----

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--max_ngram", type=int, default=2)
    p.add_argument("--max_features", type=int, default=200000)
    p.add_argument("--min_df", type=int, default=1)
    p.add_argument("--C", type=int, default=1.0)

    p.add_argument("--top_k", type=int, default=25)
    p.add_argument("--run_name", type=str, default="explainability-only")
    p.add_argument("--experiment_name", type=str, default="sklearn-svm-sentiment")

    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
