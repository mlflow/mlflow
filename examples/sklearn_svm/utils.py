#!/usr/bin/env python3

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



def save_fig(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

def top_terms_from_svm(pipe, k, class_names):
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    
    feature_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_.ravel()

    pos_idx = np.argsort(coefs)[-k:][::-1]
    neg_idx = np.argsort(coefs)[:k]

    rows = []

    for rank, wgt in enumerate(pos_idx, 1):
        rows.append({"direction": "+", "rank": rank, "feature": feature_names[wgt], "weight": float(coefs[wgt])})

    for rank, wgt in enumerate(neg_idx, 1):
        rows.append({"direction": "-", "rank": rank, "feature": feature_names[wgt], "weight": float(coefs[wgt])})

    return pd.DataFrame(rows)

def plot_top_bars(df, direction, out_path, title):
    sub = df[df["direction"] == direction].copy()
    sub = sub.sort_values("rank")
    plt.figure(figsize=(8, 6))
    plt.barh(sub["feature"].iloc[::-1], sub["weight"].iloc[::-1])
    plt.xlabel("SVM Weight")
    plt.ylabel("term")
    plt.title(title)
    plt.tight_layout()
    save_fig(out_path)

def plot_coef_hist(coefs, out_path):
    plt.figure(figsize=(7, 4))
    plt.hist(coefs, bins=60)
    plt.xlabel("SVM Weight")
    plt.ylabel("count")
    plt.title("Linear SVM Coefficient Dist")
    save_fig(out_path)
