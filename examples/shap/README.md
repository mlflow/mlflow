# SHAP Examples

Examples demonstrating use of the `mlflow.shap` APIs for model explainability.

| File                                                         | Task                      | Description                                                    |
| :----------------------------------------------------------- | :------------------------ | :------------------------------------------------------------- |
| [regression.py](regression.py)                               | Regression                | Log explanations for a LinearRegression model                  |
| [binary_classification.py](binary_classification.py)         | Binary classification     | Log explanations for a binary RandomForestClassifier model     |
| [multiclass_classification.py](multiclass_classification.py) | Multiclass classification | Log explanations for a multiclass RandomForestClassifier model |

## Prerequisites

Run the following command to install required packages:

```
pip install mlflow scikit-learn shap matplotlib
```
