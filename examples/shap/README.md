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

## How to run the scripts

```bash
python <script_name>
```

## How to view the logged explanations:

- Run `mlflow ui` to launch the MLflow UI.
- Open http://127.0.0.1:5000 on your browser.
- Click the latest run in the runs table.
- Scroll down to the artifact viewer.
- Open a folder named `model_explanations_shap`.
