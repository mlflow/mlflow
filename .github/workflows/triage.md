# Synthetic triage dataset

Each issue is separated by `---`. Expected result is noted in the header.

---

## Issue 1 — UI bug, no screenshot (expect comment)

**Title:** Sidebar overlaps main content on narrow screens

**Body:**
When I resize the browser window to less than 1000px, the left sidebar overlaps
with the experiment table. The columns become unreadable and I can't click on
any runs.

---

## Issue 2 — UI bug with screenshot (expect no comment)

**Title:** Run name truncated in experiment list

**Body:**
The run name gets cut off when it's longer than 30 characters. See below:

![truncated run name](https://user-images.githubusercontent.com/12345/screenshot.png)

MLflow 2.17.0, Python 3.11, macOS 14.

Steps:

1. Create a run with a long name
2. Open the experiment page
3. Observe the truncated name

---

## Issue 3 — Bug, no repro steps, no env info (expect comment)

**Title:** `mlflow.log_metric` silently drops NaN values

**Body:**
I noticed that when I log NaN as a metric value, it just disappears. No error,
no warning, nothing in the UI. This is really confusing because I thought my
training was going fine but the metrics were just not being recorded.

---

## Issue 4 — Bug with repro steps and env info (expect no comment)

**Title:** `mlflow server` crashes on startup with SQLite backend

**Body:**
**Environment:**

- OS: Ubuntu 22.04
- Python: 3.10.12
- MLflow: 2.18.0

**Steps to reproduce:**

1. Install mlflow 2.18.0
2. Run `mlflow server --backend-store-uri sqlite:///mlflow.db`
3. Server crashes with `OperationalError: database is locked`

**Traceback:**

```
Traceback (most recent call last):
  File "/home/user/.local/lib/python3.10/site-packages/mlflow/server/__init__.py", line 42, in _run_server
    store = _get_store(backend_store_uri)
  File "/home/user/.local/lib/python3.10/site-packages/mlflow/store/tracking/__init__.py", line 75, in _get_store
    return _tracking_store_registry.get_store(store_uri)
  File "/home/user/.local/lib/python3.10/site-packages/mlflow/store/tracking/sqlalchemy_store.py", line 112, in __init__
    self._setup_db()
sqlite3.OperationalError: database is locked
```

---

## Issue 5 — Feature request (expect no comment)

**Title:** Add support for grouping runs by tag in the UI

**Body:**
It would be great to be able to group runs in the experiment view by a specific
tag value. For example, I tag my runs with `model_type=cnn` or
`model_type=transformer` and I'd love to see them grouped in the table.

This is similar to how you can group by "dataset" in Weights & Biases.

---

## Issue 6 — Bug with repro steps but no env info (expect comment)

**Title:** Autologging fails with PyTorch Lightning 2.5

**Body:**
When I enable autologging with PyTorch Lightning 2.5, I get an AttributeError.

```python
import mlflow

mlflow.pytorch.autolog()

trainer = pl.Trainer(max_epochs=5)
trainer.fit(model, dataloader)
```

```
AttributeError: module 'pytorch_lightning' has no attribute 'callbacks'
```

---

## Issue 7 — Bug with env info but no repro steps (expect comment)

**Title:** Model serving returns 500 on valid input

**Body:**
I deployed a model using `mlflow models serve` and it returns 500 errors
on inputs that used to work fine. This started happening after I upgraded
MLflow.

Environment: Python 3.11, MLflow 2.18.0, macOS Sonoma.

---

## Issue 8 — UI bug, has env and repro but no screenshot (expect comment)

**Title:** Chart tooltip shows wrong metric value

**Body:**
The tooltip on the metric chart shows a different value than what's actually
plotted. The line is at ~0.95 but the tooltip says 0.42.

**Environment:** MLflow 2.17.0, Python 3.10, Chrome 130, Ubuntu 22.04

**Steps:**

1. Log 100 metric values for "accuracy"
2. Open the run page
3. Hover over the chart line near the end
4. Tooltip shows an incorrect value

---

## Issue 9 — Documentation issue (expect no comment)

**Title:** Typo in quickstart guide

**Body:**
In the quickstart guide, the command `mlflow server --port 500` should be
`mlflow server --port 5000`. The wrong port causes a "permission denied" error
on Linux.

---

## Issue 10 — Bug, detailed report with everything (expect no comment)

**Title:** `mlflow.evaluate()` fails with custom metrics on Spark DataFrame

**Body:**
**Environment:**

- OS: CentOS 7
- Python: 3.10.8
- MLflow: 2.17.2
- PySpark: 3.5.0

**Description:**
When passing a Spark DataFrame to `mlflow.evaluate()` with a custom metric
function, the evaluation fails with a serialization error.

**Steps to reproduce:**

1. Create a Spark DataFrame with columns `prediction` and `target`
2. Define a custom metric: `def rmse(eval_df, _): return np.sqrt(np.mean(...))`
3. Call `mlflow.evaluate(model, data=spark_df, extra_metrics=[rmse])`

**Error:**

```
pickle.PicklingError: Could not serialize object
```

**Expected behavior:**
The evaluation should work with Spark DataFrames just like it does with
Pandas DataFrames.

---

## Issue 11 — Security Vulnerability (expect no comment - skipped by workflow)

**Title:** Security Vulnerability: XSS in MLflow UI allows script injection

**Body:**
I discovered a cross-site scripting vulnerability in the MLflow UI that allows
an attacker to inject malicious JavaScript code through experiment names.

This is a serious security issue that needs immediate attention.
