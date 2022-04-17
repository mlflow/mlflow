# Usage

## Build jobs

```bash
# Build all jobs
cross-version-test build

# Build jobs without using cache
cross-version-test build --no-cache

# Build sklearn jobs
cross-version-test build -p sklearn

# Build sklearn autologging jobs
cross-version-test build -p 'sklearn.+autologging'
```

## Build jobs relevant to configuration file changes

```bash
# Build sklearn autologging jobs if v1.yml and v2.yml have the difference below
cross-version-test diff --versions-yaml v1.yml --ref-version-yaml v2.yml
```

```diff
diff --git a/v1.yml b/v2.yml
index 3738051ea..2251369b1 100644
--- a/v1.yml
+++ b/v2.yml
@@ -19,7 +19,7 @@ sklearn:
   autologging:
     minimum: "0.20.3"
     maximum: "1.0.2"
     run: |
-      pytest tests/sklearn/test_sklearn_autolog.py --large
+      pytest tests/sklearn/test_sklearn_autolog_new.py --large
```

## Build jobs relevant to flavor file changes

```bash
# Build all sklearn jobs
cross-version-test diff --changed-files mlflow/sklearn/__init__.py
```

## List jobs

```bash
cross-version-test list
```

## Run jobs

```bash
# Run all jobs
cross-version-test run

# Run sklearn jobs
cross-version-test run -p sklearn

# Run sklearn autologging jobs
cross-version-test run -p 'sklearn.+autologging'

# Run a single job
cross-version-test run -p sklearn_1.0.2_autologging

# Run `pytest /path/to/test.py` in sklearn autologging jobs
cross-version-test run -p 'sklearn.+autologging' pytest /path/to/test.py
```

## Remove jobs

```bash
# Remove containers, networks, and volumes
cross-version-test down

# Remove containers, networks, volumes, and images
cross-version-test down --rmi
```
