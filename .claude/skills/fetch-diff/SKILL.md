---
name: fetch-diff
description: Fetch PR diff with filtering and line numbers for code review.
allowed-tools:
  - Bash(uv run skills fetch-diff:*)
---

# Fetch PR Diff

Fetches a pull request diff and adds line numbers for easier review comment placement.

Auto-generated files (protobuf `.py`/`.pyi` in `mlflow/protos`, lock files like `yarn.lock`/`uv.lock`/`package-lock.json`, and generated `.java` files) are included in the output with their `diff --git` header lines visible, but their hunk content is replaced with `[Auto-generated file - diff masked]`. This lets reviewers know that a PR touches these files without being overwhelmed by noisy generated diffs.

Notebook (`.ipynb`) files are treated as regular files and show their full diff.

## Usage

```bash
uv run skills fetch-diff <pr_url> [--files <pattern> ...]
```

Examples:

```bash
# Fetch the full diff
uv run skills fetch-diff https://github.com/mlflow/mlflow/pull/123

# Fetch only Python files
uv run skills fetch-diff https://github.com/mlflow/mlflow/pull/123 --files '*.py'

# Fetch only frontend files
uv run skills fetch-diff https://github.com/mlflow/mlflow/pull/123 --files 'mlflow/server/js/*'

# Multiple patterns
uv run skills fetch-diff https://github.com/mlflow/mlflow/pull/123 --files '*.py' '*.ts'
```

Token is auto-detected from `GH_TOKEN` env var or `gh auth token`.

## Output Example

**Regular file:**

```
diff --git a/path/to/file.py b/path/to/file.py
index abc123..def456 100644
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,7 +10,7 @@
10    10 |  import os
11    11 |  import sys
12    12 |  from typing import Optional
13       | -from old_module import OldClass
      14 | +from new_module import NewClass
14    15 |
15    16 |  def process_data(input_file: str) -> dict:
```

**Auto-generated file (masked):**

```
diff --git a/mlflow/protos/service_pb2.py b/mlflow/protos/service_pb2.py
index abc123..def456 100644
--- a/mlflow/protos/service_pb2.py
+++ b/mlflow/protos/service_pb2.py
[Auto-generated file - diff masked]
```

Each line is annotated as `old_line new_line | <marker> content`:

- `-` marker (left number only) -> deleted line, `side=LEFT`, `line=old_line`
- `+` marker (right number only) -> added line, `side=RIGHT`, `line=new_line`
- No marker (both numbers) -> unchanged line, `side=RIGHT`, `line=new_line`
