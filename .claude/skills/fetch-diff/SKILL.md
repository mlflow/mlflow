---
name: fetch-diff
description: Fetch PR diff with filtering and line numbers for code review.
allowed-tools:
  - Bash(uv run skills fetch-diff:*)
---

# Fetch PR Diff

Fetches a pull request diff, filters out auto-generated files, and adds line numbers for easier review comment placement.

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

Each line is annotated as `old_line new_line | <marker> content`:

- `-` marker (left number only) -> deleted line, `side=LEFT`, `line=old_line`
- `+` marker (right number only) -> added line, `side=RIGHT`, `line=new_line`
- No marker (both numbers) -> unchanged line, `side=RIGHT`, `line=new_line`
