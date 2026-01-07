---
name: fetch-diff
description: Fetch PR diff with filtering and line numbers for code review.
allowed-tools:
  - Bash
---

# Fetch PR Diff

Fetches a pull request diff, filters out auto-generated files, and adds line numbers for easier review comment placement.

## Usage

```bash
uv run .claude/skills/fetch-diff/fetch_diff.py <pr_url>
```

Example:

```bash
uv run .claude/skills/fetch-diff/fetch_diff.py https://github.com/mlflow/mlflow/pull/123
```

Token is auto-detected from `GITHUB_TOKEN` env var or `gh auth token`.

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
