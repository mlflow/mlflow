---
name: annotate-diff
description: Annotate a diff read from stdin with line numbers for code review.
---

# Annotate Diff

Reads a diff on stdin and adds line numbers for easier review comment placement. Auto-generated files are shown with masked diffs.

## Usage

```bash
<diff source> | uv run --package skills skills annotate-diff [--files <pattern> ...]
```

In a `refs/pull/<pr_number>/merge` checkout, which is what the review workflows set up, `HEAD^1 HEAD` is exactly the PR diff:

```bash
# Annotate the full diff
git diff HEAD^1 HEAD | uv run --package skills skills annotate-diff

# Only Python files
git diff HEAD^1 HEAD | uv run --package skills skills annotate-diff --files '*.py'

# Only frontend files
git diff HEAD^1 HEAD | uv run --package skills skills annotate-diff --files 'mlflow/server/js/*'

# Multiple patterns
git diff HEAD^1 HEAD | uv run --package skills skills annotate-diff --files '*.py' '*.ts'
```

Outside such a checkout, pipe the PR diff in instead:

```bash
gh pr diff https://github.com/mlflow/mlflow/pull/123 | uv run --package skills skills annotate-diff
```

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
diff --git a/uv.lock b/uv.lock
index abc123..def456 100644
--- a/uv.lock
+++ b/uv.lock
[Auto-generated file - diff masked]
```

**Deleted file (masked):**

```
diff --git a/path/to/removed.py b/dev/null
index abc123..0000000 100644
--- a/path/to/removed.py
+++ /dev/null
[Deleted file - diff masked]
```

Each line is annotated as `old_line new_line | <marker> content`:

- `-` marker (left number only) -> deleted line, `side=LEFT`, `line=old_line`
- `+` marker (right number only) -> added line, `side=RIGHT`, `line=new_line`
- No marker (both numbers) -> unchanged line, `side=RIGHT`, `line=new_line`
