---
name: add-review-comment
description: Add a review comment to a GitHub pull request.
allowed-tools:
  - Bash(gh api:*)
  - Bash(gh pr view:*)
  - Bash(uv run skills fetch-diff:*)
---

# Add Review Comment

Adds a review comment to a specific line in a GitHub pull request.

## Finding the Right Line

Before posting a comment you need to know the exact `path`, `line` number, and `side` (`LEFT`/`RIGHT`). Use `fetch-diff` to find these values directly from the annotated diff output.

### Step 1 — Fetch the diff for relevant files

```bash
uv run skills fetch-diff <pr_url> --files '<glob_pattern>'
```

Example — narrow to a single file:

```bash
uv run skills fetch-diff https://github.com/mlflow/mlflow/pull/123 --files 'mlflow/tracking/client.py'
```

### Step 2 — Read the annotated output

The diff output uses the format `old_line new_line | <marker> content`:

```
diff --git a/mlflow/tracking/client.py b/mlflow/tracking/client.py
--- a/mlflow/tracking/client.py
+++ b/mlflow/tracking/client.py
@@ -42,7 +42,7 @@
42    42 |  import os
43    43 |  import sys
44       | -def old_function():
      44 | +def new_function():
45    45 |      pass
```

- **`old_line new_line`** — both numbers present means the line is unchanged (`side=RIGHT`).
- **`old_line `** (right column blank) and marker `-` — line was deleted; use that `old_line` as `line` and `side=LEFT`.
- **` new_line`** (left column blank) and marker `+` — line was added; use that `new_line` as `line` and `side=RIGHT`.

Grep for the code you want to comment on:

```bash
uv run skills fetch-diff https://github.com/mlflow/mlflow/pull/123 --files 'mlflow/tracking/client.py' \
  | grep -n 'new_function'
```

### Step 3 — End-to-end example

```bash
# 1. Fetch the diff filtered to the file of interest
uv run skills fetch-diff https://github.com/mlflow/mlflow/pull/123 \
  --files 'mlflow/tracking/client.py'

# 2. Suppose the output shows the added line:
#        44 | +def new_function():
#    → new_line=44, marker="+", so line=44 and side=RIGHT

# 3. Post the comment on that line
gh api repos/mlflow/mlflow/pulls/123/comments \
  -f body='Consider adding a docstring here.

🤖 Generated with Claude' \
  -f path='mlflow/tracking/client.py' \
  -F line=44 \
  -f side=RIGHT \
  -f commit_id="$(gh pr view 123 --repo mlflow/mlflow --json headRefOid -q .headRefOid)" \
  --jq '.html_url'
```

## Usage

**Single-line comment:**

```bash
gh api repos/<owner>/<repo>/pulls/<pr_number>/comments \
  # Body must end with "🤖 Generated with Claude" on a separate line
  -f body=<comment> \
  -f path=<file_path> \
  -F line=<line_number> \
  -f side=<side> \
  -f commit_id="$(gh pr view <pr_number> --repo <owner>/<repo> --json headRefOid -q .headRefOid)" \
  --jq '.html_url'
```

**Multi-line comment:**

```bash
gh api repos/<owner>/<repo>/pulls/<pr_number>/comments \
  # Body must end with "🤖 Generated with Claude" on a separate line
  -f body=<comment> \
  -f path=<file_path> \
  -F start_line=<first_line> \
  -f start_side=<side> \
  -F line=<last_line> \
  -f side=<side> \
  -f commit_id="$(gh pr view <pr_number> --repo <owner>/<repo> --json headRefOid -q .headRefOid)" \
  --jq '.html_url'
```

## Parameters

- `line`: Line number in the file (for multi-line, the last line)
- `side`: `RIGHT` for added/modified lines (+), `LEFT` for deleted lines (-)
- `start_line`/`start_side`: For multi-line comments, the first line of the range

## Best Practices

- Use suggestion blocks (three backticks + "suggestion") for simple fixes that maintainers can apply with one click

  ````
  ```suggestion
  <suggested code here>
  ```
  ````

  Note: Preserve original indentation exactly in suggestion blocks

- For repetitive issues, leave one representative comment instead of flagging every instance
- For bugs, explain the potential problem and suggested fix clearly
