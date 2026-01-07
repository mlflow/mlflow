---
name: review-comment
description: Add a review comment to a GitHub pull request.
allowed-tools:
  - Bash
---

# Review Comment

Adds a review comment to a specific line in a GitHub pull request.

## Usage

**Single-line comment:**

```bash
gh api repos/<owner>/<repo>/pulls/<pr_number>/comments \
  -f body="<comment>" \
  -f path="<file_path>" \
  -F line=<line_number> \
  -f side="RIGHT" \
  -f commit_id="$(gh pr view <pr_number> --repo <owner>/<repo> --json headRefOid -q .headRefOid)"
```

**Multi-line comment:**

```bash
gh api repos/<owner>/<repo>/pulls/<pr_number>/comments \
  -f body="<comment>" \
  -f path="<file_path>" \
  -F start_line=<first_line> \
  -f start_side="RIGHT" \
  -F line=<last_line> \
  -f side="RIGHT" \
  -f commit_id="$(gh pr view <pr_number> --repo <owner>/<repo> --json headRefOid -q .headRefOid)"
```

## Parameters

- `line`: Line number in the file (for multi-line, the last line)
- `side`: `RIGHT` for added/modified lines (+), `LEFT` for deleted lines (-)
- `start_line`/`start_side`: For multi-line comments, the first line of the range
