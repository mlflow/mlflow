---
name: add-review-comment
description: Add a review comment to a GitHub pull request.
allowed-tools:
  - Bash(gh api:*)
  - Bash(gh pr view:*)
---

# Add Review Comment

Adds a review comment to a specific line in a GitHub pull request.

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
