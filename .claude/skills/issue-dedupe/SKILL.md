---
name: issue-dedupe
description: Find likely duplicates of a GitHub issue and write a comment body linking to them
disable-model-invocation: true
allowed-tools:
  - Bash(gh issue view:*)
  - Bash(gh issue list:*)
  - Bash(gh search issues:*)
  - Write(//tmp/dedupe-comment.md)
argument-hint: "<owner_repo> <issue_number>"
arguments: [owner_repo, issue_number]
---

# Find Duplicate Issues

## Usage

```
/issue-dedupe <owner_repo> <issue_number>
```

## Inputs

- Owner/Repo: `$owner_repo`
- Issue number: `$issue_number` (the newly opened issue to find duplicates for)

## Output contract

Your only output is `/tmp/dedupe-comment.md`, posted verbatim as a comment on the
issue. Write it only when you have found duplicates. If there are none, or the
issue should be skipped per step 2, write nothing and stop: a missing file is the
normal outcome.

Format, with up to 3 links, best match first:

```markdown
Found 2 possible duplicate issues:

1. https://github.com/mlflow/mlflow/issues/123
2. https://github.com/mlflow/mlflow/issues/456

If one of these already covers your report, please close this issue and 👍 the existing one instead so the discussion stays in one place.
```

Add no commentary beyond that. An AI disclaimer and a workflow link are appended
for you. The `gh` token here is read-only, so do not try to comment, label, or
close anything yourself.

## Instructions

1. Read the issue: `gh issue view $issue_number --repo $owner_repo --comments`.

2. Stop without writing the file if any of these hold:

   - It is already closed.
   - There is nothing specific to match on: broad product feedback, a support
     question, positive feedback, or a feature idea with no concrete behavior.
   - It is a security report.
   - It already has a "possible duplicate" comment from an earlier run.

3. Summarize the issue in two or three sentences: the observed behavior, the
   component involved, and any distinctive error text. Everything below matches
   against that summary, not the raw body.

4. Search for candidates using
   `gh search issues "<query>" --repo $owner_repo --limit 10`.
   Run several searches in parallel covering different angles, since any single
   phrasing misses issues that describe the same bug in other words:

   - the distinctive error message or exception type, verbatim
   - the API or component name (e.g. `mlflow.log_model`, `MlflowClient`)
   - the user-facing symptom in the reporter's own words
   - the same symptom in the vocabulary a maintainer would use
   - the flavor, integration, or backend involved (e.g. `langchain`, `sqlalchemy`)

   Include closed issues: a closed duplicate is still the right place to point the
   reporter.

5. Filter hard. Keep a candidate only if a maintainer reading both issues would
   close this one as a duplicate. The same component failing a different way is
   not a duplicate, and neither is the same error message from an unrelated cause.
   Precision matters more than recall: a wrong link wastes the reporter's time,
   while a missing one costs nothing. When torn, drop the candidate.
