---
name: issue-dedupe
description: Find likely duplicates of a GitHub issue and write their numbers to a payload file
disable-model-invocation: true
allowed-tools:
  - Bash(gh issue view:*)
  - Bash(gh issue list:*)
  - Bash(gh search issues:*)
  - Write(//tmp/dedupe-payload.json)
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

Your only output is `/tmp/dedupe-payload.json`, holding up to 3 issue numbers,
best match first:

```json
{ "duplicates": [123, 456] }
```

Write nothing else to it. The workflow builds the comment from these numbers
alone, so any prose you produce is discarded. Write the file only when you have
found duplicates: if there are none, or the issue should be skipped per step 2,
write nothing and stop, which is the normal outcome.

The `gh` token here is read-only, so do not try to comment, label, or close
anything yourself.

## Instructions

1. Read the issue: `gh issue view $issue_number --repo $owner_repo --comments`.

2. Stop without writing the file if any of these hold:

   - It is already closed.
   - There is nothing specific to match on: broad product feedback, a support
     question, positive feedback, or a feature idea with no concrete behavior.
   - It is a security report.
   - It already has a comment containing `<!-- issue-dedupe -->`, the marker this
     workflow stamps on its own comments, meaning an earlier run already ran.

3. Summarize the issue in two or three sentences: the observed behavior, the
   component involved, and any distinctive error text. Everything below matches
   against that summary, not the raw body.

4. Search for candidates with
   `gh search issues "<query>" --repo $owner_repo --limit 10 --json number,title,url`.
   Run several searches in parallel covering different angles, since any single
   phrasing misses issues that describe the same bug in other words:

   - the exception type, or the most distinctive fragment of the error message
   - the API or component name (e.g. `mlflow.log_model`, `MlflowClient`)
   - the user-facing symptom in the reporter's own words
   - the same symptom in the vocabulary a maintainer would use
   - the flavor, integration, or backend involved (e.g. `langchain`, `sqlalchemy`)

   Keep each query to two or three distinctive terms. GitHub ANDs every term, so
   a descriptive phrase matches almost nothing: `chart y axis` returns no results
   on this repo while `y axis` returns several. Quote a phrase only when the
   words have to be adjacent, such as an error message.

   Leave `--state` off. It only accepts `open` or `closed`, and omitting it
   searches both, which is what you want: a closed duplicate is still the right
   place to point the reporter. Empty output means no matches, not an error: drop
   the least distinctive term and search again, and move on to the next angle
   only once a two-term query still comes back empty.

5. Filter hard. Keep a candidate only if a maintainer reading both issues would
   close this one as a duplicate. The same component failing a different way is
   not a duplicate, and neither is the same error message from an unrelated cause.
   Precision matters more than recall: a wrong link wastes the reporter's time,
   while a missing one costs nothing. When torn, drop the candidate.
