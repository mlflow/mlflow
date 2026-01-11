---
name: fetch-unresolved-comments
description: Fetch unresolved PR review comments using GitHub GraphQL API, filtering out resolved and outdated feedback.
allowed-tools:
  - Bash(uv run skills fetch-unresolved-comments:*)
---

# Fetch Unresolved PR Review Comments

Uses GitHub's GraphQL API to fetch only unresolved review thread comments from a pull request.

## When to Use

- You need to get only unresolved review comments from a PR
- You want to filter out already-resolved and outdated feedback

## Instructions

1. **Get PR URL**:

   - First check for environment variables:
     - If `PR_NUMBER` and `GITHUB_REPOSITORY` are set, construct URL as `https://github.com/${GITHUB_REPOSITORY}/pull/${PR_NUMBER}`
   - Otherwise:
     - Use `gh pr view --json url -q '.url'` to get the current branch's PR URL

2. **Run the skill**:

   ```bash
   uv run skills fetch-unresolved-comments <pr_url>
   ```

   Example:

   ```bash
   uv run skills fetch-unresolved-comments https://github.com/mlflow/mlflow/pull/18327
   ```

   The script automatically reads the GitHub token from:

   - `GITHUB_TOKEN` or `GH_TOKEN` environment variables, or
   - `gh auth token` command if environment variables are not set

## Example Output

```json
{
  "total": 3,
  "by_file": {
    ".github/workflows/resolve.yml": [
      {
        "thread_id": "PRRT_kwDOAL...",
        "line": 40,
        "startLine": null,
        "diffHunk": "@@ -0,0 +1,245 @@\n+name: resolve...",
        "comments": [
          {
            "id": 2437935275,
            "body": "We can remove this once we get the key.",
            "author": "harupy",
            "createdAt": "2025-10-17T00:53:20Z"
          },
          {
            "id": 2437935276,
            "body": "Good catch, I'll update it.",
            "author": "contributor",
            "createdAt": "2025-10-17T01:10:15Z"
          }
        ]
      }
    ],
    ".gitignore": [
      {
        "thread_id": "PRRT_kwDOAL...",
        "line": 133,
        "startLine": null,
        "diffHunk": "@@ -130,0 +133,2 @@\n+.claude/*",
        "comments": [
          {
            "id": 2437935280,
            "body": "Should we add this to .gitignore?",
            "author": "reviewer",
            "createdAt": "2025-10-17T01:15:42Z"
          }
        ]
      }
    ]
  }
}
```
