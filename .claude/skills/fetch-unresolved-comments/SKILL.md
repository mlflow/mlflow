---
name: fetch-unresolved-comments
description: Fetch unresolved PR review comments using GitHub GraphQL API, filtering out resolved and outdated feedback.
---

# Fetch Unresolved PR Review Comments

Uses GitHub's GraphQL API to fetch only unresolved review thread comments from a pull request.

## When to Use

- You need to get only unresolved review comments from a PR
- You want to filter out already-resolved and outdated feedback

## Instructions

1. **Parse PR information**:

   - First check for environment variables:
     - If `PR_NUMBER` and `GITHUB_REPOSITORY` are set, read them and parse `GITHUB_REPOSITORY` as `owner/repo` and use `PR_NUMBER` directly
   - Otherwise:
     - Use `gh pr view --json url -q '.url'` to get the current branch's PR URL and parse to extract owner, repo, and PR number

2. **Run the Python script**:

   ```bash
   GITHUB_TOKEN=$(gh auth token) \
       uv run python .claude/skills/fetch-unresolved-comments/fetch_unresolved_comments.py <owner> <repo> <pr_number>
   ```

3. **Script options**:

   - `--token <token>`: Provide token explicitly (default: GITHUB_TOKEN or GH_TOKEN env var)

4. **Parse the JSON output**:
   The script always outputs JSON with:
   - `total`: Total number of unresolved comments across all threads
   - `by_file`: Review threads grouped by file path (each thread contains multiple comments in a conversation)

## Example JSON Output

```json
{
  "total": 3,
  "by_file": {
    ".github/workflows/resolve.yml": [
      {
        "thread_id": "PRRT_kwDOAL...",
        "isOutdated": false,
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
        "isOutdated": false,
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
