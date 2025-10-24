---
allowed-tools: Skill, Read, Edit, Write, Glob, Grep, Bash
argument-hint: [extra_context]
description: Resolve PR review comments by fetching unresolved feedback and making necessary code changes
---

# Resolve PR Review Comments

Automatically fetch and address PR review comments. This command examines review feedback and makes necessary code changes to resolve the issues.

## Usage

```
/resolve [extra_context]
```

## Arguments

- `extra_context` (optional): Additional instructions or filtering context (e.g., focus on specific files or issue types)

## Examples

```
/resolve                                           # Address all unresolved PR comments
/resolve Focus on type hint issues only            # Filter specific comment types
/resolve Skip comments from automated bots         # Apply custom filtering
/resolve Only resolve comments in mlflow/tracking/ # Focus on specific directories
```

## Instructions

1. **Auto-detect PR context**:

   - First check for environment variables:
     - If `PR_NUMBER` and `GITHUB_REPOSITORY` are set, read them and parse `GITHUB_REPOSITORY` as `owner/repo` and use `PR_NUMBER` directly
   - Otherwise:
     - Use `gh pr view --json url -q '.url'` to get PR info for the current branch and parse to extract owner, repo, and PR number
   - If neither method works, inform the user that no PR was found and exit

2. **Fetch unresolved review comments**:

   - Invoke the `fetch-unresolved-comments` skill to get only unresolved review threads
   - If no unresolved comments are found, inform the user and exit

3. **Apply additional filtering** from user instructions if provided (e.g., focus on specific files or issue types)

4. For each unresolved review comment:

   - Read the file and surrounding code for context
   - Make minimal, precise changes to address the feedback
   - Follow project style guides (Python: see `dev/guides/python.md`)

5. After making all changes, commit them:

   - Stage changes: `git add .`
   - Create DCO-signed commit with this exact format:

     ```bash
     git commit -s -m "Address PR review comments

     🤖 Generated with Claude Code

     Co-Authored-By: Claude <noreply@anthropic.com>"
     ```

   - Handle any pre-commit hook failures (they may auto-fix formatting)
   - **DO NOT PUSH** the changes; just commit them locally
