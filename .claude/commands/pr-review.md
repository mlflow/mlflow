---
allowed-tools: Read, Skill, Bash, Grep, Glob
argument-hint: [extra_context]
description: Review a GitHub pull request and add review comments for issues found
---

# Review Pull Request

Automatically review a GitHub pull request and provide feedback on code quality, style guide violations, and potential bugs.

## Usage

```
/pr-review [extra_context]
```

## Arguments

- `extra_context` (optional): Additional instructions or filtering context (e.g., focus on specific issues or areas)

## Examples

```
/pr-review                                    # Review all changes
/pr-review Please focus on security issues    # Focus on security
/pr-review Only review Python files           # Filter specific file types
/pr-review Check for performance issues       # Focus on specific concern
```

## Important Note

The current local branch may not be the PR branch being reviewed. Always rely on the PR diff fetched via the GitHub API rather than local file contents.

## Instructions

### 1. Auto-detect PR context

- First check for environment variables:
  - If `PR_NUMBER` and `GITHUB_REPOSITORY` are set, parse `GITHUB_REPOSITORY` as `owner/repo` and use `PR_NUMBER`
  - Then use `gh pr view <PR_NUMBER> --repo <owner/repo> --json 'title,body'` to retrieve the PR title and description
- Otherwise:
  - Use `gh pr view --json 'title,body,url,number'` to get PR info for the current branch
  - Parse the output to extract owner, repo, PR number, title, and description
- If neither method works, inform the user that no PR was found and exit

### 2. Fetch PR Diff

- Run the fetch-diff skill to fetch the PR diff:
  ```bash
  uv run .claude/skills/fetch-diff/fetch_diff.py <pr_url>
  ```

### 3. Review Changed Lines

**Apply additional filtering** from user instructions if provided (e.g., focus on specific issues or areas)

Carefully examine **only the changed lines** (added, modified, or deleted) in the diff for:

- Style guide violations (see `.claude/rules/` for language-specific rules)
- Potential bugs and code quality issues
- Common mistakes

**Important**: Ignore unchanged/context lines and pre-existing code.

### 4. Decision Point

- If **no issues found** → Output "No issues found" and exit successfully
- If **issues found** → Continue to step 5

### 5. Add Review Comments

For each issue found, use the `add-review-comment` skill to post review comments.
