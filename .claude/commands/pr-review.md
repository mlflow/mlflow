---
allowed-tools: Read, Skill, Bash, Grep, Glob, mcp__review__fetch_diff, mcp__review__add_pr_review_comment
description: Review a GitHub pull request and add review comments for issues found
---

# Review Pull Request

Automatically review a GitHub pull request and provide feedback on code quality, style guide violations, and potential bugs.

## Usage

```
/pr-review
```

## Instructions

### 1. Auto-detect PR context

- First check for environment variables:
  - If `PR_NUMBER` and `GITHUB_REPOSITORY` are set, parse `GITHUB_REPOSITORY` as `owner/repo` and use `PR_NUMBER`
- Otherwise:
  - Use `gh pr view --json url -q '.url'` to get PR info for the current branch and parse to extract owner, repo, and PR number
- If neither method works, inform the user that no PR was found and exit

### 2. Fetch PR Diff

- Use `mcp__review__fetch_diff` tool to fetch the PR diff
- **If reviewing Python files**: Read `dev/guides/python.md` and create a checklist of all style rules with their exceptions before proceeding

### 3. Review Changed Lines

Carefully examine **only the changed lines** (added or modified) in the diff for:

- Style guide violations (using your checklist if Python files)
- Potential bugs and code quality issues
- Common mistakes

**Important**: Ignore unchanged/context lines and pre-existing code.

### 4. Decision Point

- If **no issues found** → Output "No issues found" and exit successfully
- If **issues found** → Continue to step 5

### 5. Add Review Comments

For each issue found, use `mcp__review__add_pr_review_comment` with:

**What to comment on:**

- **Only** lines marked as added (+) or modified in the diff
- Never unchanged context lines or pre-existing code

**How to write comments:**

- Use suggestion blocks (three backticks + "suggestion") for simple fixes that maintainers can apply with one click

  ````
  ```suggestion
  <corrected code here>
  ```
  ````

- Copy original indentation exactly in suggestion blocks
- For repetitive issues, leave one representative comment instead of flagging every instance
- Be specific about the issue and why it needs changing
- For bugs, explain the potential problem and suggested fix clearly
- End each comment with `🤖 Generated with Claude Code`

**Tool parameters:**

- Single-line comment: Set `subject_type` to `line`, specify `line`
- Multi-line comment: Set `subject_type` to `line`, specify both `start_line` and `line`
