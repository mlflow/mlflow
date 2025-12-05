---
allowed-tools: Read, Skill, Bash, Grep, Glob, mcp__review__fetch_diff, mcp__review__add_pr_review_comment
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

- Use `mcp__review__fetch_diff` tool to fetch the PR diff
- **If reviewing Python files**: Read `dev/guides/python.md` and create a checklist of all style rules with their exceptions before proceeding

### 3. Review Changed Lines

**Apply additional filtering** from user instructions if provided (e.g., focus on specific issues or areas)

Carefully examine **only the changed lines** (added or modified) in the diff for:

- Style guide violations (using your checklist if Python files)
- Potential bugs and code quality issues
- Common mistakes

**Workspace awareness reminder**: If the diff touches the SQLAlchemy tracking store or other tracking persistence layers, verify that workspace-aware behavior remains intact and that new functionality includes matching workspace-aware tests (for example, additions in `tests/store/tracking/test_sqlalchemy_store_workspace.py`).

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
