---
disable-model-invocation: true
allowed-tools: Read, Skill, Bash, Grep, Glob
argument-hint: [extra_context]
description: Review a GitHub pull request, add review comments for issues found, and approve if no significant issues exist
---

# Review Pull Request

Automatically review a GitHub pull request and provide feedback on code quality, style guide violations, and potential bugs. Approves the PR if no significant issues are found.

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

The current local branch may not be the PR branch being reviewed. Always rely on the PR diff fetched via the `fetch-diff` skill.

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

Run the `fetch-diff` skill to fetch the PR diff for the identified PR.

### 3. In-Depth Analysis

**Apply additional filtering** from user instructions if provided (e.g., focus on specific issues or areas).

Carefully examine **only the changed lines** (added, modified, or deleted) in the diff. Ignore unchanged/context lines and pre-existing code; they are not in scope, even if they look suboptimal.

Evaluate the changed code across these dimensions:

- **Correctness**: logic errors, off-by-one, incorrect API usage, broken invariants, regressions in behavior
- **Security**: injection, unsafe deserialization, secret leakage, missing authz/authn, unsafe defaults
- **Edge cases**: None/empty/zero inputs, concurrency, error paths, retries, large/unicode inputs
- **Efficiency**: needless N+1 queries, redundant work in hot paths, allocations in tight loops
- **Readability & maintainability**: unclear names, dead code, premature abstractions, comments that restate the code
- **Test coverage**: new behavior lacks tests, tests assert on the wrong thing, mocks hide real failures
- **Style guide**: see `.claude/rules/` for language-specific rules and `CLAUDE.md` for repo conventions

**Workspace awareness reminder**: If the diff touches the SQLAlchemy tracking store or other tracking persistence layers, verify workspace-aware behavior remains intact and that new functionality includes matching workspace-aware tests (e.g., additions in `tests/store/tracking/test_sqlalchemy_store_workspace.py`).

### 4. Decision Point

Classify each finding by severity:

- **Critical**: bugs, logic errors, security issues, data loss risk, broken public API, missing tests for new behavior
- **Improvement**: non-blocking quality concerns where the code works but could be clearer or safer
- **Nitpick**: pure style/preference; prefix the comment with `nit:` so the author can ignore it

Then:

- **No findings** → skip to step 6 (approve)
- **Only improvements/nitpicks** → step 5 (add comments), then step 6 (approve)
- **Any critical finding** → step 5 (add comments); do NOT approve

### 5. Add Review Comments

For each finding, use the `add-review-comment` skill. One comment per issue, anchored to the most relevant changed line. Keep comments constructive and specific: state the problem, why it matters, and a concrete suggestion when possible. Prefix nitpicks with `nit:`.

### 6. Approve the PR

Approve the PR when there are no issues or only minor issues, but **only if the PR author has the `admin` or `maintain` role**.

First, check the PR author's role:

```bash
author=$(gh api repos/<owner>/<repo>/pulls/<PR_NUMBER> --jq '.user.login')
gh api repos/<owner>/<repo>/collaborators/"$author"/permission --jq '.role_name'
```

- If the role is `admin` or `maintain` -> approve the PR:
  ```bash
  gh pr review <PR_NUMBER> --repo <owner/repo> --approve
  ```
- Otherwise (including API errors, e.g., 404 for non-collaborators) -> do NOT approve. Do not mention the reason for not approving in the review.
