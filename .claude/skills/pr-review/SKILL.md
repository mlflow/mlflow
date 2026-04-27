---
disable-model-invocation: true
allowed-tools: Read, Skill, Bash, Grep, Glob
argument-hint: [extra_context]
description: Review a GitHub pull request, add review comments for issues found, and approve if no significant issues exist
---

# Review Pull Request

Automatically review a GitHub pull request across correctness, security, edge cases, efficiency, readability, test coverage, and style. Approves the PR when there are no findings or only MODERATE/NIT findings.

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

You may read unchanged/context lines to understand the change, but only file findings against the changed lines (added, modified, or deleted). Pre-existing code is not in scope, even if it looks suboptimal.

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

Classify each finding by severity (matches `.github/instructions/code-review.instructions.md`):

| Severity | Emoji | Use for                                                                          |
| -------- | ----- | -------------------------------------------------------------------------------- |
| CRITICAL | 🔴    | bugs, logic errors, security issues, data loss risk, broken public API           |
| MODERATE | 🟡    | non-blocking quality concerns where the code works but could be clearer or safer |
| NIT      | 🟢    | pure style/preference the author can ignore                                      |

Then:

- **No findings** -> skip to step 6 (approve)
- **Only MODERATE/NIT findings** -> step 5 (add comments), then step 6 (approve)
- **Any CRITICAL finding** -> step 5 (add comments); do NOT approve

### 5. Add Review Comments

For each finding, use the `add-review-comment` skill. One comment per distinct finding, anchored to the most relevant changed line. For repeated identical issues, leave a single representative comment rather than flagging every instance.

Every comment MUST use this exact format: `<emoji> **<severity>:** <description>`

Keep comments constructive and specific: state the problem, why it matters, and a concrete suggestion when possible.

### 6. Approve the PR

Approve the PR when there are no findings or only MODERATE/NIT findings, but **only if the PR author has the `admin` or `maintain` role**.

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
