---
name: pr-review
description: Review a GitHub pull request and emit a single review payload (comments + approval decision) for the workflow to validate and post
disable-model-invocation: true
allowed-tools:
  - Read
  - Skill
  - Bash
  - Grep
  - Glob
  - Agent
  - Edit(//tmp/review-payload.json)
argument-hint: "<owner_repo> <pr_number> [extra_context]"
arguments: [owner_repo, pr_number, extra_context]
---

# Review Pull Request

Automatically review a GitHub pull request across correctness, security, edge cases, efficiency, readability, test coverage, and style. Emits a single `/tmp/review-payload.json` that the workflow validates against [`review-payload.schema.json`](./review-payload.schema.json) and posts via `POST /repos/{owner}/{repo}/pulls/{pull_number}/reviews`. Sets the payload's `event` to `APPROVE` when there are no findings or only MODERATE/NIT findings.

## Usage

```
/pr-review <owner_repo> <pr_number> [extra_context]
```

## Arguments

- `<owner_repo>` (required): repository slug, e.g. `mlflow/mlflow`
- `<pr_number>` (required): pull request number
- `[extra_context]` (optional): additional filtering or focus instructions (e.g., a specific concern or file type)

## Examples

```
/pr-review mlflow/mlflow 23320
/pr-review mlflow/mlflow 23320 Please focus on security issues
/pr-review mlflow/mlflow 23320 Only review Python files
```

## Inputs

This invocation is reviewing:

- Owner/Repo: `$owner_repo`
- PR number: `$pr_number`
- Extra context: `$extra_context`

The `<owner>`/`<repo>`/`<pr_number>` placeholders in the steps below refer to the values above (split `$owner_repo` on `/` for `<owner>` and `<repo>`).

## Instructions

### 1. Fetch PR context

Fetch the PR title, description, and author:

```bash
gh pr view <pr_number> --repo "<owner>/<repo>" --json title,body,author
```

> **Note:** You have a **read-only** GitHub token. Do NOT call write APIs (`gh api ... POST`, `gh pr review`, `gh pr comment`, etc.).

### 2. Fetch PR Diff

Fetch the diff hunks via the `fetch-diff` skill. For context beyond the diff (existing patterns, call sites of changed symbols, file conventions), `Read` and `Grep` the working tree, which holds the PR merged into the base (`refs/pull/<pr>/merge`), so file contents reflect the post-merge state.

### 3. Fetch Existing Review Comments

Fetch up to 100 review threads on the PR (open, resolved, and outdated, with up to 20 comments each) so you can avoid duplicating prior feedback:

```bash
gh api graphql -F owner=<owner> -F repo=<repo> -F pr=<pr_number> -f query='
  query($owner: String!, $repo: String!, $pr: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        reviewThreads(first: 100) {
          nodes {
            isResolved
            isOutdated
            path
            line
            comments(first: 20) {
              nodes { author { login } body }
            }
          }
        }
      }
    }
  }'
```

### 4. In-Depth Analysis

#### Don't comment on

- Pre-existing code. You may read unchanged/context lines to understand the change, but only file findings against the changed lines (added, modified, or deleted), even if surrounding code looks suboptimal.
- Issues already caught by formatters or linters (unused imports, formatting, line length, simple typos, etc.).

Evaluate the changed code across these dimensions:

- **Correctness**: logic errors, off-by-one, incorrect API usage, broken invariants, regressions in behavior
- **Security**: injection, unsafe deserialization, secret leakage, missing authz/authn, unsafe defaults
- **Edge cases**: None/empty/zero inputs, concurrency, error paths, retries, large/unicode inputs
- **Efficiency**: needless N+1 queries, redundant work in hot paths, allocations in tight loops
- **Readability & maintainability**: unclear names, dead code, premature abstractions, comments that restate the code
- **Test coverage**: new behavior lacks tests, tests assert on the wrong thing, mocks hide real failures
- **Style guide**: see `.claude/rules/` for language-specific rules and `CLAUDE.md` for repo conventions

**Workspace awareness reminder**: If the diff touches the SQLAlchemy tracking store or other tracking persistence layers, verify workspace-aware behavior remains intact and that new functionality includes matching workspace-aware tests (e.g., additions in `tests/store/tracking/test_sqlalchemy_store_workspace.py`).

### 5. Decision Point

Classify each finding by severity (matches `.github/instructions/code-review.instructions.md`):

| Severity | Emoji | Use for                                                                          |
| -------- | ----- | -------------------------------------------------------------------------------- |
| CRITICAL | 🔴    | bugs, logic errors, security issues, data loss risk, broken public API           |
| MODERATE | 🟡    | non-blocking quality concerns where the code works but could be clearer or safer |
| NIT      | 🟢    | pure style/preference the author can ignore                                      |

Determine the review `event`:

- **No CRITICAL findings AND author has `admin`/`maintain` role** -> `event: "APPROVE"`
- **Any CRITICAL finding, OR author role is anything else (or the API errors, e.g., 404 for non-collaborators)** -> `event: "COMMENT"`. Do not mention the reason for not approving in the review body.

Check the author's role (use the `author.login` from step 1 as `<author>`):

```bash
gh api repos/<owner>/<repo>/collaborators/<author>/permission --jq '.role_name'
```

### 6. Emit Review Payload

Read [`review-payload.schema.json`](./review-payload.schema.json) for the full payload spec (field types, required fields, patterns, enums) and write `/tmp/review-payload.json` matching it.

Authoring rules not captured by the schema:

- One comment per distinct finding, anchored to the most relevant changed line. For repeated identical issues, leave a single representative comment rather than flagging every instance.
- Keep comments constructive and specific: state the problem, why it matters, and a concrete suggestion when possible.
- Use suggestion blocks for simple fixes — fence with ` ```suggestion ` and preserve original indentation.
- If you have no findings, emit an empty `comments` array.

Validate before finishing — fix any errors and re-emit until this passes:

```bash
uv run --package skills skills validate-review /tmp/review-payload.json
```
