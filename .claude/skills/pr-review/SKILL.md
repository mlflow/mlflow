---
name: pr-review
description: Review a pull request and emit a validated review payload.
disable-model-invocation: true
argument-hint: "<pr_url>"
arguments: [pr_url]
---

# Review Pull Request

Review $pr_url and write a JSON review payload to `/tmp/review-payload.json`. Do not post anything:
writing that payload is the whole job.

## Instructions

The commands below take `<pr_url>` from the PR URL above, and `<owner>`, `<repo>`, and
`<pr_number>` from its parts.

### 1. Gather context (run in parallel)

These reads are independent. Issue them as parallel tool calls in a single turn, not sequentially.

**PR title and description**

```bash
gh pr view <pr_url> --json title,body
```

**PR diff hunks** via the [`fetch-diff`](../fetch-diff/SKILL.md) skill:

```bash
uv run --package skills skills fetch-diff <pr_url>
```

Its annotated output gives you the `line` and `side` to anchor each comment on.

**Existing review threads**, so you can avoid duplicating prior feedback. Up to 100 threads (open,
resolved, and outdated) with up to 20 comments each:

```bash
gh api graphql -F owner=<owner> -F repo=<repo> -F pr=<pr_number> \
  --jq '.data.repository.pullRequest.reviewThreads.nodes | map(.comments = .comments.nodes)' \
  -f query='
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

### 2. Load repository style rules

Load the repository style rules applicable to the changed files:

```bash
git diff --name-only HEAD^1 | uv run --package skills skills load-rules
```

### 3. Analyze the change

The working tree holds the PR merged into the base (`refs/pull/<pr_number>/merge`), so file contents
reflect the post-merge state. Explore it for context beyond the diff (existing patterns, call sites
of changed symbols, file conventions).

The merge ref's base parent is reachable as `HEAD^1`. When the diff doesn't show enough (verifying
a refactor preserved behavior, reading a masked deleted file, or seeing the pre-change version of a
heavily modified one), use `git show HEAD^1:<path>` rather than re-fetching the file over the API.
The checkout is shallow, so nothing older than `HEAD^1` exists: `git log` and `git blame` stop at
the shallow boundary rather than reaching the commit that actually introduced a line. Neither
errors, so don't trust them for pre-change history.

Verify rather than infer. A `grep` through the installed package, a `uv run python -c '...'`, or a
quick search and fetch of the upstream docs will settle most questions in seconds, and an unverified
finding should be dropped rather than hedged.

Node and `agent-browser` are on PATH for docs and UI changes. Render when it settles whether the
change is correct, or when a capture shows a finding more plainly than prose can. Building the
docs site or the UI is expensive; do it only when the finding justifies it. Capture to an
absolute path named for what it shows:
`agent-browser screenshot --full /tmp/review-media/traces-table.png`, and cite that name in a
finding.

Evaluate the changed code across these dimensions:

- **Correctness**: logic errors, off-by-one, incorrect API usage, broken invariants, regressions in behavior
- **Security**: injection, unsafe deserialization, secret leakage, missing authz/authn, unsafe defaults
- **Edge cases**: None/empty/zero inputs, concurrency, error paths, retries, large/unicode inputs
- **Efficiency**: needless N+1 queries, redundant work in hot paths, allocations in tight loops
- **Readability & maintainability**: unclear names, dead code, premature abstractions, comments that restate the code
- **Test coverage**: new behavior lacks tests, tests assert on the wrong thing, mocks hide real failures
- **Style guide**: violations of the rules loaded in step 2

#### Don't comment on

- **Pre-existing code.** You may read unchanged/context lines to understand the change, but only
  file findings against the changed lines (added, modified, or deleted), even if surrounding code
  looks suboptimal.
- **Anything a formatter or linter already catches**: unused imports, formatting, line length,
  simple typos.
- **Unfamiliar names and values.** Model names, runner types, library versions, and dates that
  postdate your training data are new, not wrong.
- **Hypothetical edge cases.** If the finding needs "while unlikely", "could potentially", or "edge
  case where" to stand up, skip it. Only flag what realistically happens.
- **Naming preferences.** Flag a name only when it is actively misleading.
- **One-off literals.** Don't ask for a constant to be extracted for a single use site.

#### Severity

Classify each finding that survives those exclusions:

- **CRITICAL**: bugs, logic errors, security issues, data loss risk, broken public API.
- **MODERATE**: non-blocking quality concerns where the code works but could be clearer or safer.
- **NIT**: pure style/preference the author can ignore.

### 4. Write and validate the review payload

Read [`review-payload.schema.json`](./review-payload.schema.json), then write
`/tmp/review-payload.json` matching it. It defines the severity prefix each comment body carries
and derives `event` from those prefixes.

Authoring rules not captured by the schema:

- One comment per distinct finding, anchored to the most relevant changed line. For repeated
  identical issues, leave a single representative comment rather than flagging every instance.
- For findings about out-of-diff code, anchor to any changed line (prefer the same file when it has
  hunks) and name the actual `path:line` in the body.
- Keep comments short: state the problem, why it matters, and a concrete fix in roughly three
  sentences plus an optional suggestion block. Cut the investigation trail (commands you ran,
  alternatives you weighed, evidence for a claim the reader can check in one look) and anything the
  suggestion block already shows.
- Use suggestion blocks for simple fixes: fence with ` ```suggestion ` and preserve original
  indentation.
- If you have no findings, emit an empty `comments` array.
- To attach an image or video (a diagram, a chart, a captured repro), write the file into
  `/tmp/review-media/` and reference it by bare filename: `![desc](name.png)` to embed, or
  `[desc](name.png)` to link. A later workflow step uploads it and rewrites the reference to
  a URL. Do not upload anything yourself. Skip this unless a visual genuinely beats prose;
  most reviews need none.
- Put a video reference (`.mp4`, `.mov`, `.webm`) on a line of its own. GitHub renders a
  player only for a bare URL in its own paragraph, so a video cited mid-sentence falls back
  to a plain link.

Validate before finishing, then fix any errors and re-emit until both of these pass:

```bash
uv run --package skills skills validate-review /tmp/review-payload.json
uv run --package skills skills upload-media --check \
  --dir /tmp/review-media --target /tmp/review-payload.json
```

Do not post the review: no `gh pr review`, no review/comment APIs, no other skills. Stop
after writing and validating `/tmp/review-payload.json`.
