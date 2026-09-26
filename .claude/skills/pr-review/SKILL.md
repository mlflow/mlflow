---
name: pr-review
description: Review a pull request and emit a validated review payload.
disable-model-invocation: true
argument-hint: "<pr_url> <pr_checkout> <payload_path> <media_dir> <base_dir>"
arguments: [pr_url, pr_checkout, payload_path, media_dir, base_dir]
---

# Review Pull Request

Review $pr_url and write a JSON review payload to $payload_path. Do not post anything: writing
that payload is the whole job.

## The reviewed tree

The PR is checked out at `$pr_checkout`, not in the working directory:

```text
$base_dir             # the working directory: this skill and the `skills`
                      # CLI come from here, and nothing you review does
$pr_checkout          # the reviewed tree: the PR merged into its base
```

The working directory holds a checkout of the same repository, so it looks like the code under
review and is not guaranteed to match it. Everything aimed at the PR needs the prefix:
`git -C $pr_checkout ...`, `$pr_checkout/<path>` to open or grep a file, and
`cd $pr_checkout && ...` for anything that builds or runs repository code. The
`uv run` commands below are the exception: `--directory $base_dir` pins each one to
this checkout, so it keeps using this tree's `skills` package and rules even when the
working directory has moved into `$pr_checkout`. uv resolves its workspace from the
working directory, and so does the rule loader, so dropping the flag silently hands
both to the code under review.

## Instructions

The commands below take `<pr_url>` from the PR URL above, and `<owner>`, `<repo>`, and
`<pr_number>` from its parts.

### 1. Gather context (run in parallel)

These reads are independent. Issue them as parallel tool calls in a single turn, not sequentially.

**PR title and description**

```bash
gh pr view <pr_url> --json title,body
```

**PR diff hunks**. `$pr_checkout` holds the merge ref (see step 3), so `HEAD^1 HEAD` is exactly
the PR diff:

```bash
git -C $pr_checkout diff HEAD^1 HEAD | uv run --directory $base_dir --package skills skills annotate-diff
```

Each line comes back as `old_line new_line | <marker> content`, which gives you the `line` and
`side` to anchor each comment on: `-` is `side=LEFT` at `old_line`, `+` is `side=RIGHT` at
`new_line`, and an unmarked context line is `side=RIGHT` at `new_line`. Pass `--help` for the rest.

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
git -C $pr_checkout diff --name-only HEAD^1 | uv run --directory $base_dir --package skills skills load-rules
```

### 3. Analyze the change

`$pr_checkout` holds the PR merged into the base (`refs/pull/<pr_number>/merge`), so its file
contents reflect the post-merge state. Explore it for context beyond the diff (existing patterns,
call sites of changed symbols, file conventions), scoping every search to that directory.

The merge ref's base parent is reachable as `HEAD^1`. When the diff doesn't show enough (verifying
a refactor preserved behavior, reading a masked deleted file, or seeing the pre-change version of a
heavily modified one), use `git -C $pr_checkout show HEAD^1:<path>` rather than re-fetching the file
over the API.
The checkout is shallow, so nothing older than `HEAD^1` exists: `git log` and `git blame` stop at
the shallow boundary rather than reaching the commit that actually introduced a line. Neither
errors, so don't trust them for pre-change history.

Verify rather than infer. A `grep` through the installed package, a `uv run python -c '...'`, a web
fetch, or a web search (`$base_dir/.claude/skills/pr-review/search-web.sh "<query>"`) will settle
most questions in seconds, and an unverified finding should be dropped rather than hedged. When the
cheap checks don't settle it, escalate to the expensive ones: build the docs site, build and boot
the UI, start the backend.

Node and `agent-browser` are on PATH for docs and UI changes. Capture to an absolute path named for
what it shows: `agent-browser screenshot --full $media_dir/example.png`, and cite that same
path in a finding.

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
`$payload_path` matching it. It defines the severity prefix each comment body carries
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
- To attach an image or video (a diagram, a chart, a captured repro), write the file into
  `$media_dir` and cite it by the absolute path you wrote it to:
  `![desc]($media_dir/name.png)` to embed, or `[desc]($media_dir/name.png)`
  to link. A later workflow step uploads it and rewrites the reference to a URL. Do not
  upload anything yourself. Skip this unless a visual genuinely beats prose; most reviews
  need none.
- Put a video reference (`.mp4`, `.mov`, `.webm`) on a line of its own. GitHub renders a
  player only for a bare URL in its own paragraph, so a video cited mid-sentence falls back
  to a plain link.

Validate before finishing, then fix any errors and re-emit until both of these pass:

```bash
uv run --directory $base_dir --package skills skills validate-review $payload_path
# only when you wrote a file into $media_dir
uv run --directory $base_dir --package skills skills embed-media --check --dir $media_dir --target $payload_path
```

Do not post the review: no `gh pr review`, no review/comment APIs, no other skills. Stop
after writing and validating `$payload_path`.
