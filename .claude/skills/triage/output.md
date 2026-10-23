# Triage output

Create `$out_dir` and write `$out_dir/payload.json` using the selected action's schema.
The issue comment and optional PR body should lead with conclusions for someone who has
read the issue, use runnable examples where helpful, and avoid @-mentions.
Prefer permalinks to file paths:
`https://github.com/<repository>/blob/<sha>/<path>#L<start>-L<end>`, with the SHA from
`git rev-parse HEAD`.

Do not rename, comment on, label, or close the issue, and do not commit, push, or open a PR.
A PR in the payload is a proposal; the workflow publishes it later, so do not claim it
is already open.

After writing and validating the output, restore the checkout to `HEAD`, then stop:

```bash
git reset -q --hard HEAD
git clean -fdq
git status --porcelain --untracked-files=all
```

The last command must print nothing. `fix.patch` and media under `$out_dir` are outside the
checkout and survive this.

## Optional issue title

During initial triage, suggest a concise, specific title only when the current title is
blank or whitespace-only, template-only (e.g. `[BUG]`), or semantically vague (e.g.
`Bug`, `Help`, or `It doesn't work`). Preserve short but informative titles. Derive the
proposal from the reported symptom and affected component, preserve an appropriate
existing issue-type prefix, and do not assert an unverified root cause. This applies
even when reproduction is skipped or needs more information. Omit the proposal when
the report lacks enough information for a useful title. The workflow applies it; do not
rename the issue yourself.

## Optional pull request

Propose a pull request only when all of these hold:

- The `PR_DISABLED` environment variable is not `true`.
- The change leaves no design choice or trade-off unresolved.
- Every changed file is under `mlflow/` or `tests/`.

Also apply the PR eligibility and verification rules for the selected type and action.
If any requirement is unmet, explain the findings in the comment without proposing a PR.

Add regression tests when they add meaningful coverage. Place new regression tests next to
the existing tests for that code, verify that they fail without the fix and pass with it,
and run them together with the directly relevant existing tests, within the
[runtime time budget](runtime.md#time-budget).
Then leave only the fix and any regression tests changed in the checkout (revert anything else
you edited there) and write the patch:

```bash
git add -N mlflow tests
git diff HEAD -- mlflow tests > "$out_dir/fix.patch"
git diff HEAD --stat
```

The last command must list only the files in the patch. Then add `pull_request` to the payload
using `pull_request` from the selected action's payload schema.

For visible UI changes, include a before/after comparison in the PR body.
Stack screenshots vertically with **Before** above **After**, rather than
side-by-side. Capture the broken behavior before applying the fix and the
fixed behavior with the same data, page state, and viewport. Label the captures **Before**
and **After**, save them under `$out_dir/media`, and cite their exact paths as described below.
For a feature request, show the existing workflow or limitation before the change and the
requested behavior after it.

## Media

Save useful screenshots and recordings under `$out_dir/media`. Cite their exact absolute
paths in `comment` or `pull_request.body`, for example
`![Broken UI](/tmp/triage-out/media/broken-ui.png)`. Cite videos on their own line.
The workflow uploads only cited files and rewrites their paths. Do not upload media yourself.

If you wrote media, check its citations before finishing:

```bash
jq -r '.comment, .pull_request.body // ""' "$out_dir/payload.json" |
  uv run --directory .claude/skills --package skills skills embed-media \
    --check --dir "$out_dir/media"
```

Fix any media check errors before finishing.
