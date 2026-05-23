---
paths: ".github/workflows/**/*.yml"
---

# GitHub Actions Workflow Guidelines

## Use `ubuntu-slim` for Lightweight Tasks

Prefer `ubuntu-slim` over `ubuntu-latest` for simple jobs (e.g., labeling, commenting, notifications).

Note: `ubuntu-slim` has a 15-minute timeout limit. Use `ubuntu-latest` for long-running jobs (e.g., polling).

```yaml
# Bad
runs-on: ubuntu-latest

# Good
runs-on: ubuntu-slim
```

## Use Workflow Context Instead of Fetching

If the trigger event already carries the data, read it from the `github` context instead of calling `gh` or `actions/github-script`. Extra API calls burn rate-limit budget and add a flaky network hop for nothing.

```yaml
# Bad
- env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    PR_NUMBER=$(gh pr view ${{ github.event.pull_request.html_url }} --json number -q .number)

# Good
- env:
    PR_NUMBER: ${{ github.event.pull_request.number }}
  run: echo "PR #$PR_NUMBER"
```

Only fetch when the data isn't in the payload (e.g., check runs, review threads, changed files on `issue_comment`).

## Prefer `gh` CLI over `actions/github-script`

For simple GitHub API operations (commenting, labeling, cancelling runs, etc.),
use `gh` CLI instead of `actions/github-script`. It avoids the need for
`actions/checkout` and JavaScript boilerplate.

```yaml
# Bad
- uses: actions/checkout@...
- uses: actions/github-script@...
  with:
    script: |
      const script = require(".github/workflows/my-script.js");
      await script({ context, github });

# Good
- env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    gh pr comment ...
```

## Use `sparse-checkout` When Only a Subset of Files Is Needed

When a workflow only needs a small subset of the repo (e.g., a single script under `.github/`), pass `sparse-checkout` to `actions/checkout` instead of cloning the whole tree. A full checkout of this repo takes around 10 seconds on average; a sparse checkout finishes in a fraction of that.

```yaml
# Bad: clones the entire repo just to run one script
- uses: actions/checkout@...
- run: bash .github/scripts/my-script.sh

# Good: only fetches what the job actually reads
- uses: actions/checkout@...
  with:
    sparse-checkout: |
      .github/scripts/my-script.sh
    sparse-checkout-cone-mode: false
- run: bash .github/scripts/my-script.sh
```

When listing directories, leave cone mode on (the default):

```yaml
- uses: actions/checkout@...
  with:
    sparse-checkout: |
      .github/scripts
      dev
```

Set `sparse-checkout-cone-mode: false` only when you need to target individual files or non-prefix glob patterns.

## `pipefail` Is Already On

Every workflow in this repo sets top-level `defaults.run.shell: bash` (enforced by [`.github/policy.rego`](../../.github/policy.rego)). GitHub Actions runs `shell: bash` as `bash --noprofile --norc -eo pipefail {0}`, so `pipefail` is already enabled. Don't ask for `set -o pipefail` in workflow `run:` steps. ([docs](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#defaultsrunshell))
