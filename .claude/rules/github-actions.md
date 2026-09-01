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
    PR_NUMBER: ${{ github.event.pull_request.number }}
  run: |
    HEAD_SHA=$(gh pr view "$PR_NUMBER" --json headRefOid -q .headRefOid)

# Good
- env:
    HEAD_SHA: ${{ github.event.pull_request.head.sha }}
  run: echo "$HEAD_SHA"
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

## Prefer the Shared Setup Actions

Set up toolchains through the composite actions in `.github/actions/` rather than
calling the upstream actions directly. Each one centralizes the pinned version and
the env defaults every job expects, so an upstream bump stays a one-line change
instead of a repo-wide sweep.

| Toolchain       | Action                           |
| --------------- | -------------------------------- |
| Python and `uv` | `./.github/actions/setup-python` |
| Node            | `./.github/actions/setup-node`   |
| Java            | `./.github/actions/setup-java`   |

```yaml
# Bad: a second copy of the pin to keep in sync, and no uv or env defaults
- uses: actions/setup-python@...

# Good
- uses: ./.github/actions/setup-python
```

Local action paths resolve relative to `$GITHUB_WORKSPACE`, so a workflow that
checks the repo out into a subdirectory needs that prefix
(`./mlflow/.github/actions/setup-python`). Never point it at a PR head checkout:
that runs author-controlled code with whatever secrets the workflow holds.

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

## Mask Secrets Generated Mid-Job

Values that come from `secrets.*` are masked automatically. Anything a step
mints at runtime (an OAuth token exchanged over `curl`, a random passphrase, a
value decoded out of another secret) is not, so the first command that echoes it
prints it in clear text. Emit
[`::add-mask::`](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands#masking-a-value-in-a-log)
the moment the value exists, before it reaches `$GITHUB_OUTPUT` or any other
command.

```yaml
# Bad: nothing stops a later command from printing the token
- run: |
    TOKEN=$(curl -sS ... | jq -r .access_token)
    echo "token=$TOKEN" >> "$GITHUB_OUTPUT"

# Good
- run: |
    TOKEN=$(curl -sS ... | jq -r .access_token)
    echo "::add-mask::$TOKEN"
    echo "token=$TOKEN" >> "$GITHUB_OUTPUT"
```

`set -x` needs separate care: xtrace prints the assignment itself, so the value
is in the log before the mask can register. Don't enable it in a step that mints
a secret.

The mask covers only the job that registered it, and a masked value cannot be
handed to another job through job-level `outputs`: GitHub redacts it on the
runner. Mint and mask it again in the job that needs it.

## Never Write Secrets to `$GITHUB_ENV`

`$GITHUB_ENV` promotes a value into the environment of every later step in the
job, and of every process those steps spawn (third-party actions, `npm install`
lifecycle scripts, the test suite). Hand it to the consuming step through
`$GITHUB_OUTPUT` and a step-level `env:` block instead, so a step that has no
use for the secret never holds it.

```yaml
# Bad: the test step inherits a token it never asked for
- id: auth
  run: |
    TOKEN=$(curl -sS ... | jq -r .access_token)
    echo "::add-mask::$TOKEN"
    echo "TOKEN=$TOKEN" >> "$GITHUB_ENV"
- run: uv run dev/deploy.py # needs $TOKEN
- run: uv run pytest tests/ # doesn't, but gets it anyway

# Good: the token reaches only the step that consumes it
- id: auth
  run: |
    TOKEN=$(curl -sS ... | jq -r .access_token)
    echo "::add-mask::$TOKEN"
    echo "token=$TOKEN" >> "$GITHUB_OUTPUT"
- env:
    TOKEN: ${{ steps.auth.outputs.token }}
  run: uv run dev/deploy.py
- run: uv run pytest tests/ # no $TOKEN in its environment
```

The same holds for a secret that comes straight from `secrets.*`: put it in the
consuming step's `env:` block rather than exporting it job-wide.
