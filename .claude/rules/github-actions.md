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

## Prefer `gh` CLI over `actions/github-script`

For simple GitHub API operations (commenting, labeling, cancelling runs, etc.),
use `gh` CLI instead of `actions/github-script`. It avoids the need for
`actions/checkout` and JavaScript boilerplate.

```yaml
# Bad
- uses: actions/checkout@v4
- uses: actions/github-script@v8
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
