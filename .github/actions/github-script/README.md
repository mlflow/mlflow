# GitHub Script Composite Action

This is a composite action that wraps [actions/github-script](https://github.com/actions/github-script) to provide a consistent interface across the MLflow repository.

## Usage

### Basic usage with inline script

```yaml
- uses: ./.github/actions/github-script
  with:
    script: |
      github.rest.issues.createComment({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: context.issue.number,
        body: '👋 Thanks for reporting!'
      })
```

### Using a script file

```yaml
- uses: ./.github/actions/github-script
  with:
    script-file: ./.github/workflows/my-script.js
```

### Passing script parameters

```yaml
- uses: ./.github/actions/github-script
  with:
    script: |
      const script = require('./.github/workflows/my-script.js');
      await script({ context, github });
```

## Inputs

| Name                        | Description                                    | Required | Default               |
| --------------------------- | ---------------------------------------------- | -------- | --------------------- |
| `script`                    | Script to execute                              | No       | N/A                   |
| `script-file`               | Path to the script file to execute             | No       | N/A                   |
| `github-token`              | GitHub token                                   | No       | `${{ github.token }}` |
| `debug`                     | Whether to show debug logs                     | No       | N/A                   |
| `result-encoding`           | Encoding of the result (either json or string) | No       | N/A                   |
| `retries`                   | Number of times to retry a request             | No       | N/A                   |
| `retry-exempt-status-codes` | List of status codes that will not be retried  | No       | N/A                   |

## Notes

This action is a thin wrapper around actions/github-script@60a0d83039c74a4aee543508d2ffcb1c3799cdea (v7.0.1).
