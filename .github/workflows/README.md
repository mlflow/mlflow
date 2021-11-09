# GitHub Actions workflows

## Testing

| File                      | Role                                                                 |
| :------------------------ | :------------------------------------------------------------------- |
| `cross-version-tests.yml` | Run cross version tests. See `cross-version-testing.md` for details. |
| `examples.yml`            | Run tests for example scripts & projects                             |
| `master.yml `             | Run unit and integration tests                                       |

## Automation

| File                        | Role                                                      |
| :-------------------------- | :-------------------------------------------------------- |
| `labeling.yml`              | Automatically apply labels on issues and PRs              |
| `notify-dco-failure.yml`    | Notify a DCO check failure                                |
| `release-note-category.yml` | Validate a release-note category label is applied on a PR |
