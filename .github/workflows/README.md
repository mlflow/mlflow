# GitHub Actions workflows

## Testing

| File                      | Role                                                                 |
| :------------------------ | :------------------------------------------------------------------- |
| `cross-version-tests.yml` | Run cross version tests. See `cross-version-testing.md` for details. |
| `examples.yml`            | Run tests for example scripts & projects                             |
| `master.yml `             | Run unit and integration tests                                       |

> Note: This fork intentionally omits upstream-only governance workflows
> (maintainer approval, release-note validation, automatic patch labeling,
> autoformat, team-review helpers, etc.) because they depend on MLflow's
> internal GitHub App credentials and Databricks review teams.
