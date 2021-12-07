# GitHub Actions workflows

## Testing

| File                      | Role                                                                 |
| :------------------------ | :------------------------------------------------------------------- |
| `cross-version-tests.yml` | Run cross version tests. See `cross-version-testing.md` for details. |
| `examples.yml`            | Run tests for example scripts & projects                             |
| `master.yml `             | Run unit and integration tests                                       |

## Automation

| File                        | Role                                                           |
| :-------------------------- | :------------------------------------------------------------- |
| `autoformat.yml`            | Apply autoformatting when a PR is commented with `autoformat`  |
| `autoformat.js`             | Define utility functions used in the `autoformat.yml` workflow |
| `labeling.yml`              | Automatically apply labels on issues and PRs                   |
| `notify-dco-failure.yml`    | Notify a DCO check failure                                     |
| `notify-dco-failure.js`     | The main script of the `notify-dco-failure.yml` workflow       |
| `release-note-category.yml` | Validate a release-note category label is applied on a PR      |
| `release-note-category.js`  | The main script of the `release-note-category.yml` workflow    |
