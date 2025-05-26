# Copilot Instructions

## Linting and Formatting

We use `pre-commit` to run linters and formatters on the codebase. See [`lint.yml`](../.github/workflows/lint.yml) for instructions on how to install required tools.
We strongly encourage you to run `pre-commit` locally before pushing your changes and to fix any issues it reports.

## Comments to ignore

In this repository, we use the following comments to trigger GitHub Action workflows on the PR.
When you see these comments in a PR, please ignore them, as they are not relevant to the code review process.

- `/autoformat`: Triggers the [`autoformat.yml`](../.github/workflows/autoformat.yml) workflow.
- `/cvt`: Triggers the [`cross-version-tests.yml`](../.github/workflows/cross-version-tests.yml) workflow.
