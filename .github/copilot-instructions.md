<!-- https://docs.github.com/en/copilot/customizing-copilot/adding-repository-custom-instructions-for-github-copilot -->

# Copilot Instructions

This file provides guidelines and instructions for customizing GitHub Copilot's behavior in this repository.
## Linting and Formatting

We use `pre-commit` to run linters and formatters on the codebase. See [`lint.yml`](/.github/workflows/lint.yml) for instructions on how to install required tools.
We strongly encourage you to run `pre-commit` locally before pushing your changes and to fix any issues it reports.

## Workflow Trigger Comments

In this repository, we use the following comments to trigger GitHub Action workflows on the PR.
These comments are not relevant to the code review process but are used to automate specific tasks. Only maintainers are allowed to use these commands.

- `/autoformat`: Triggers the [`autoformat.yml`](/.github/workflows/autoformat.yml) workflow.
- `/cvt`: Triggers the [`cross-version-tests.yml`](/.github/workflows/cross-version-tests.yml) workflow.
