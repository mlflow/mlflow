# Copilot Instructions

This file provides guidelines and instructions for customizing GitHub Copilot's behavior in this repository.

## Linting and Formatting

We use `pre-commit` to run linters and formatters on the codebase. We strongly encourage you to run `pre-commit` locally before pushing your changes and to fix any issues it reports.

## Python Version Requirements

When contributing code or reviewing pull requests, always check the `.python-version` file in the repository root to determine the minimum supported Python version. This file specifies the baseline Python version that all code must be compatible with. Ensure that:

- Code uses syntax and features compatible with the minimum Python version specified in `.python-version`
- Type hints and language constructs are appropriate for the target Python version
- Dependencies and imports are available in the minimum supported Python version

## Workflow Trigger Comments

In this repository, we use the following comments to trigger GitHub Action workflows on the PR. These comments are not relevant to the code review process but are used to automate specific tasks. Please ignore them if you see them in a PR.

- `/autoformat`: Triggers the [`autoformat.yml`](./workflows/autoformat.yml) workflow.
- `/cvt`: Triggers the [`cross-version-tests.yml`](./workflows/cross-version-tests.yml) workflow.

## Language-Specific Guidelines

- [Python](../dev/style-guides/python.md)
