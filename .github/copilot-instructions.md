# Copilot Instructions

This file provides guidelines and instructions for customizing GitHub Copilot's behavior in this repository.

## Linting and Formatting

We use `pre-commit` to run linters and formatters on the codebase. We strongly encourage you to run `pre-commit` locally before pushing your changes and to fix any issues it reports.

## Workflow Trigger Comments

In this repository, we use the following comments to trigger GitHub Action workflows on the PR. These comments are not relevant to the code review process but are used to automate specific tasks. Please ignore them if you see them in a PR.

- `/autoformat`: Triggers the [`autoformat.yml`](./workflows/autoformat.yml) workflow.
- `/cvt`: Triggers the [`cross-version-tests.yml`](./workflows/cross-version-tests.yml) workflow.

## Code Review Guidelines

When reviewing code, please follow these guidelines to ensure consistency and quality across the codebase:

### Python

- Functions should have type hints to improve code readability and maintainability:

  ```python
  # Bad
  def process_items(items):
      return {item: len(item) for item in items}


  # Good
  def process_items(items: list[str]) -> dict[str, int]:
      return {item: len(item) for item in items}
  ```

- Avoid ambiguous types like `dict`, `list`, or `Callable`. Use more specific types like `dict[str, int]`, `list[str]`, or `Callable[[int], str]`:

  ```python
  # Bad
  def process_items(items: list) -> dict:
      return {item: len(item) for item in items}


  # Good
  def process_items(items: list[str]) -> dict[str, int]:
      return {item: len(item) for item in items}
  ```

- Try-catch blocks should only wrap operations that can actually fail. Move operations that don't throw exceptions outside the try block:

  ```python
  # Bad
  try:
      never_fails()
      another_never_fails()
      can_fail()
  except ...:
      handle_error()

  # Good
  never_fails()
  another_never_fails()
  try:
      can_fail()
  except ...:
      handle_error()
  ```

- Replace generic `except Exception` with specific exception types:

  ```python
  # Bad
  try:
      json.loads(some_string)
  except Exception:
      ...

  # Good
  try:
      json.loads(some_string)
  except json.JSONDecodeError:
      ...
  ```

- Functions returning tuples with more than 2 elements should use a `dataclass` instead:

  ```python
  # Bad
  def get_user_info() -> tuple[str, int, str]:
      return "Alice", 30, "Engineer"


  name, age, occupation = get_user_info()
  print(name)

  # Good
  from dataclasses import dataclass


  @dataclass
  class UserInfo:
      name: str
      age: int
      occupation: str


  def get_user_info() -> UserInfo:
      return UserInfo(name="Alice", age=30, occupation="Engineer")


  user_info = get_user_info()
  print(user_info.name)
  ```
