# cross version test

A python package to provide CLI commands to run cross version tests for MLflow.

## Prerequisites

- Docker
- Docker Compose

## Installation

```bash
pip install -e .
```

## Quick starter

```bash
cross-version-test build -p sklearn_0.24.2_autologging
cross-version-test run -p sklearn_0.24.2_autologging
```

## Usage

See [usage.md](./usage.md).

## Development

```bash
# Install development dependencies
pip install -e .[dev]

# Test
pytest tests

# Type check
mypy --disallow-untyped-defs .
```
