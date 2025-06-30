# CLAUDE.md

## Product Overview

MLflow is an open-source platform, purpose-built to assist machine learning practitioners and teams in handling the complexities of the machine learning process. MLflow focuses on the full lifecycle for machine learning projects, ensuring that each phase is manageable, traceable, and reproducible.

The core components of MLflow are:

- Experiment Tracking: A set of APIs to log models, params, and results in ML experiments and compare them using an interactive UI.
- Model Packaging: A standard format for packaging a model and its metadata, such as dependency versions, ensuring reliable deployment and strong reproducibility.
- Model / Prompt Registry: A centralized model store, set of APIs, and UI, to collaboratively manage the full lifecycle of MLflow Models and Prompts.
- Serving: Tools for seamless model deployment to batch and real-time scoring on platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker.
- Evaluation: A suite of automated model evaluation tools, seamlessly integrated with experiment tracking to record model performance and visually compare results across multiple models.
- Observability: Tracing integrations with various GenAI libraries and a Python SDK for manual instrumentation, offering smoother debugging experience and supporting online monitoring.

## Tech Stack

* Mainly written in Python (>= 3.10)
* UI components: React/TypeScript
* Tracking server: Flask and Uvicorn, protobuf for communication between the tracking server and the client.
* Model serving server: FastAPI and Uvicorn
* Tracing: OpenTelemetry

## Architecture Overview

### Core Components
- **mlflow/tracking/** - Experiment and run management, fluent API
- **mlflow/entities/** - Python data classes for MLflow entities, such as experiments, runs, traces, etc.
- **mlflow/models/** - Model packaging, serving, and registry, and evaluation for traditional ML / deep learning frameworks.
- **mlflow/genai/** - GenAI integration, such as LLM evaluation, prompt optimization, etc.
- **mlflow/data/** - Dataset management and lineage
- **mlflow/tracing/** - LLM observability and tracing
- **mlflow/evaluation/** - Model evaluation framework
- **mlflow/server/** - REST API, GraphQL, and React UI
- **mlflow/store/** - Abstract storage layer with multiple backends
- **mlflow/dev/** - Development utilities, such as proto generation script.

### ML Framework Integrations (Flavors)
MLflow supports a lot of frameworks through "flavors" in dedicated directories:
- **Deep Learning**: pytorch, tensorflow, keras, onnx
- **Traditional ML**: sklearn, xgboost, lightgbm, h2o
- **LLM/GenAI**: openai, anthropic, langchain, transformers, mistral
- **AutoML**: optuna, pmdarima, prophet

Each flavor defines the common set of functionalities:
* Logging a model using `mlflow.<flavor_name>.log_model`
* Loading a model using `mlflow.<flavor_name>.load_model`
* Loading a model with a unified interface using `mlflow.pyfunc.load_model`
* Autologging for training and inference with `mlflow.<flavor_name>.autolog`
  * For traditional ML / deep learning frameworks, autologging is primary for tracking training process.
  * For GenAI frameworks, autologging is primary for tracing (observability).

### UI Components
- UI components are defined in `mlflow/server/js/` using React/TypeScript.
- To build JS assets, run `cd mlflow/server/js && yarn install && yarn build`.
- To run the development server, run `cd mlflow/server/js && yarn start`.

## Databricks Integration
- MLflow is tightly integrated with Databricks products such as Managed MLflow, Model Serving, Unity Catalog, and more. When working with a code piece that is used by Databricks integration as well, be careful not to break the integration.

## Package

### Sub Packages
MLflow includes following sub packages:
- `mlflow-skinny`: A minimal version of MLflow that only includes the client-side functionality, i.e., excluding server.
- `mlflow-tracing`: A lightweight version of MLflow that only includes the tracing functionality.

### SDKs
MLflow support following language-specific SDKs:
- Python
- R
- Java
- Typescript (to be added in a month)

## Documentation
- Documentation in `docs/` using Docusaurus for user docs and Sphinx for API docs
- Refer to @mlflow/docs/README.md for more details on how to build and deploy the docs.
- The published documentation is available at https://mlflow.org/docs/latest/.

### Package Configuration

MLflow defines its package configurations in `pyproject.toml`.

## Development Commands

### Common Development Flow

1. Create a new branch for the feature or bug fix.
2. Make changes to the code.
3. Run tests with `pytest` to make sure the changes are working as expected.
4. Run `pre-commit run --all-files` to run the linting and formatting checks.
5. If needed, create a simple E2E test script and check UI to verify the changes.

### Environment Setup

Use `uv` to install the dependencies quickly.

```bash
# Create a virtual environment (if not already created)
uv venv --python 3.10
source .venv/bin/activate
# Install the package in editable mode
uv pip install -e .
# Install test dependencies
uv pip install -r requirements/test-requirements.txt
# Install lint dependencies
uv pip install -r requirements/lint-requirements.txt
```

This setup install the minimum dependencies to run MLflow and tests for core functionality. You may also want to install extra dependencies such as machine learning frameworks, if you face missing dependencies errors.

### Code Quality and Formatting

```bash
pre-commit run --all-files
```

### Testing

- Tests are organized in `tests/` directory.
- Tests are written with `pytest` and heavily use fixtures. `conftest.py` in several subdirectories contains the fixtures for the tests.
- Each flavor has corresponding tests in `tests/[flavor_name]/`
- When running tests for those flavors, refer to the @ml-package-versions.yaml file to install the correct versions of the test dependencies for the flavor.
- Running all tests with `pytest` will take hours, so you shouldn't do it. Instead, run tests for specific flavors or specific files.
- Always run tests and make sure they pass before committing.

### UI Testing
TODO: Integrate with Playwright for automated testing.

### Code Quality Standards
- Use specific type hints (avoid generic `dict`, `list`)
- Narrow try-catch blocks to operations that can actually fail
- Use specific exception types instead of `except Exception`
- Follow Google Python Style Guide for docstrings
- Prefer function-based tests over class-based tests in pytest.
