# MLflow API Documentation

This directory contains the MLflow API reference. The source code (`.rst` files) is relatively minimal, as the API docs are mainly populated by docstrings in the MLflow Python source.

## Building the docs

First, install dependencies for building docs as described in the [Environment Setup and Python configuration](https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md#environment-setup-and-python-configuration) section of the main MLflow contribution guide.

Building documentation requires [Pandoc](https://pandoc.org/index.html). It should have already been
installed if you used the automated env setup script
([dev-env-setup.sh](https://github.com/mlflow/mlflow/blob/master/dev/dev-env-setup.sh)),
but if you are manually installing dependencies, please follow [the official instruction](https://pandoc.org/installing.html).

Also, check the version of your installation via `pandoc --version` and ensure it is 2.2.1 or above.
If you are using Mac OSX, be aware that the Homebrew installation of Pandoc may be outdated. If you are using Linux,
you should use a deb installer or install from the source, instead of running `apt` / `apt-get` commands. Pandoc package available on official
repositories is an older version and contains several bugs. You can find newer versions at <https://github.com/jgm/pandoc/releases>.

To generate a live preview of Python & other rst documentation, run the
following snippet. Note that R & Java API docs must be regenerated
separately after each change and are not live-updated; see subsequent
sections for instructions on generating R and Java docs.

```bash
cd docs
make livehtml
```

Generate R API rst doc files via:

```bash
cd docs
make rdocs
```

---

**NOTE**

If you attempt to build the R documentation on an ARM-based platform (Apple silicon M1, M2, etc.)
you will likely get an error when trying to execute the Docker build process for the make command.
To address this, set the default docker platform environment variable as follows:

```bash
export DOCKER_DEFAULT_PLATFORM=linux/amd64
```

---

Generate Java API rst doc files via:

```bash
cd docs
make javadocs
```

Generate API docs for all languages via:

```bash
cd docs
make html
```

Generate only the main .rst based documentation:

```bash
cd docs
make rsthtml
```

After running these commands, a build folder containing the generated
HTML will be generated at `build/html`.

If changing existing Python APIs or adding new APIs under existing
modules, ensure that references to the modified APIs are updated in
existing docs under `docs/source`. Note that the Python doc generation
process will automatically produce updated API docs, but you should
still audit for usages of the modified APIs in guides and examples.

If adding a new public Python module, create a corresponding doc file
for the module under `docs/source/python_api` - [see
here](https://github.com/mlflow/mlflow/blob/v0.9.1/docs/source/python_api/mlflow.tracking.rst#mlflowtracking)
for an example.

> Note: If you are experiencing issues with rstcheck warning of failures in files that you did not modify, try:

```bash
cd docs
make clean; make html
```
