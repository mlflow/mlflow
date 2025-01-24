# MLflow Documentation

This README covers information about the main MLflow documentation. The API reference is built separately and included as a static folder during the full build process. Please check out the [README](https://github.com/mlflow/mlflow/blob/master/docs/api_reference/README.md) in the `api_reference` folder for more information.

## Prerequisites

**Necessary**
- NodeJS >= 18.0 (see the NodeJS documentation for installation instructions)

**Optional**
- (For building MDX files from `.ipynb` files) Python 3.9+ and [nbconvert](https://pypi.org/project/nbconvert/).
- (For building API docs) See [doc-requirements.txt] for API doc requirements.

## Installation

```
$ yarn
```

## Local Development

```
$ yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

In order to build the full MLflow documentation (i.e. the contents of https://mlflow.org/docs/latest/), please follow the following steps:

1. Run `yarn build-api-docs` in order to build the API reference and copy the generated HTML to `static/api_reference`.
2. Run `yarn update-api-modules` to keep API references up-to-date (this is used by the [APILink component](https://github.com/mlflow/mlflow/blob/master/docs/src/components/APILink/index.tsx)).
3. Run `yarn convert-notebooks` to convert `.ipynb` files to `.mdx` files (do not commit the results)
4. **⚠️ Important!** Run `export DOCS_BASE_URL=/docs/latest` (or wherever the docs are expected to be served). This configures the [Docusaurus baseUrl](https://docusaurus.io/docs/api/docusaurus-config#baseUrl), and the site may not render correctly if this is improperly set.
5. Finally, run `yarn build`. This generates static files in the `build` directory, which can then be served.
