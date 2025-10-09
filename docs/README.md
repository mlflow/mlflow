# MLflow Documentation

This README covers information about the main MLflow documentation. The API reference is built separately and included as a static folder during the full build process. Please check out the [README](https://github.com/mlflow/mlflow/blob/master/docs/api_reference/README.md) in the `api_reference` folder for more information.

## Prerequisites

**Necessary**

- NodeJS >= 18.0 (see the [NodeJS documentation](https://nodejs.org/en/download) for installation instructions)
- (For building MDX files from `.ipynb` files) Python 3.9+, [nbconvert](https://pypi.org/project/nbconvert/), [nbformat](https://pypi.org/project/nbformat/) and [pyyml](https://pypi.org/project/pyyml/)

**Optional**

- (For building API docs) See [doc-requirements.txt](https://github.com/mlflow/mlflow/blob/master/requirements/doc-requirements.txt) for API doc requirements.

## Installation

```
$ npm install
```

## Local Development

1. If you haven't done this before, run `npm run convert-notebooks` to convert `.ipynb` files to `.mdx` files. The generated files are git-ignored.

2. Run the development server:

```
$ npm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

**Note**: Some server-side rendering features will not work in this mode (e.g. the [client redirects plugin](https://docusaurus.io/docs/api/plugins/@docusaurus/plugin-client-redirects)). To test these, please use the "Build and Serve" workflow below.

## Build and Serve

In order to build the full MLflow documentation (i.e. the contents of https://mlflow.org/docs/latest/), please follow the following steps:

1. Run `npm run build-api-docs` in order to build the API reference and copy the generated HTML to `static/api_reference`.
   a. To speed up the build locally, you can run `npm run build-api-docs:no-r` to skip building R documentation
2. Run `npm run convert-notebooks` to convert `.ipynb` files to `.mdx` files. The generated files are git-ignored.
3. **⚠️ Important!** Run `export DOCS_BASE_URL=/docs/latest` (or wherever the docs are expected to be served). This configures the [Docusaurus baseUrl](https://docusaurus.io/docs/api/docusaurus-config#baseUrl), and the site may not render correctly if this is improperly set.
4. Finally, run `npm run build`. This generates static files in the `build` directory, which can then be served.
5. (Optional) To serve the artifacts generated in the above step, run `npm run serve`.

## Building for release

The generated `build` folder is expected to be hosted at https://mlflow.org/docs/latest. However, as our docs are versioned, we also have to generate the documentation for `https://mlflow.org/docs/{version}`. To do this conveniently, you can run the following command:

```
npm run build-all
```

This command will run all the necessary steps from the "Build and Serve" workflow above, and set the correct values for `DOCS_BASE_URL`. The generated HTML will be dumped to `build/latest` and `build/{version}`. These two folders can then be copied to the [docs repo](https://github.com/mlflow/mlflow-legacy-website/tree/main/docs) and uploaded to the website.

## Troubleshooting

### `Error: Invalid sidebar file at "sidebarsGenAI.ts". These sidebar document ids do not exist:`

This error occurs when some links in the sidebar point to non-existent documents.

When it errors for `-ipynb` pages, it is not the link problem but the notebook conversion script is not run. Run `npm run convert-notebooks` in the above steps to convert `.ipynb` files to `.mdx` files. The generated files are git-ignored.

```
[ERROR] Error: Invalid sidebar file at "sidebarsGenAI.ts".
These sidebar document ids do not exist:

eval-monitor/notebooks/huggingface-evaluation-ipynb
eval-monitor/notebooks/question-answering-evaluation-ipynb
...
```
