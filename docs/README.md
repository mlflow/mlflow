# MLflow Docusaurus Migration

The `docusaurus` branch is the working branch for all Docusaurus migration work. We are switching away from Sphinx for our main informational docs, and using [Docusaurus](https://docusaurus.io/) instead.

## Jobs to be Done

1. Migrating all non-API related docs from `.rst` to `.mdx`, the Docusaurus format
  a. These files can be found in the `rst_to_migrate` folder
2. Migrating notebooks to be rendered via [this plugin](https://github.com/datalayer/jupyter-ui/tree/main/packages/docusaurus-plugin), rather than by `nbsphinx`
  a. Notebooks are also found in the `rst_to_migrate` folder, but will end in `.ipynb` 

### Installation

```
$ yarn
```

### Local Development

```
$ yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

```
$ yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

### Formatting

```
$ yarn format
```

This command automatically formats all `.mdx` files within the `/docs` folder


### Deployment

Using SSH:

```
$ USE_SSH=true yarn deploy
```

Not using SSH:

```
$ GIT_USER=<Your GitHub username> yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
