# Du Bois Design System

DuBois is a shared language for building products at Databricks. You can find the live demos and documentation for this system at **[go/dubois](http://go/dubois)**.

## Running Locally

`yarn && yarn storybook`

## Installation

Refer to [this PR](https://github.com/databricks/universe/pull/95010) for an example of how we integrated the design system into webapp. This process is under development by the UI Infra Devloop Track.

## Development

Run prettier to ensure that formatting is consistent:

### Storybook cache cleaning

Storybook caches general state such as flags/modes/addons. If you are encountering missing flags, addons not loading, or other issues, you can clear the cache by running:

```
yarn workspace @databricks/design-system storybook --no-manager-cache
```

Or alternatively you can directly delete storybook's cache from
`universe/design-system/node_modules/.cache/storybook`

```
yarn prettier
```

### Running Cypress Tests

Cypress tests are located in the `cypress/integration` directory.

To run Cypress tests, first ensure that an existing storybook local development server is running.

```bash
yarn start:storybook
```

Then, open the Cypress UI in another terminal.

```bash
yarn cy:open
```

## Publishing to NPM

To publish the package to NPM, you'll need to ask one of the UI Core leads. Ping @oncall in #help-frontend for assistance.

If you have permissions to publish, First make sure you have authenticated with `yarn npm login`. Then, run `yarn && npm version $(npm view @databricks/design-system version) --no-workspaces-update && npm version patch --no-workspaces-update && yarn npm publish --access public && yarn postpublish && yarn`
