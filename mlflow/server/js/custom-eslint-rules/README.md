# Custom ESLint Rules

This directory contains custom ESLint rules specific to the MLflow UI codebase.

## How It Works

This package is locally referenced by `mlflow/server/js/package.json` as `eslint-plugin-mlflow`.
In `mlflow/server/js/.eslintrc.json`, we automatically load the recommended rules in `index.js`.

## Adding New Custom Rules

1. Create a new `.js` file in this directory (e.g., `my-custom-rule.js`)
2. Export an ESLint rule module with `meta` and `create` functions:

```javascript
module.exports = {
  meta: {
    type: 'problem',
    docs: {
      description: 'Description of your rule',
      category: 'Best Practices',
      recommended: true,
    },
    messages: {
      myMessage: 'Your error message here',
    },
  },
  create(context) {
    return {
      // Your rule implementation
    };
  },
};
```

3. Import the rule in `index.js`, and add it to the `rules` sections of the export (add to the
   top-level `rules` config as well as the one in `recommended`):

```javascript
const myCustomRule = require('./my-custom-rule');

module.exports = {
  rules: {
    // ...
    'my-custom-rule': myCustomRule,
  },
  configs: {
    recommended: {
      plugins: ['mlflow'],
      rules: {
        'mlflow/my-custom-rule': 'error',
      },
    },
  },
};
```

4. Run `yarn install` from `mlflow/server/js` to make sure the new rules are loaded

5. Run `yarn lint` from `mlflow/server/js` to make sure the new rule works as intended

## Testing Custom Rules

Run `yarn test` to run all tests in the directory. We currently use ESlint's `RuleTester` util to
write unit tests for our rules. Alternatively you can spot-check by running `yarn lint` in
`mlflow/server/js` on files you expect to fail your rule.
