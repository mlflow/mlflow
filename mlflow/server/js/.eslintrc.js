const { createConfig, OverrideFiles } = require('@databricks/config-eslint');

module.exports = createConfig({})({
  plugins: ['plugin:mlflow/recommended'],
  rules: {
    // BEGIN-ESLINT-MIGRATION (FEINF-1337)
    '@databricks/no-hardcoded-colors': 'off',
    'import/order': 'off', // TODO: enable and run --fix
    'import/no-duplicates': 'off', // TODO: enable and run --fix
    'import/no-anonymous-default-export': 'off', // TODO: enable and run --fix
    '@databricks/no-double-negation': 'off',
    '@databricks/no-wrapper-formui-label': 'off',
    '@databricks/no-restricted-imports-regexp': [
      'error',
      {
        patterns: [
          ...require('@databricks/config-eslint/shared/no-restricted-imports-regexp-base').filter(
            (pattern) => pattern.pattern !== '^enzyme$',
          ),
        ],
      },
    ],
    '@databricks/no-uncaught-localstorage-setitem': 'off',
    '@databricks/no-window-top': 'off',
    // END-ESLINT-MIGRATION (FEINF-1337)
    // Exempt mlflow from Apollo singleton rules because MLflow has its own
    // Apollo client separate from workspace console and thus doesn't need a workspace-scoped provider
    '@databricks/no-singleton-apollo-client': 'off',
    '@databricks/no-apollo-client-provider': 'off',
  },
  overrides: [
    {
      files: OverrideFiles.TS,
      rules: {
        // BEGIN-ESLINT-MIGRATION (FEINF-1337)
        '@typescript-eslint/no-unused-vars': 'off',
        '@typescript-eslint/no-redeclare': 'off',
        // END-ESLINT-MIGRATION (FEINF-1337)

        // Allow unreachable code because EDGE blocks result in unreachable code.
        'no-unreachable': 'off',
      },
    },
    {
      files: OverrideFiles.JSX_TSX,
      rules: {
        // BEGIN-ESLINT-MIGRATION (FEINF-1337)
        'react/no-unused-prop-types': 'off',
        // END-ESLINT-MIGRATION (FEINF-1337)
      },
    },
    {
      files: ['*.test.js', '*-test.js', '*-test.jsx', '*.test.ts', '*.test.tsx', '*-test.ts', '*-test.tsx', 'test/**'],
      // Allow absolute AJAX URLs in test files for mocking
      rules: {
        'mlflow/no-absolute-ajax-urls': 'off',
      },
    }
  ],
});
