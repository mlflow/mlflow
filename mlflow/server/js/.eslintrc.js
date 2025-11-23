const { createConfig, OverrideFiles } = require('@databricks/config-eslint');

module.exports = createConfig({})({
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
      files: OverrideFiles.TEST,
      rules: {
        // BEGIN-ESLINT-MIGRATION (FEINF-1337)
        'jest/expect-expect': 'off',
        // END-ESLINT-MIGRATION (FEINF-1337)
      },
    },
  ],
});
