const { createConfig, OverrideFiles } = require('@databricks/config-eslint');

module.exports = createConfig({
  jestOverrideRules: {
    // TODO(FEINF-1784): Enable rule
    '@databricks/no-restricted-globals-with-module': 'off',
  },
})({
  rules: {
    // BEGIN-ESLINT-MIGRATION (FEINF-1337)
    '@databricks/no-hardcoded-colors': 'off',
    'import/order': 'off', // TODO: enable and run --fix
    'import/first': 'off', // TODO: enable and run --fix
    'import/no-duplicates': 'off', // TODO: enable and run --fix
    'import/no-anonymous-default-export': 'off', // TODO: enable and run --fix
    'react/react-in-jsx-scope': 'off',
    'react/forbid-component-props': 'off', // need to rename all data-test-id to data-testid
    'react/forbid-dom-props': 'off', // need to rename all data-test-id to data-testid
    '@databricks/no-double-negation': 'off',
    'react/self-closing-comp': 'off',
    'react/jsx-key': 'off',
    'react/prop-types': 'off',
    'formatjs/enforce-default-message': 'off', // Requires turning all JSX string props from <X s={'str'}> to <X s="str">, there are some typos as well
    'formatjs/enforce-description': 'off', // Requires turning all JSX string props from <X s={'str'}> to <X s="str">, there are some typos as well
    '@databricks/no-wrapper-formui-label': 'off',
    '@databricks/no-restricted-imports-regexp': [
      'error',
      {
        patterns: [
          ...require('@databricks/config-eslint/shared/no-restricted-imports-regexp-base').filter(
            (pattern) => pattern.pattern !== '^enzyme$',
          ),
          {
            pattern: '^antd$',
            message:
              'Do not import from `antd`, consider using replacement components from `@databricks/design-system` module instead. If a direct replacement is not available, please contact design system team.',
            allowTypeImports: true,
          },
        ],
      },
    ],
    '@databricks/no-global-uninitialized': 'off',
    '@databricks/no-uncaught-localstorage-setitem': 'off',
    '@databricks/no-window-top': 'off',
    // END-ESLINT-MIGRATION (FEINF-1337)
  },
  overrides: [
    {
      files: OverrideFiles.TS,
      rules: {
        // BEGIN-ESLINT-MIGRATION (FEINF-1337)
        '@typescript-eslint/consistent-type-imports': 'off', // TODO: enable and run --fix
        '@typescript-eslint/no-unused-vars': 'off',
        '@typescript-eslint/no-redeclare': 'off',
        // END-ESLINT-MIGRATION (FEINF-1337)
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
        'jest/valid-title': 'off',
        'jest/valid-expect': 'off',
        'jest/no-identical-title': 'off',
        'jest/no-done-callback': 'off',
        'jest/expect-expect': 'off',
        'jest/no-conditional-expect': 'off',
        // END-ESLINT-MIGRATION (FEINF-1337)
      },
    },
  ],
});
