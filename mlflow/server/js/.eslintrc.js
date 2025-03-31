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
        // react-app original rules

        // TypeScript's `noFallthroughCasesInSwitch` option is more robust (#6906)
        'default-case': 'off',
        // 'tsc' already handles this (https://github.com/typescript-eslint/typescript-eslint/issues/291)
        'no-dupe-class-members': 'off',
        // 'tsc' already handles this (https://github.com/typescript-eslint/typescript-eslint/issues/477)
        'no-undef': 'off',

        // Add TypeScript specific rules (and turn off ESLint equivalents)
        '@typescript-eslint/consistent-type-assertions': 'warn',
        'no-array-constructor': 'off',
        '@typescript-eslint/no-array-constructor': 'warn',
        'no-redeclare': 'off',
        '@typescript-eslint/no-redeclare': 'warn',
        'no-use-before-define': 'off',
        '@typescript-eslint/no-use-before-define': [
          'warn',
          {
            functions: false,
            classes: false,
            variables: false,
            typedefs: false,
          },
        ],
        'no-unused-expressions': 'off',
        '@typescript-eslint/no-unused-expressions': [
          'error',
          {
            allowShortCircuit: true,
            allowTernary: true,
            allowTaggedTemplates: true,
          },
        ],
        'no-unused-vars': 'off',
        // '@typescript-eslint/no-unused-vars': [
        //   'warn',
        //   {
        //     args: 'none',
        //     ignoreRestSiblings: true,
        //   },
        // ],
        'no-useless-constructor': 'off',
        '@typescript-eslint/no-useless-constructor': 'warn',
        // end react-app original rules

        // Turning off temporarily until TS migration is complete
        'import/first': 0,
        'import/extensions': 0,
        'import/newline-after-import': 0,
        'import/no-duplicates': 0,
        '@typescript-eslint/no-unused-vars': 0,
        'react/prop-types': 0,
        'max-lines': 0,
        'jsx-a11y/click-events-have-key-events': 0,
        'jsx-a11y/no-static-element-interactions': 0,
        'jsx-a11y/interactive-supports-focus': 0,
        'jsx-a11y/label-has-associated-control': 0,
        'jsx-a11y/no-noninteractive-element-interactions': 0,
        'jsx-a11y/no-autofocus': 0,

        // Do not require functions (especially react components) to have explicit returns
        '@typescript-eslint/explicit-function-return-type': 'off',
        // Do not require to type every import from a JS file to speed up development
        '@typescript-eslint/no-explicit-any': 'off',
        'no-empty-function': 'off',
        '@typescript-eslint/no-empty-function': ['error', { allow: ['arrowFunctions', 'methods'] }],
        // Many API fields and generated types use camelcase
        '@typescript-eslint/naming-convention': 'off',

        // TODO(thielium): This should be re-enabled (REDASH-796)
        '@typescript-eslint/explicit-module-boundary-types': 'off',

        // ts-migrate introduces a lot of ts-expect-error. turning into warning until we finalize the migration
        '@typescript-eslint/ban-ts-comment': 'warn',

        'no-shadow': 'off',
        '@typescript-eslint/no-shadow': 'off',

        '@typescript-eslint/no-empty-object-type': 'error',
        '@typescript-eslint/no-unsafe-function-type': 'error',
        '@typescript-eslint/no-wrapper-object-types': 'error',
        '@typescript-eslint/no-restricted-types': [
          'error',
          {
            types: {
              object: {
                message:
                  "Don't use `object` as a type. The `object` type is currently hard to use ([see this issue](https://github.com/microsoft/TypeScript/issues/21732)).\nConsider using `Record<string, unknown>` instead, as it allows you to more easily inspect and use the keys.",
                fixWith: 'Record<string, unknown>',
              },
            },
          },
        ],
        // By using "auto" JSX runtime in TS, we have react automatically injected and
        // adding "React" manually results in TS(6133) error
        'react/react-in-jsx-scope': 'off',

        // '@typescript-eslint/no-unused-vars': ['error', { varsIgnorePattern: '^oss_' }],
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
