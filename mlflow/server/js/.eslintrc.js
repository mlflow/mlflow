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
        '@typescript-eslint/consistent-type-imports': 'off', // TODO: enable and run --fix
        '@typescript-eslint/no-unused-vars': 'off',
        '@typescript-eslint/no-redeclare': 'off',
        // END-ESLINT-MIGRATION (FEINF-1337)

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
        'import/namespace': 0,
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

        // Please leave this rule as the last item in the array, as it's quite large
        '@typescript-eslint/ban-types': [
          'error',
          {
            types: {
              Function: {
                message:
                  'The `Function` type accepts any function-like value. It provides no type safety when calling the function, which can be a common source of bugs.\nIt also accepts things like class declarations, which will throw at runtime as they will not be called with `new`.\nIf you are expecting the function to accept certain arguments, you should explicitly define the function shape.',
                fixWith: '(...args: unknown[]) => unknown',
              },
              '{}': {
                message:
                  '`{}` actually means "any non-nullish value".\n- If you want a type meaning "any object", you probably want `Record<string, unknown>` instead.\n- If you want a type meaning "any value", you probably want `unknown` instead.\n- If you want a type meaning "empty object", you probably want `Record<string, never>` instead.',
                fixWith: 'Record<string, never>',
              },
              Object: {
                message:
                  '`Object` actually means "any non-nullish value".\n- If you want a type meaning "any object", you probably want `Record<string, unknown>` instead.\n- If you want a type meaning "any value", you probably want `unknown` instead.\n- If you want a type meaning "empty object", you probably want `Record<string, never>` instead.',
                fixWith: 'Record<string, unknown>',
              },
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
      files: ['*.test.js', '*-test.js', '*-test.jsx', '*.test.ts', '*-test.ts', '*.test.tsx', '*-test.tsx', 'test/**'],
      plugins: ['jest', 'chai-expect', 'chai-friendly', 'testing-library'],
      globals: {
        sinon: true,
        chai: true,
        expect: true,
        assert: true,
      },
      rules: {
        'func-names': 0,
        'max-lines': 0,
        'chai-expect/missing-assertion': 2,
        'no-unused-expressions': 0,
        'chai-friendly/no-unused-expressions': 2,
        'testing-library/no-debugging-utils': 'error',
        'testing-library/no-dom-import': 'error',
        'testing-library/await-async-utils': 'error',
        '@typescript-eslint/no-non-null-assertion': 'off',
      },
    },
    {
      files: OverrideFiles.TEST,
      rules: {
        // BEGIN-ESLINT-MIGRATION (FEINF-1337)
        'jest/expect-expect': 'off',
        'jest/no-conditional-expect': 'off',
        // END-ESLINT-MIGRATION (FEINF-1337)
      },
    },
  ],
});
