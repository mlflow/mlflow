require('@rushstack/eslint-patch/modern-module-resolution');

const restrictedGlobals = require('confusing-browser-globals');

const { OverrideFiles } = require('./override-files');
const playwrightConfig = require('./playwright');

/**
 * @typedef {import('eslint').Linter.Config} ESLintConfig
 */

/**
 * @typedef {Object} CreateConfigFactoryOptions
 * @property {boolean} [hasOxlint=false] Whether oxlint is enabled, if true, some rules are disabled in eslint and covered by oxlint
 * @property {Record<string, any>} [jestOverrideRules={}] Jest override rules
 * @property {ESLintConfig['rules']} [importOrderOverridePathGroups=[]] Override import order path groups for import/order rule
 * @property {(baseConfig: ESLintConfig) => ESLintConfig} [eslint] Custom eslint config factory
 * @property {boolean} [restrictReactQueryImports=false] Whether to restrict 'react-query' imports
 * @property {boolean} [allowGlobalFetch=false] Whether to allow global fetch
 * @property {boolean} [allowQueryClientProvider=false] Whether to allow direct QueryClientProvider usage
 * @property {boolean} [noBrowserOnlyGlobals=false] Whether to restrict browser-only globals (window, document, etc.) for platform-agnostic packages
 */

/**
 * @type {Partial<CreateConfigFactoryOptions>}
 */
const defaultOptions = {
  hasOxlint: false,

  jestOverrideRules: {},

  importOrderOverridePathGroups: [],
  restrictReactQueryImports: false,
  allowGlobalFetch: false,
  allowQueryClientProvider: false,
  noBrowserOnlyGlobals: false,
};

const FORBIDDEN_PROPS = {
  forbid: [
    {
      propName: 'data-test-id',
      message:
        "Use 'data-testid' instead of 'data-test-id' to match @testing-library/react convention (https://testing-library.com/docs/queries/bytestid/)",
    },
  ],
};

// Additional globals that are blocked via no-restricted-globals when noBrowserOnlyGlobals is provided. Some browser
// only globals are already blocked via restrictedGlobals such as history or location. The purpose of this list is to
// allow blocking browser only APIs by default while still being able to access globals that are available in both
// browser and Node.
const BROWSER_ONLY_GLOBALS = [
  'window',
  'document',
  'navigator',
  'localStorage',
  'sessionStorage',
  'indexedDB',
  'alert',
  'prompt',
  'requestAnimationFrame',
  'cancelAnimationFrame',
  'requestIdleCallback',
  'cancelIdleCallback',
  'getComputedStyle',
  'matchMedia',
  'ResizeObserver',
  'IntersectionObserver',
  'MutationObserver',
  'Worker',
  'ServiceWorker',
  'Notification',
  'XMLHttpRequest',
];

/**
 * @param {Partial<CreateConfigFactoryOptions>} [options]
 * @return {(appESLintConfig?: ESLintConfig | ((baseConfig: ESLintConfig) => ESLintConfig)) => ESLintConfig}
 *
 * @example
 *
 * module.exports = createConfig({
 *   eslint(baseConfig) {
 *     return {
 *       ...baseConfig,
 *       rules: {
 *         ...baseConfig.rules,
 *         'extra-rule': 'error',
 *       },
 *     };
 *   }
 * })
 *
 */
function createConfigFactory(options = {}) {
  options = { ...defaultOptions, ...options };

  // Rules for all test files and test helper files (e.g. /test/ and /test-utils/)
  const testCommonRules = {
    // Disable no-non-null-assertion in tests as there's no risk of hitting nulls (tests just won't pass)
    '@typescript-eslint/no-non-null-assertion': 'off',
    // Disable compat checks in tests as they always run with the same Node.js version
    'compat/compat': 'off',
    // '@databricks/no-uncaught-localstorage-setitem': 'off',

    // Allow direct localStorage in tests
    // '@databricks/no-direct-storage': 'off',

    'react/no-unused-prop-types': 'off',

    // Allow RegExp lookbehind and negative lookbehind in tests as they only break Safari
    'no-lookahead-lookbehind-regexp/no-lookahead-lookbehind-regexp': 'off',

    // Allow hardcoded colors in tests for assertions
    // '@databricks/no-hardcoded-colors': 'off',
    // Allow hardcoded doc links in tests for assertions
    // '@databricks/no-hardcoded-doc-links': 'off',
    // Allow top-level dbguidelinks calls in tests for assertions
    // '@databricks/no-top-level-dbguidelinks-calls': 'off',
    // No need to warn about unstable nested components in tests
    // '@databricks/no-unstable-nested-components': 'off',
    // Allow direct QueryClientProvider in tests (use workspace scoped providers in app code)
    // '@databricks/no-query-client-provider': 'off',
    // Allow singleton QueryClient instances in tests
    // '@databricks/no-singleton-query-client': 'off',
    // Allow singleton ApolloClient and ApolloProvider in tests
    // '@databricks/no-singleton-apollo-client': 'off',
    // '@databricks/no-apollo-client-provider': 'off',
    // '@databricks/semantic-html-single-main': 'off',

    // '@databricks/no-use-react-table': ['error', { requireWrapper: false }],
  };

  // Rules for .test.ext files and .jest.ext files
  const testSuiteFileCommonRules = {
    // NOTE(FEINF-1838): Forbid exports in test files
    'jest/no-export': 'error',
  };

  const jestCommonRules = {
    // NOTE(FEINF-1783): Require Jest globals to be imported
    // '@databricks/no-restricted-globals-with-module': [
    //   'error',
    //   {
    //     afterAll: '@jest/globals',
    //     afterEach: '@jest/globals',
    //     beforeAll: '@jest/globals',
    //     beforeEach: '@jest/globals',
    //     describe: '@jest/globals',
    //     expect: '@jest/globals',
    //     it: '@jest/globals',
    //     jest: '@jest/globals',
    //     test: '@jest/globals',
    //   },
    // ],
    // NOTE(FEINF-4111): Disallow untyped jest.requireActual() calls
    // '@databricks/no-untyped-jest-require-actual': 'error',
    // NOTE(FEINF-4390): Disallow mocking window.{history,location} in tests
    // '@databricks/no-mock-location': 'error',
    // '@databricks/no-restricted-jest-mock-modules': 'error',
    // Consider any function prefixed with "expect" to be an assertion function
    // Also allow "expect" prefixed functions on objects like simpleSelectTestUtils.expect*
    'jest/expect-expect': ['error', { assertFunctionNames: ['expect*', '*.expect*'] }],
    // Allow conditional expect since there are legit use cases for it
    'jest/no-conditional-expect': 'off',
    'jest/no-alias-methods': 'error',
    // Some jest tests pass in variables to describe/it so we ignore the type of the describe/it name
    'jest/valid-title': ['error', { ignoreTypeOfDescribeName: true, ignoreTypeOfTestName: true }],
    'jest/prefer-jest-mocked': 'error',

    // NOTE(FEINF-4359): Disable this rule because the autofix logic for
    // `toHaveTextContent` is weird with how it handles non-literals.
    // See https://github.com/testing-library/eslint-plugin-jest-dom/issues/337.
    'jest-dom/prefer-to-have-text-content': 'off',
    // NOTE(FEINF-4359): Rules disabled temporarily
    'jest-dom/prefer-checked': 'off',
    'jest-dom/prefer-empty': 'off',
    'jest-dom/prefer-enabled-disabled': 'off',
    'jest-dom/prefer-focus': 'off',
    'jest-dom/prefer-required': 'off',
    'jest-dom/prefer-in-document': 'off',
    'jest-dom/prefer-to-have-attribute': 'off',
    'jest-dom/prefer-to-have-class': 'off',
    'jest-dom/prefer-to-have-style': 'off',
    'jest-dom/prefer-to-have-value': 'off',
  };

  const testOverrides = [
    {
      files: [
        '**/{test,testing}/**',
        '**/{test-util,testUtil}*',
        '**/{test-util,testUtil}*/**',
        '*TestUtil*',
        '*TestHelper*',
      ],
      rules: {
        ...testCommonRules,
        ...jestCommonRules,
        ...options.jestOverrideRules,
      },
    },
    {
      files: [...OverrideFiles.TEST, ...OverrideFiles.JEST_ONLY],
      excludedFiles: [...OverrideFiles.CYPRESS],
      extends: ['plugin:jest/recommended', 'plugin:jest-dom/recommended'],
      plugins: ['jest', 'testing-library'],
      env: {
        jest: true,
      },
      rules: {
        ...testCommonRules,
        ...jestCommonRules,
        ...testSuiteFileCommonRules,
        ...options.jestOverrideRules,
        'testing-library/no-debugging-utils': 'error',
        'testing-library/no-dom-import': 'error',
        'testing-library/await-async-utils': 'error',
      },
    },
  ];

  /** @type {ESLintConfig} */
  const config = {
    extends: [
      require.resolve('../base.js'),
      // Plugins from 3rd party packages
      'plugin:compat/recommended',
      'prettier',
      'plugin:jsx-a11y/recommended',
    ],
    ignorePatterns: ['__generated__', '/coverage', '/dist', '/dist-types', '/storybook-static'],
    rules: {
      'array-callback-return': 'error',
      curly: ['error', 'multi-line'],
      'default-case': ['error', { commentPattern: '^no default$' }],

      ...(!options.hasOxlint && {
        // Rules that are already covered by oxlint are only added if oxlint doesn't run.
        eqeqeq: ['error', 'smart'],
        'no-array-constructor': 'error',
        'no-caller': 'error',
        'no-const-assign': 'error',
      }),

      'new-parens': 'error',
      'no-cond-assign': ['error', 'except-parens'],
      'no-constant-binary-expression': 'error',
      'no-control-regex': 'error',
      'no-delete-var': 'error',
      'no-duplicate-case': 'error',
      'no-empty': ['error', { allowEmptyCatch: true }],
      'no-empty-character-class': 'error',
      'no-empty-pattern': 'error',
      'no-eval': 'error',
      'no-ex-assign': 'error',
      'no-extend-native': 'error',
      'no-extra-bind': 'error',
      'no-extra-label': 'error',
      'no-fallthrough': 'error',
      'no-func-assign': 'error',
      'no-implied-eval': 'error',
      'no-invalid-regexp': 'error',
      'no-iterator': 'error',
      'no-label-var': 'error',
      'no-labels': ['error', { allowLoop: true, allowSwitch: false }],
      'no-lone-blocks': 'error',
      'no-loop-func': 'error',
      'no-mixed-operators': [
        'error',
        {
          groups: [
            ['&', '|', '^', '~', '<<', '>>', '>>>'],
            ['==', '!=', '===', '!==', '>', '>=', '<', '<='],
            ['&&', '||'],
            ['in', 'instanceof'],
          ],
          allowSamePrecedence: false,
        },
      ],
      'no-multi-str': 'error',
      'no-global-assign': 'error',
      'no-unsafe-negation': 'error',
      'no-new-func': 'error',
      'no-new-object': 'error',
      'no-new-symbol': 'error',
      'no-new-wrappers': 'error',
      'no-obj-calls': 'error',
      'no-octal': 'error',
      'no-octal-escape': 'error',
      'no-redeclare': 'error',
      'no-regex-spaces': 'error',
      'no-restricted-syntax': [
        'error',
        'WithStatement',
        {
          selector:
            "CallExpression[callee.object.callee.name='getMonacoApi'][callee.property.name=/^register\\w+Provider$/], " +
            'CallExpression[callee.object.name=/[lL]anguages/][callee.property.name=/^register\\w+Provider$/]',
          message:
            "Use shared language provider instead: createLanguageProvider() from editor package's SharedLanguageProvider.ts",
        },
        {
          selector: "MemberExpression[object.name='jest'][property.name='retryTimes']",
          message:
            'jest.retryTimes is forbidden. Tests must pass without retries. If a test is flaky, fix it instead of adding retries.',
        },
        {
          selector: "MemberExpression[object.name='jest'][property.name='setTimeout']",
          message: 'jest.setTimeout is forbidden. Tests should use the default timeout.',
        },
      ],
      'no-script-url': 'error',
      'no-self-assign': 'error',
      'no-self-compare': 'error',
      'no-sequences': 'error',
      'no-shadow-restricted-names': 'error',
      'no-sparse-arrays': 'error',
      'no-template-curly-in-string': 'error',
      'no-this-before-super': 'error',
      'no-throw-literal': 'error',
      'no-trailing-spaces': 'error',
      'no-undef': 'error',
      'no-restricted-globals': [
        'error',
        ...restrictedGlobals,
        { name: 'history', message: 'Use @databricks/web-shared/routing to change routes.' },
        ...(options.noBrowserOnlyGlobals ? BROWSER_ONLY_GLOBALS : []),
      ],
      'no-unreachable': 'error',
      'no-unused-expressions': [
        'error',
        {
          allowShortCircuit: true,
          allowTernary: true,
          allowTaggedTemplates: true,
          enforceForJSX: true,
        },
      ],
      'no-unused-labels': 'error',
      'no-use-before-define': [
        'error',
        {
          functions: false,
          classes: false,
          variables: false,
        },
      ],
      'no-useless-computed-key': 'error',
      'no-useless-concat': 'error',
      'no-useless-constructor': 'error',
      'no-useless-escape': 'error',
      'no-useless-rename': [
        'error',
        {
          ignoreDestructuring: false,
          ignoreImport: false,
          ignoreExport: false,
        },
      ],
      'prefer-const': 'error',
      'require-yield': 'error',
      strict: ['error', 'never'],
      'unicode-bom': ['error', 'never'],
      'use-isnan': 'error',
      'valid-typeof': 'error',
      'no-restricted-properties': [
        'error',
        {
          object: 'require',
          property: 'ensure',
          message:
            'Please use import() instead. More info: https://facebook.github.io/create-react-app/docs/code-splitting',
        },
        {
          object: 'System',
          property: 'import',
          message:
            'Please use import() instead. More info: https://facebook.github.io/create-react-app/docs/code-splitting',
        },
        {
          object: 'window',
          property: 'history',
          message: 'Use @databricks/web-shared/routing to change routes.',
        },
      ],
      // '@databricks/no-import-use-sync-external-store': 'error',
      'no-restricted-imports': [
        'error',
        {
          paths: [
            {
              name: 'react',
              importNames: ['Suspense'],
              message: "Import Suspense from '@databricks/web-shared/react' instead of 'react'.",
            },
            {
              name: 'react-dom/test-utils',
              message: 'Use @testing-library/react instead.',
            },
            {
              name: 'lodash',
              importNames: ['default', 'chain'],
              message:
                'Default lodash imports are forbidden. Use specific function imports like "import { debounce } from \'lodash\'" or "import debounce from \'lodash/debounce\'" instead.',
            },
            {
              name: '@apollo/client',
              message:
                'Direct import of Apollo dependencies is not allowed. Please import from @databricks/web-shared/graphql instead. See go/data-fetching-client-usage-tracking for more details.',
            },
            {
              name: '@tanstack/react-query',
              message:
                'Direct import of Tanstack React Query dependencies is not allowed. Please import from @databricks/web-shared/query-client instead. See go/TO-DO for more details.',
            },
            ...(options.restrictReactQueryImports
              ? [
                  {
                    name: 'react-query',
                    importNames: ['useQuery', 'useMutation'],
                    message:
                      'Direct import of React Query hooks is not allowed, as we try to standardize UI data fetching onto Apollo. Please import from @databricks/web-shared/query-client instead. See go/standardized-data-fetching-client for more details.',
                  },
                ]
              : []),
          ],
          patterns: [
            {
              group: ['monaco-editor'],
              importNames: ['*', 'monaco'],
              message:
                'Please import individual objects instead of the whole package, especially outside of Unified Editor',
            },
          ],
        },
      ],

      // https://github.com/benmosher/eslint-plugin-import/tree/master/docs/rules
      'import/first': 'error',
      'import/no-amd': 'error',
      'import/no-anonymous-default-export': 'error',
      'import/no-empty-named-blocks': 'error',
      'import/no-useless-path-segments': 'error',
      'import/no-webpack-loader-syntax': 'error',

      // https://github.com/yannickcr/eslint-plugin-react/tree/master/docs/rules
      'react/forbid-foreign-prop-types': ['error', { allowInPropTypes: true }],
      'react/jsx-no-comment-textnodes': 'error',
      'react/jsx-no-duplicate-props': 'error',
      'react/jsx-no-target-blank': 'error',
      'react/jsx-no-undef': 'error',
      'react/jsx-pascal-case': [
        'error',
        {
          allowAllCaps: true,
          ignore: [],
        },
      ],
      'react/no-danger-with-children': 'error',
      'react/no-direct-mutation-state': 'error',
      'react/no-is-mounted': 'error',
      'react/no-typos': 'error',
      'react/require-render-return': 'error',
      'react/style-prop-object': 'error',
      // Disable because we still use v17 APIs
      'react/no-deprecated': 'off',

      // https://github.com/evcohen/eslint-plugin-jsx-a11y/tree/master/docs/rules
      'jsx-a11y/alt-text': 'error',
      'jsx-a11y/anchor-has-content': 'error',
      'jsx-a11y/aria-activedescendant-has-tabindex': 'error',
      'jsx-a11y/aria-props': 'error',
      'jsx-a11y/aria-proptypes': 'error',
      'jsx-a11y/aria-role': ['error', { ignoreNonDOM: true }],
      'jsx-a11y/aria-unsupported-elements': 'error',
      'jsx-a11y/heading-has-content': 'error',
      'jsx-a11y/iframe-has-title': 'error',
      'jsx-a11y/img-redundant-alt': 'error',
      'jsx-a11y/no-access-key': 'error',
      'jsx-a11y/no-distracting-elements': 'error',
      'jsx-a11y/no-redundant-roles': 'error',
      'jsx-a11y/role-has-required-aria-props': 'error',
      'jsx-a11y/role-supports-aria-props': 'error',
      'jsx-a11y/scope': 'error',

      // https://github.com/facebook/react/tree/main/packages/eslint-plugin-react-hooks
      'react-hooks/exhaustive-deps': 'error',
      'react-hooks/rules-of-hooks': 'error',

      // Except for "exhaustive-deps" and "rules-of-hooks", all the rest of the react-hooks rules
      // invoke the React compiler and incur a very large compute time. But the results of that
      // compilation are shared among all the rules, so linting will take the same amount of time no
      // matter how many are enabled. More info at https://github.com/facebook/react/issues/35395
      'react-hooks/set-state-in-render': 'error',

      /**
       * import rules
       */
      'import/no-extraneous-dependencies': ['error', { includeTypes: true }],
      'import/order': [
        'error',
        {
          groups: ['builtin', 'external', 'internal', ['parent', 'sibling', 'index'], 'unknown'],
          pathGroups: [
            {
              pattern: '@{databricks,redash,cypress-tests}/**',
              group: 'internal',
              position: 'before',
            },
            {
              pattern: '$*/**',
              group: 'internal',
              position: 'before',
            },
            {
              pattern: '@/**',
              group: 'internal',
            },
            ...options.importOrderOverridePathGroups,
          ],
          // Cannot include 'external', otherwise it will think "@databricks/..." and "@redash/..." are external imports.
          pathGroupsExcludedImportTypes: ['builtin'],
          'newlines-between': 'always',
          alphabetize: { order: 'asc', orderImportKind: 'asc' },
        },
      ],
      // Prevent duplicate imports (allowing for one regular "import" and one TypeScript
      // "import type" of the same module)
      'import/no-duplicates': 'error',

      /**
       * react rules
       */
      'react/self-closing-comp': 'error',
      'react/forbid-component-props': ['error', FORBIDDEN_PROPS],
      'react/forbid-dom-props': ['error', FORBIDDEN_PROPS],
      'react/forbid-elements': [
        'error',
        {
          forbid: [
            {
              element: 'DesignSystemThemeProvider',
              message:
                'Only to be accessed by SupportsDuBoisThemes, except for special exceptions like tests. Ask in #dubois first if you need to use it.',
            },
            {
              element: 'uses-legacy-bootstrap',
              message: 'Bootstrap is deprecated. Please use components from @databricks/design-system.',
            },
          ],
        },
      ],
      'react/jsx-boolean-value': 'error',
      'react/jsx-key': ['error', { checkFragmentShorthand: true, warnOnDuplicates: true }],
      // We use the new JSX transform: https://legacy.reactjs.org/blog/2020/09/22/introducing-the-new-jsx-transform.html
      'react/jsx-uses-react': 'off',
      'react/jsx-uses-vars': 'error',
      'react/no-danger': 'error',
      'react/no-did-mount-set-state': 'error',
      'react/no-did-update-set-state': 'error',
      'react/jsx-curly-brace-presence': ['error', { children: 'never', props: 'never', propElementValues: 'always' }],
      'react/no-unknown-property': ['error', { ignore: ['css'] }],
      // Require props to be typed either through TypeScript (recommended) or PropTypes (deprecated)
      // This rule should not be turned off
      // See the section on typing React components at go/eslint for more information
      'react/prop-types': 'error',
      // We use the new JSX transform: https://legacy.reactjs.org/blog/2020/09/22/introducing-the-new-jsx-transform.html
      'react/react-in-jsx-scope': 'off',

      /**
       * emotion rules
       */
      '@emotion/pkg-renaming': 'error',
      '@emotion/jsx-import': 'off', // autoimported by plugin
      '@emotion/no-vanilla': 'error',
      '@emotion/import-from-emotion': 'error',
      '@emotion/styled-import': 'error',

      /**
       * Misc js rules
       *
       * Some base rules are turned off to allow @typescript-eslint rule to take over.
       */
      'no-debugger': 'error',
      'no-extra-semi': 'error',
      'guard-for-in': 'error',
      // This rule uses root level .browserslistrc file to determine browser target
      'compat/compat': 'error',
      'id-denylist': 'error',
      'id-match': 'error',
      'max-nested-callbacks': 'error',
      'no-alert': 'error',
      // Console logging should only be used in development
      'no-console': 'error',
      'no-constant-condition': 'error',
      'no-div-regex': 'error',
      'no-multi-assign': 'error',
      'no-unmodified-loop-condition': 'error',
      'no-useless-call': 'error',

      // TODO: check if this already exists in jest
      'no-only-tests/no-only-tests': 'error',

      // Lookbehinds are not supported in Safari at the moment, and will cause a syntax error if used.
      'no-lookahead-lookbehind-regexp/no-lookahead-lookbehind-regexp': [
        'error',
        'no-lookbehind',
        'no-negative-lookbehind',
      ],

      'object-shorthand': ['error', 'methods'],

      /**
       * jsx-a11y rules
       */
      'jsx-a11y/anchor-is-valid': [
        'off', // TODO: remove after a11y PRs are merged
        {
          components: ['Link'],
          aspects: ['noHref', 'invalidHref', 'preferButton'],
        },
      ],
      'jsx-a11y/no-autofocus': 'off',
      'jsx-a11y/click-events-have-key-events': 'off', // TODO: turn on after tech debt is fixed
      'jsx-a11y/no-static-element-interactions': 'off', // TODO: turn on after tech debt is fixed
      'jsx-a11y/no-noninteractive-element-interactions': 'off', // TODO: turn on after tech debt is fixed
      'jsx-a11y/label-has-associated-control': 'off', // TODO: turn on after tech debt is fixed
      'jsx-a11y/interactive-supports-focus': 'off', // TODO: turn on after tech debt is fixed

      /*
       * formatjs rules
       */
      'formatjs/enforce-id': [
        // turn on to generate IDs upon autofix
        'off',
        {
          idInterpolationPattern: '[sha512:contenthash:base64:6]',
        },
      ],
      'formatjs/no-id': 'error', // prefers no id (will be generated)
      'formatjs/enforce-description': ['error', 'literal'],
      'formatjs/enforce-default-message': ['error', 'literal'],
      'formatjs/no-multiple-plurals': 'error',
      'formatjs/no-complex-selectors': ['error', { limit: 12 }],
      'formatjs/enforce-placeholders': [
        'error',
        {
          ignoreList: ['strong', 'em', 'p', 'span', 'div', 'b', 'label', 'i', 'ul', 'li'],
        },
      ],

      /**
       * databricks rules
       */
      // '@databricks/require-tool-schema-strict-mode': 'error',
      // '@databricks/no-direct-react-root': 'error',
      // '@databricks/no-direct-safe-flags-access': 'error',
      // '@databricks/no-preview-metadata-prefix-in-safex': 'error',
      // '@databricks/no-disable-lint': 'error',
      // '@databricks/no-double-negation': 'error',
      // '@databricks/no-global-uninitialized': 'error',
      // '@databricks/no-hardcoded-colors': 'error',
      // '@databricks/no-hardcoded-doc-links': 'error',
      // '@databricks/no-top-level-dbguidelinks-calls': 'error',
      // '@databricks/no-missing-react-hook-dependency-array': 'error',
      // Prevent direct usage of QueryClientProvider in app code
      // '@databricks/no-query-client-provider': options.allowQueryClientProvider ? 'off' : 'error',
      // Prevent singleton QueryClient instances - use workspace-scoped queryClientByWorkspace instead
      // '@databricks/no-singleton-query-client': 'error',
      // Prevent singleton ApolloClient and ApolloProvider usage (use workspace-scoped patterns instead)
      // '@databricks/no-singleton-apollo-client': 'error',
      // '@databricks/no-apollo-client-provider': 'error',
      // '@databricks/no-restricted-imports-regexp': [
      //   'error',
      //   {
      //     patterns: [...require('../shared/no-restricted-imports-regexp-base')],
      //   },
      // ],
      // '@databricks/avoid-manual-logerror': 'error',
      // '@databricks/no-uncaught-localstorage-setitem': 'error',
      // '@databricks/no-direct-storage': 'error',
      // '@databricks/no-wrapper-formui-label': 'error',
      // '@databricks/no-window-top': 'error',
      // '@databricks/no-dynamic-property-value': 'error',
      // '@databricks/no-new-object-or-array-in-zustand-selector': 'error',
      // '@databricks/no-unstable-nested-components': [
      //   'error',
      //   {
      //     allowAsPropsInElements: [
      //       { name: 'FormattedMessage', props: ['values'] },
      //       { name: 'PanelBoundary', props: ['fallbackRender'] },
      //       { name: 'Column', props: ['cellRenderer'] },
      //       { name: 'Table', props: ['noRowsRenderer', 'rowRenderer'] },
      //       { name: 'List', props: ['noRowsRenderer', 'rowRenderer'] },
      //       { name: 'Grid', props: ['noContentRenderer'] },
      //       { name: 'Collapse', props: ['expandIcon'] },
      //       { name: 'LegacySelect', props: ['dangerouslySetAntdProps.dropdownRender'] },
      //       { name: 'RHFControlledComponents.LegacySelect', props: ['dangerouslySetAntdProps.dropdownRender'] },
      //     ],
      //     allowAsPropsInFunctionCalls: ['formatMessage'],
      //   },
      // ],
      // '@databricks/no-out-of-root-relative-imports': 'error',
      // '@databricks/no-dollar-signs-in-jsxtext': 'error',
      // '@databricks/no-react-prop-types': 'error',
      // '@databricks/prefer-project-alias-imports': 'error',
      // '@databricks/no-unauthorized-lakeviewconfig-usage': 'error',
      // '@databricks/no-instanceof-apollo-error': 'error',
      // '@databricks/no-use-react-table': ['error', { requireWrapper: true }],
      // '@databricks/no-const-object-record-string': 'error',
      // '@databricks/react-lazy-only-at-top-level': 'error',
      // '@databricks/realtime-metric-labels-as-const': 'error',

      'no-unused-vars': 'off',
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          // Allow args prefixed with "_" to be unused
          // argsIgnorePattern: '^_',
          args: 'none',
          // Allow unused elements in destructure assignments (const [a, _unused, b] = x)
          destructuredArrayIgnorePattern: '^_',
          // Allow rest siblings to be unused
          ignoreRestSiblings: true,
          // Allow caught errors to be unused
          caughtErrors: 'none',
        },
      ],

      // NOTE(FEINF-4274): Prevent classes with only `static` members
      '@typescript-eslint/no-extraneous-class': 'error',
    },
    overrides: [
      // JS only rules. These are rules that are already covered by TS itself.
      {
        files: OverrideFiles.JS_JSX,
        rules: {
          'no-dupe-args': 'error',
          'no-dupe-class-members': 'error',
          'no-dupe-keys': 'error',
          'getter-return': 'error',
        },
      },
      {
        // Only run typescript-eslint on TS files
        files: OverrideFiles.TS,
        parser: '@typescript-eslint/parser',
        parserOptions: {
          ecmaVersion: 2018,
          sourceType: 'module',
          ecmaFeatures: {
            jsx: true,
          },

          // typescript-eslint specific options
          warnOnUnsupportedTypeScriptVersion: true,
        },
        plugins: ['@typescript-eslint'],
        extends: ['plugin:@typescript-eslint/recommended'],
        rules: {
          // TypeScript's `noFallthroughCasesInSwitch` option is more robust (#6906)
          'default-case': 'off',
          // 'tsc' already handles this (https://github.com/typescript-eslint/typescript-eslint/issues/291)
          'no-dupe-class-members': 'off',
          // 'tsc' already handles this (https://github.com/typescript-eslint/typescript-eslint/issues/477)
          'no-undef': 'off',

          // Add TypeScript specific rules (and turn off ESLint equivalents)
          '@typescript-eslint/consistent-type-assertions': 'error',
          'no-array-constructor': 'off',
          '@typescript-eslint/no-array-constructor': 'error',
          'no-redeclare': 'off',
          '@typescript-eslint/no-redeclare': 'error',
          'no-use-before-define': 'off',
          '@typescript-eslint/no-use-before-define': [
            'error',
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
              enforceForJSX: true,
            },
          ],
          'no-useless-constructor': 'off',
          '@typescript-eslint/no-useless-constructor': 'error',

          // Original rules from create-react-app preset

          // Do not require functions (especially react components) to have explicit returns
          '@typescript-eslint/explicit-function-return-type': 'off',
          // Allow the 'any' type to be used. tsconfig.json can be modified to be
          // stricter in the future to prevent 'any' from being used.
          '@typescript-eslint/no-explicit-any': 'off',
          // Allow empty functions in some contexts where the alternative is awkward and less
          // readable: e.g., () => {} and class FakeFoo { get() {} }
          'no-empty-function': 'off',
          '@typescript-eslint/no-empty-function': ['error', { allow: ['arrowFunctions', 'methods'] }],
          // Many API fields and generated types use camelcase
          '@typescript-eslint/naming-convention': 'off',
          // Require "import type ..." for type-only imports
          '@typescript-eslint/consistent-type-imports': ['error', { disallowTypeAnnotations: false }],

          // Disable checks for explicit module boundary types in all files. tsx will
          // provide the necessary type checks to ensure type safety. This rule could
          // potentially re-enabled in the future.
          '@typescript-eslint/explicit-module-boundary-types': 'off',

          // ts-migrate introduces a lot of ts-expect-error. turning into warning until we finalize the migration
          '@typescript-eslint/ban-ts-comment': 'error',

          // Allow empty interfaces that extend at least one other interface
          '@typescript-eslint/no-empty-object-type': ['error', { allowInterfaces: 'with-single-extends' }],
          '@typescript-eslint/no-unsafe-function-type': 'error',
          '@typescript-eslint/no-wrapper-object-types': 'error',
          '@typescript-eslint/no-restricted-types': [
            'error',
            {
              types: {
                object: {
                  message:
                    'Don\'t use `object` as a type unless you have a deep understanding of the TS type system. The `object` type is currently hard to use ([see this issue](https://github.com/microsoft/TypeScript/issues/21732)).\nConsider using `Record<string, unknown>` instead for "an object where all attributes aren\'t checked", as it allows you to more easily inspect and use the keys.\nHowever, it may be fine to use `object` for generic type requirements (`type MyType<T extends object> = ...`)',
                  fixWith: 'Record<string, unknown>',
                },
              },
            },
          ],

          // Banning parameter properties because they transpile differently in Babel/TSC/SWC. See FEINF-2235.
          '@typescript-eslint/parameter-properties': 'error',

          // We have many duplicated enum values with legit use cases.
          '@typescript-eslint/no-duplicate-enum-values': 'off',

          // Prevent non-null assertion operator (!) in codebase. This operator is unsafe and should be avoided.
          '@typescript-eslint/no-non-null-assertion': 'error',

          // Need set `no-unreachable` to `error` for TS files since `plugin:@typescript-eslint/recommended` disables the rule
          'no-unreachable': 'error',

          // This rule is redundant with TypeScript checks
          'react/prop-types': 'off',
        },
      },
      {
        files: OverrideFiles.JSX_TSX,
        rules: {
          'react/no-unused-prop-types': 'error',
        },
      },
      {
        files: [...OverrideFiles.CYPRESS],
        rules: {
          ...testCommonRules,
          '@databricks/no-uncaught-localstorage-setitem': 'off',
          '@databricks/no-hardcoded-colors': 'off',
          // Disable rule to support chai expect syntax (e.g. expect(foo).to.exist)
          '@typescript-eslint/no-unused-expressions': 'off',
        },
      },
      {
        files: OverrideFiles.GRAPHQL,
        rules: {
          // Disable no-trailing-spaces for graphql files because they are generated by
          // the @graphql-eslint processor and do not need to conform to this rule.
          'no-trailing-spaces': 'off',
        },
      },
      {
        files: OverrideFiles.STORYBOOK,
        rules: {
          // Allow direct localStorage in Storybook tests
          // '@databricks/no-direct-storage': 'off',
          // Disable no-non-null-assertion in Storybook files as there's no harm in hitting wrong assertions
          '@typescript-eslint/no-non-null-assertion': 'off',
          // Stories are often defined as anonymous default exports
          'import/no-anonymous-default-export': 'off',
          // Disable no-hardcoded-doc-links in Storybook files as they can hardcoded links.
          // '@databricks/no-hardcoded-doc-links': 'off',
          // Do not enforce QueryClientProvider rule in Storybook files
          // '@databricks/no-query-client-provider': 'off',
          // Allow singleton QueryClient instances in Storybook files
          // '@databricks/no-singleton-query-client': 'off',
          // Do not enforce Apollo singleton rules in Storybook files
          // '@databricks/no-singleton-apollo-client': 'off',
          // '@databricks/no-apollo-client-provider': 'off',
        },
      },
      {
        files: OverrideFiles.PLAYWRIGHT,
        extends: 'plugin:playwright/recommended',
        rules: {
          ...testCommonRules,
          ...playwrightConfig.rules,
        },
      },
      ...testOverrides,
    ],
  };

  if (options.eslint) {
    return options.eslint(config);
  }

  /**
   * Merges user (consumer package) ESLint config with base config
   * @param {ESLintConfig} userConfig
   * @return {ESLintConfig}
   */
  function mergeWithBaseConfig(userConfig = {}) {
    return {
      ...userConfig,
      extends: [...config.extends, ...(userConfig.extends || [])],
      ignorePatterns: [...config.ignorePatterns, ...(userConfig.ignorePatterns || [])],
      rules: {
        ...config.rules,
        ...userConfig.rules,
      },
      overrides: [...config.overrides, ...(userConfig.overrides || [])],
    };
  }

  // If no eslint function was provided, return the merge function so user does createConfig(options)(eslintConfig)
  // This is considered legacy. `eslint` is preferred.
  return mergeWithBaseConfig;
}

module.exports = {
  createConfigFactory,
};
