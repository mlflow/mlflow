require('@rushstack/eslint-patch/modern-module-resolution');

const config = {
  root: true,
  parser: '@babel/eslint-parser',
  parserOptions: {
    sourceType: 'module',
    requireConfigFile: false,
    ecmaFeatures: {
      jsx: true,
    },
    ecmaVersion: 12,
    babelOptions: {
      presets: [require.resolve('@babel/preset-react')],
    },
  },
  extends: [
    'plugin:compat/recommended',
    'prettier',
    'plugin:jsx-a11y/recommended',
    'plugin:import/typescript',
    'plugin:no-lookahead-lookbehind-regexp/recommended',
    'plugin:react-component-name/recommended',
  ],
  plugins: [
    '@databricks', // Points to @databricks/eslint-plugin
    '@emotion',
    '@typescript-eslint',
    'compat',
    'file-progress',
    'formatjs',
    'import',
    'jest',
    'jsx-a11y',
    'no-only-tests',
    'react-component-name',
    'react-hooks',
    'react',
  ],
  settings: {
    react: {
      version: 'detect',
    },
    jest: {
      version: 'latest',
    },
    'import/ignore': ['node_modules'],
    'import/resolver': {
      typescript: {},
    },
    // Used by compat/compat to ignore compatibility issues with polyfilled APIs.
    // Polyfills are defined in js/packages/polyfill package.
    polyfills: ['requestIdleCallback'],
  },
  env: {
    browser: true,
    commonjs: true,
    es6: true,
    node: true,
  },
};

module.exports = config;
