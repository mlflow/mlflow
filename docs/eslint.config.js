const { defineConfig } = require('eslint/config');
const docusaurusEslintPlugin = require('@docusaurus/eslint-plugin');
const mlflowDocsPlugin = require('./eslint-plugin-mlflow-docs');
const js = require('@eslint/js');
const { FlatCompat } = require('@eslint/eslintrc');
const unusedImports = require('eslint-plugin-unused-imports');
const reactPlugin = require('eslint-plugin-react');

// Prevent autofixing as it can corrupt file contents
if (process.argv.includes('--fix') && !process.env.MLFLOW_DOCS_ALLOW_ESLINT_FIX) {
  throw new Error(
    'ESLint autofix is disabled because it can corrupt file contents ' +
      '(e.g., https://github.com/sweepline/eslint-plugin-unused-imports/issues/115). ' +
      'If you want to use auto-fix anyway, run this command and ' +
      'carefully review ALL changes before committing:\n\n' +
      'MLFLOW_DOCS_ALLOW_ESLINT_FIX=1 npm run eslint -- --fix',
  );
}

const compat = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: js.configs.recommended,
  allConfig: js.configs.all,
});

module.exports = defineConfig([
  {
    ignores: ['**/*-ipynb.mdx'],
  },
  {
    files: ['**/*.md', '**/*.mdx'],
    extends: compat.extends('plugin:mdx/recommended'),
    plugins: {
      '@docusaurus': docusaurusEslintPlugin,
      'mlflow-docs': mlflowDocsPlugin,
      'unused-imports': unusedImports,
      react: reactPlugin,
    },
    settings: {
      'mdx/code-blocks': true,
      react: {
        version: 'detect',
      },
    },
    rules: {
      '@docusaurus/no-html-links': 'error',
      'mlflow-docs/valid-notebook-url': 'error',
      'mlflow-docs/use-base-url-for-images': 'error',
      'mlflow-docs/prefer-apilink-component': 'error',
      'unused-imports/no-unused-imports': 'error',
      // These React rules prevent component imports from being flagged as unused.
      // Required when using eslint-plugin-unused-imports with JSX/React code.
      // https://www.npmjs.com/package/eslint-plugin-unused-imports
      'react/jsx-uses-vars': 'error',
      'react/jsx-uses-react': 'error',
    },
  },
]);
