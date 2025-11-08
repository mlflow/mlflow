const { defineConfig } = require('eslint/config');
const docusaurusEslintPlugin = require('@docusaurus/eslint-plugin');
const mlflowDocsPlugin = require('./eslint-plugin-mlflow-docs');
const js = require('@eslint/js');
const { FlatCompat } = require('@eslint/eslintrc');

const compat = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: js.configs.recommended,
  allConfig: js.configs.all,
});

module.exports = defineConfig([
  {
    files: ['**/*.md', '**/*.mdx'],
    extends: compat.extends('plugin:mdx/recommended'),
    plugins: {
      '@docusaurus': docusaurusEslintPlugin,
      'mlflow-docs': mlflowDocsPlugin,
    },
    rules: {
      '@docusaurus/no-html-links': 'error',
      'mlflow-docs/valid-notebook-url': 'error',
      '@docusaurus/no-html-links': 'error',
      'mlflow-docs/valid-notebook-url': 'error',
      'mlflow-docs/use-base-url-for-images': 'error',
      'mlflow-docs/prefer-apilink-component': 'error',
    },
  },
]);
