const { defineConfig } = require('eslint/config');
const docusaurusEslintPlugin = require('@docusaurus/eslint-plugin');
const mlflowDocsPlugin = require('./eslint-plugin-mlflow-docs');
const js = require('@eslint/js');
const { FlatCompat } = require('@eslint/eslintrc');
const unusedImports = require('eslint-plugin-unused-imports');
const reactPlugin = require('eslint-plugin-react');

// Prevent auto-fixing MDX files as it can corrupt file contents
// Can be bypassed by setting MLFLOW_DOCS_ALLOW_ESLINT_FIX=1 environment variable
if (process.argv.includes('--fix') && !process.env.MLFLOW_DOCS_ALLOW_ESLINT_FIX) {
  throw new Error(
    'ESLint --fix is disabled for MDX files because it can corrupt file contents ' +
      '(e.g., removing heading markers like # or inconsistent semicolon handling).\n\n' +
      'If you want to use auto-fix anyway:\n' +
      '1. Set the environment variable: MLFLOW_DOCS_ALLOW_ESLINT_FIX=1\n' +
      '2. Run: MLFLOW_DOCS_ALLOW_ESLINT_FIX=1 eslint --fix\n' +
      '3. Carefully review ALL changes before committing\n' +
      '4. Manually restore any corrupted content\n\n' +
      'Otherwise, please fix issues manually.',
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
      // These React rules prevent false positives in MDX files where JSX components
      // (like <Tabs>, <TabItem>) are used but ESLint doesn't recognize them as "used"
      'react/jsx-uses-vars': 'error',
      'react/jsx-uses-react': 'error',
    },
  },
]);
