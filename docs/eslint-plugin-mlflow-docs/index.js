/**
 * ESLint plugin for MLflow documentation with custom rules.
 *
 * @type {import('eslint').ESLint.Plugin}
 */
module.exports = {
  rules: {
    /** Rule to validate NotebookDownloadButton URLs */
    'valid-notebook-url': require('./rules/valid-notebook-url'),
    /** Rule to detect raw image paths that should use useBaseUrl */
    'use-base-url-for-images': require('./rules/use-base-url-for-images'),
    /** Rule to enforce <APILink> usage when referencing the API doc **/
    'prefer-apilink-component': require('./rules/prefer-apilink-component'),
  },
};
