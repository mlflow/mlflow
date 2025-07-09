/**
 * ESLint plugin for MLflow documentation with custom rules.
 *
 * @type {import('eslint').ESLint.Plugin}
 */
module.exports = {
  rules: {
    /** Rule to validate NotebookDownloadButton URLs */
    'valid-notebook-url': require('./rules/valid-notebook-url'),
  },
};
