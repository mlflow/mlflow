const noAbsoluteAjaxUrls = require('./no-absolute-ajax-urls');
const noWebSharedMlflowImports = require('./no-web-shared-mlflow-imports');

module.exports = {
  rules: {
    'no-absolute-ajax-urls': noAbsoluteAjaxUrls,
    'no-web-shared-mlflow-imports': noWebSharedMlflowImports,
  },
  configs: {
    recommended: {
      plugins: ['@mlflow'],
      rules: {
        '@mlflow/no-absolute-ajax-urls': 'error',
        '@mlflow/no-web-shared-mlflow-imports': 'error',
      },
    },
  },
};
