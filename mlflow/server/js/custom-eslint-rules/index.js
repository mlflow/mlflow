const noAbsoluteAjaxUrls = require('./no-absolute-ajax-urls');

module.exports = {
  rules: {
    'no-absolute-ajax-urls': noAbsoluteAjaxUrls,
  },
  configs: {
    recommended: {
      plugins: ['mlflow'],
      rules: {
        'mlflow/no-absolute-ajax-urls': 'error',
      },
    },
  },
};
