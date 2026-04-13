const noAbsoluteAjaxUrls = require('./no-absolute-ajax-urls');
const componentIdMustBeInRegistry = require('./componentId-must-be-in-registry');

module.exports = {
  rules: {
    'no-absolute-ajax-urls': noAbsoluteAjaxUrls,
    'componentId-must-be-in-registry': componentIdMustBeInRegistry,
  },
  configs: {
    recommended: {
      plugins: ['@mlflow'],
      rules: {
        '@mlflow/no-absolute-ajax-urls': 'error',
        '@mlflow/componentId-must-be-in-registry': 'error',
      },
    },
  },
};
