const base = require('../config');

/**
 * These preset should be used by libraries that are only used in Node and not in the browser.
 */
module.exports = {
  ...base,
  rules: {
    ...base.rules,
    // Node libraries don't need browser compatibility rules
    'compat/compat': 'off',
  },
};
