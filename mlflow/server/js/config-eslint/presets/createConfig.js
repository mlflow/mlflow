require('@rushstack/eslint-patch/modern-module-resolution');

const { createConfigFactory } = require('./createConfigFactory');

function createConfig(options = {}) {
  const config = {
    extends: [require.resolve('../base.js')],
    ...createConfigFactory(options)(),
  };

  return config;
}

module.exports = {
  createConfig,
};
