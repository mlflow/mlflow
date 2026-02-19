/**
 * Babel config for running Jest tests
 */

module.exports = {
  presets: [
    [
      require.resolve('babel-preset-react-app'),
      {
        runtime: 'automatic',
      },
    ],
  ],
  plugins: [
    [
      require.resolve('babel-plugin-formatjs'),
      {
        idInterpolationPattern: '[sha512:contenthash:base64:6]',
        removeDefaultMessage: false,
      },
    ],
  ],
};
