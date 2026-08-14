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
    // babel-preset-react-app predates the ES2022 "static {}" class block syntax
    // used by @a2ui/web_core's source (its exports map has no pre-built dist).
    require.resolve('@babel/plugin-transform-class-static-block'),
  ],
};
