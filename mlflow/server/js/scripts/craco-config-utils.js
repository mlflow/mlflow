/* eslint-disable no-param-reassign */
/* eslint-disable import/no-extraneous-dependencies */
/* eslint-disable import/no-nodejs-modules */
/* eslint-disable max-len */
const path = require('path');

/**
 * Configures craco to emit a library that mounts MLflow as a HTML Web Component
 * instead of standalone web application.
 */
function configureWebComponentOutput(webpackConfig) {
  // Pass through the configuration object if build process is not
  // configured to emit a library
  if (process.env.MLFLOW_BUILD_WC_LIBRARY !== 'true') {
    return webpackConfig;
  }

  // Update entry point
  webpackConfig.entry = {
    mlflow: path.join(__dirname, '../src', '/mfe/MLFlowWebComponent.tsx'),
  };

  // Change output to "library" type
  webpackConfig.output = {
    path: path.join(__dirname, '../build'),
    globalObject: 'this',
    assetModuleFilename: '[name][ext]',
    chunkFilename: '[name].[contenthash:8].chunk.js',
    filename: '[name].js',
    library: {
      type: 'umd',
      name: 'mlflow',
    },
  };

  // Remove unused plugins
  webpackConfig.plugins = webpackConfig.plugins
    .filter((p) => p.constructor.name !== 'MiniCssExtractPlugin')
    .filter((p) => p.constructor.name !== 'WebpackManifestPlugin')
    .filter((p) => p.constructor.name !== 'InlineChunkHtmlPlugin')
    .filter((p) => p.constructor.name !== 'HtmlWebpackPlugin');

  // Reconfigure style loaders to use dynamic `style-loader` instead of CSS files
  webpackConfig.module.rules
    .filter((rule) => rule.oneOf instanceof Array)
    .forEach((rule) => {
      rule.oneOf
        .filter((oneOf) => oneOf.test?.toString() === /\.css$/.toString())
        .forEach((oneOf) => {
          // Remove default MiniCssExtractPlugin loader
          oneOf.use.shift();

          // Add 'style-loader' that injects styles into `mlflow-ui` element
          oneOf.use.unshift({
            loader: require.resolve('style-loader'),
            options: {
              insert: (styleTag) => {
                const mfeElement = 'mlflow-ui';
                window.customElements.whenDefined(mfeElement).then(function inject() {
                  const WebComponentClass = window.customElements.get(mfeElement);
                  WebComponentClass.webpackInjectStyle(styleTag);
                });
              },
            },
          });
        });
    });

  return webpackConfig;
}

module.exports = {
  configureWebComponentOutput,
};
