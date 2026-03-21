const path = require('path');

module.exports = {
  stories: ['../src/**/*.stories.@(js|jsx|ts|tsx)'],
  addons: [
    '@storybook/addon-links',
    '@storybook/addon-essentials',
    '@storybook/preset-create-react-app',
  ],
  framework: '@storybook/react',
  core: {
    builder: 'webpack5',
  },
  webpackFinal: (config) => {
    /**
     * Setting proper tsconfig file for the ForkTsChecker plugin
     */
    const ForkTsCheckerPlugin = config.plugins.find(
      (plugin) => plugin.constructor.name === 'ForkTsCheckerWebpackPlugin',
    );
    if (ForkTsCheckerPlugin) {
      ForkTsCheckerPlugin.options.typescript.configOverwrite.include = [
        path.resolve(__dirname, '../src/**/*.d.ts'),
        path.resolve(__dirname, '../src/**/*.stories.tsx'),
      ];
    }

    // Browserifying "stream" package, as in craco.config.js file
    config.resolve.fallback = {
      ...config.resolve.fallback,
      stream: require.resolve('stream-browserify'),
    };

    /**
     * Adding @emotion/react and formatjs support here.
     *
     * We're pushing additional babel-loader rule to the end of
     * the processing chain instead of messing up with existing
     * entry due to importance of the loader precedence.
     * See https://github.com/storybookjs/storybook/issues/7540
     */
    config.module.rules.push({
      test: /\.[tj]sx?$/,
      include: path.resolve(__dirname, '../src'),
      loader: require.resolve('babel-loader'),
      options: {
        presets: [require.resolve('@emotion/babel-preset-css-prop')],
        plugins: [
          ['react-require'],
          [
            require.resolve('babel-plugin-formatjs'),
            {
              idInterpolationPattern: '[sha512:contenthash:base64:6]',
            },
          ],
        ],
        overrides: [
          {
            test: /\.tsx?$/,
            presets: [[require.resolve('@babel/preset-typescript')]],
          },
        ],
      },
    });
    return config;
  },
};
