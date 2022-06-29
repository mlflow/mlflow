const path = require('path');
const util = require('util');

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
        path.resolve(__dirname, '../src/**/*'),
      ];
      ForkTsCheckerPlugin.options.typescript.configOverwrite.exclude = [
        path.resolve(__dirname, '../src/**/*.test.tsx'),
        path.resolve(__dirname, '../src/**/*.test.ts'),
      ];
    }

    /**
     * @emotion/react support.
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
