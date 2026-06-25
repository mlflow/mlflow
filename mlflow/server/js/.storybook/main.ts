import type { StorybookConfig } from '@storybook/react-webpack5';
import { config as defaultConfig } from '@databricks/config-storybook';

const config: StorybookConfig = {
  ...defaultConfig,
  stories: ['../src/**/*.stories.@(js|jsx|ts|tsx)'],
};

export default config;
