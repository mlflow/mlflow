import type { Config } from '@jest/types';

import getBaseConfig from '@databricks/config-jest/config';

const baseConfig = getBaseConfig({
  compiler: 'babel',
});

const config: Config.InitialOptions = {
  ...baseConfig,
};

process.env.TZ = 'UTC';

export default config;
