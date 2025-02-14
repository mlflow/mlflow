import PostcssFilterRules from 'postcss-filter-rules';

import type { PkgrConfig, RollupOptions } from '@databricks/pkgr';

import antDVars from './src/antd-vars';
import { lessCustomFileManager } from './src/rollup/less';
import { getLessVariablesObject } from './src/theme/convertToLessVars';

const styleVariations: { isDark: boolean; outputFileName: string }[] = [
  { isDark: false, outputFileName: 'index.css' },
  { isDark: true, outputFileName: 'index-dark.css' },
];

const config: PkgrConfig[] = [
  // TODO: enable when FEINF-4043 is done as internal-tools' next.js cannot read design system sources directly
  // process.env.DATABRICKS_DESIGN_SYSTEM_BUILD_SOURCES === 'true' && {
  {
    name: 'main',
    input: {
      index: 'src/index.ts',
      development: 'src/development/index.ts',
      patterns: 'src/~patterns/index.ts',
      'test-utils/enzyme': 'src/test-utils/enzyme/index.ts',
      'test-utils/rtl': 'src/test-utils/rtl/index.ts',
    },
  },
  ...styleVariations.map<PkgrConfig>(({ isDark, outputFileName: extractCSSOutputFileName }) => {
    const prefix = isDark ? `${antDVars['ant-prefix']}-dark` : `${antDVars['ant-prefix']}-light`;

    const lessVars = getLessVariablesObject(isDark);

    const plugins = [
      // This plugin strips out global-scoped AntD styles that will break certain applications.
      // Follow https://github.com/ant-design/ant-design/issues/9363 in case AntD fixes this.
      PostcssFilterRules({
        filter: (selector: any) => {
          // Remove -rtl classnames (ones that are not removed by lessCustomFileManager)
          if (selector.match(/-rtl\b/)) {
            return false;
          }

          if (selector.includes(antDVars['ant-prefix']) || /\.anticon|data-|\.ant-motion|^(\d)*%$/.test(selector)) {
            return true;
          }

          return false;
        },
      }),
    ].filter(Boolean);

    return {
      name: `style-${isDark ? 'dark' : 'light'}`,
      input: ['src/styles.js'],
      lessOptions: {
        javascriptEnabled: true,
        modifyVars: {
          ...lessVars,
          'ant-prefix': prefix,
        },
        plugins: [lessCustomFileManager],
      },
      cssFilename: extractCSSOutputFileName,
      postcssPlugins: plugins,
      nodeExternals: {
        // Intentionally don't consider antd imports as externals when building .css files, because
        // we want to include their content in the output.
        exclude: [/^antd($|\/)/],
      },
      emitSourceMaps: false,

      rollup(options) {
        // TODO: remove styles.js from the output
        return {
          ...options,

          onwarn(warning, warn) {
            // Prevent "Generated an empty chunk:" warnings for styles.js
            if (warning.code === 'EMPTY_BUNDLE') {
              return;
            }
            warn(warning);
          },
        } satisfies RollupOptions;
      },
    } satisfies PkgrConfig;
  }),
].filter(Boolean) as PkgrConfig[];

export default config;
