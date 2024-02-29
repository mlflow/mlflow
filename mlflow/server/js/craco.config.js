const url = require('url');
const path = require('path');
const fs = require('fs');
const TsconfigPathsPlugin = require('tsconfig-paths-webpack-plugin');
const webpack = require('webpack');

const proxyTarget = process.env.MLFLOW_PROXY;
const useProxyServer = !!proxyTarget && !process.env.MLFLOW_DEV_PROXY_MODE;

const isDevserverWebsocketRequest = (request) =>
  request.url === '/ws' && (request.headers.upgrade === 'websocket' || request.headers['sec-websocket-version']);

function mayProxy(pathname) {
  const publicPrefixPrefix = '/static-files/';
  if (pathname.startsWith(publicPrefixPrefix)) {
    const maybePublicPath = path.resolve('public', pathname.substring(publicPrefixPrefix.length));
    return !fs.existsSync(maybePublicPath);
  } else {
    const maybePublicPath = path.resolve('public', pathname.slice(1));
    return !fs.existsSync(maybePublicPath);
  }
}

/**
 * For 303's we need to rewrite the location to have host of `localhost`.
 */
function rewriteRedirect(proxyRes, req) {
  if (proxyRes.headers['location'] && proxyRes.statusCode === 303) {
    var u = url.parse(proxyRes.headers['location']);
    u.host = req.headers['host'];
    proxyRes.headers['location'] = u.format();
  }
}

/**
 * In Databricks, we send a cookie with a CSRF token and set the path of the cookie as "/mlflow".
 * We need to rewrite the path to "/" for the dev index.html/bundle.js to use the CSRF token.
 */
function rewriteCookies(proxyRes) {
  if (proxyRes.headers['set-cookie'] !== undefined) {
    const newCookies = [];
    proxyRes.headers['set-cookie'].forEach((c) => {
      newCookies.push(c.replace('Path=/mlflow', 'Path=/'));
    });
    proxyRes.headers['set-cookie'] = newCookies;
  }
}

/**
 * Since the base publicPath is configured to a relative path ("static-files/"),
 * the files referenced inside CSS files (e.g. fonts) can be incorrectly resolved
 * (e.g. /path/to/css/file/static-files/static/path/to/font.woff). We need to override
 * the CSS loader to make sure it will resolve to a proper relative path. This is
 * required for the production (bundled) builds only.
 */
function configureIframeCSSPublicPaths(config, env) {
  // eslint-disable-next-line prefer-const
  let shouldFixCSSPaths = env === 'production';

  if (!shouldFixCSSPaths) {
    return config;
  }

  let cssRuleFixed = false;
  config.module.rules
    .filter((rule) => rule.oneOf instanceof Array)
    .forEach((rule) => {
      rule.oneOf
        .filter((oneOf) => oneOf.test?.toString() === /\.css$/.toString())
        .forEach((cssRule) => {
          cssRule.use
            ?.filter((loaderConfig) => loaderConfig?.loader.match(/[\/\\]mini-css-extract-plugin[\/\\]/))
            .forEach((loaderConfig) => {
              loaderConfig.options = { publicPath: '../../' };
              cssRuleFixed = true;
            });
        });
    });

  if (!cssRuleFixed) {
    throw new Error('Failed to fix CSS paths!');
  }

  return config;
}

function enableOptionalTypescript(config) {
  /**
   * Essential TS config is already inside CRA's config - the only
   * missing thing is resolved extensions.
   */
  config.resolve.extensions.push('.ts', '.tsx');

  /**
   * We're going to exclude typechecking test files from webpack's pipeline
   */

  const ForkTsCheckerPlugin = config.plugins.find((plugin) => plugin.constructor.name === 'ForkTsCheckerWebpackPlugin');

  if (ForkTsCheckerPlugin) {
    ForkTsCheckerPlugin.options.typescript.configOverwrite.exclude = [
      '**/*.test.ts',
      '**/*.test.tsx',
      '**/*.stories.tsx',
    ].map((pattern) => path.join(__dirname, 'src', pattern));
  } else {
    throw new Error('Failed to setup Typescript');
  }

  return config;
}

function i18nOverrides(config) {
  // https://github.com/webpack/webpack/issues/11467#issuecomment-691873586
  config.module.rules.push({
    test: /\.m?js/,
    resolve: {
      fullySpecified: false,
    },
  });
  config.module.rules = config.module.rules.map((rule) => {
    if (rule.oneOf instanceof Array) {
      return {
        ...rule,
        oneOf: [
          {
            test: [new RegExp(path.join('src/i18n/', '.*json'))],
            use: [
              {
                loader: require.resolve('./I18nCompileLoader'),
              },
            ],
          },
          ...rule.oneOf,
        ],
      };
    }

    return rule;
  });

  return config;
}

module.exports = function () {
  const config = {
    babel: {
      env: {
        test: {
          plugins: [
            [
              require.resolve('babel-plugin-formatjs'),
              {
                idInterpolationPattern: '[sha512:contenthash:base64:6]',
                removeDefaultMessage: false,
              },
            ],
          ],
        },
      },
      presets: [
        [
          '@babel/preset-react',
          {
            runtime: 'automatic',
            importSource: '@emotion/react',
          },
        ],
      ],
      plugins: [
        [
          require.resolve('babel-plugin-formatjs'),
          {
            idInterpolationPattern: '[sha512:contenthash:base64:6]',
          },
        ],
        [
          require.resolve('@emotion/babel-plugin'),
          {
            sourceMap: false,
          },
        ],
      ],
    },
    devServer: {
      ...(useProxyServer && {
        hot: true,
        https: true,
        proxy: [
          // Heads up src/setupProxy.js is indirectly referenced by CRA
          // and also defines proxies.
          {
            context: function (pathname, request) {
              // Dev server's WS calls should not be proxied
              if (isDevserverWebsocketRequest(request)) {
                return false;
              }
              return mayProxy(pathname);
            },
            target: proxyTarget,
            secure: false,
            changeOrigin: true,
            ws: true,
            xfwd: true,
            onProxyRes: (proxyRes, req) => {
              rewriteRedirect(proxyRes, req);
              rewriteCookies(proxyRes);
            },
          },
        ],
        host: 'localhost',
        port: 3000,
        open: false,
      }),
      client: {
        overlay: {
          errors: false,
          warnings: false,
          runtimeErrors: (error) => {
            // It is safe to ignore based on https://stackoverflow.com/a/50387233/12110203.
            if (error?.message.match(/ResizeObserver/i)) {
              return false;
            }
            return true;
          },
        },
      },
    },
    jest: {
      configure: (jestConfig) => {
        /*
         * Jest running on the currently used node version is not yet capable of ESM processing:
         * https://jestjs.io/docs/ecmascript-modules
         * https://nodejs.org/api/vm.html#vm_class_vm_module
         *
         * Since there are certain ESM-built dependencies used in MLFLow, we need
         * to add a few exceptions to the standard ignore pattern for babel.
         */
        const createIgnorePatternForESM = () => {
          // List all the modules that we *want* to be transpiled by babel
          const transpileModules = [
            '@databricks/design-system',
            '@babel/runtime/.+?/esm',
            '@ant-design/icons',
            '@ant-design/icons-svg',
          ];

          // We'll ignore only dependencies in 'node_modules' directly within certain
          // directories in order to avoid false positive matches in nested modules.
          const validNodeModulesRoots = ['mlflow/web/js'];

          // prettier-ignore
          // eslint-disable-next-line max-len
          return `(${validNodeModulesRoots.join('|')})\\/node_modules\\/((?!(${transpileModules.join('|')})).)+(js|jsx|mjs|cjs|ts|tsx|json)$`;
        };

        jestConfig.resetMocks = false; // ML-20462 Restore resetMocks
        jestConfig.collectCoverageFrom = ['src/**/*.{js,jsx}', '!**/*.test.{js,jsx}', '!**/__tests__/*.{js,jsx}'];
        jestConfig.coverageReporters = ['lcov'];
        jestConfig.setupFiles = ['jest-canvas-mock'];
        jestConfig.setupFilesAfterEnv.push('<rootDir>/scripts/env-mocks.js');
        jestConfig.setupFilesAfterEnv.push('<rootDir>/scripts/setup-jest-dom-matchers.js');
        // Adjust config to work with dependencies using ".mjs" file extensions
        jestConfig.moduleFileExtensions.push('mjs');
        // Remove when this issue is resolved: https://github.com/gsoft-inc/craco/issues/393
        jestConfig.transform = {
          '\\.[jt]sx?$': ['babel-jest', { configFile: './jest.babel.config.js' }],
          ...jestConfig.transform,
        };
        jestConfig.transformIgnorePatterns = ['\\.pnp\\.[^\\/]+$', createIgnorePatternForESM()];
        jestConfig.globalSetup = '<rootDir>/scripts/global-setup.js';

        const moduleNameMapper = {
          ...jestConfig.moduleNameMapper,
          '@databricks/web-shared/(.*)': '<rootDir>/src/shared/web-shared/$1',
        };

        jestConfig.moduleNameMapper = moduleNameMapper;
        jestConfig.setupFilesAfterEnv = [
          '<rootDir>/scripts/jest/react-versions/index.ts',
          ...(jestConfig.setupFilesAfterEnv || []),
        ];

        return jestConfig;
      },
    },
    webpack: {
      configure: (webpackConfig, { env }) => {
        webpackConfig.output.publicPath = 'static-files/';
        webpackConfig = i18nOverrides(webpackConfig);
        webpackConfig = configureIframeCSSPublicPaths(webpackConfig, env);
        webpackConfig = enableOptionalTypescript(webpackConfig);
        webpackConfig.resolve = {
          ...webpackConfig.resolve,
          plugins: [new TsconfigPathsPlugin(), ...webpackConfig.resolve.plugins],
          alias: {
            ...webpackConfig.resolve.alias,
            // Fix integration with react 18 and react-dnd@15
            // https://github.com/react-dnd/react-dnd/issues/3433#issuecomment-1102144912
            'react/jsx-runtime.js': require.resolve('react/jsx-runtime'),
            'react/jsx-dev-runtime.js': require.resolve('react/jsx-dev-runtime'),
          },
        };
        console.log('Webpack config:', webpackConfig);
        return webpackConfig;
      },
      plugins: [
        new webpack.EnvironmentPlugin({
          HIDE_HEADER: process.env.HIDE_HEADER ? 'true' : 'false',
          HIDE_EXPERIMENT_LIST: process.env.HIDE_EXPERIMENT_LIST ? 'true' : 'false',
          SHOW_GDPR_PURGING_MESSAGES: process.env.SHOW_GDPR_PURGING_MESSAGES ? 'true' : 'false',
          USE_ABSOLUTE_AJAX_URLS: process.env.USE_ABSOLUTE_AJAX_URLS ? 'true' : 'false',
          SHOULD_REDIRECT_IFRAME: process.env.SHOULD_REDIRECT_IFRAME ? 'true' : 'false',
        }),
      ],
    },
  };
  return config;
};
