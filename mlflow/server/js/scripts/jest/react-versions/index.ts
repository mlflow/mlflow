// Intentionally not importing `jest` from `@jest/globals` because the transpilation of this library will
// cause `jest.mock` to become something like `jest_1.default.mock` which doesn't get processed by jest.
// `jest` cannot be required either due to https://github.com/jestjs/jest/issues/9920
// Restore after FEINF-2744
import { beforeAll, afterAll } from '@jest/globals';

// Prefix with mock_ so it can be accessed inside jest.mock properly.
// Use `require` to avoid rename during transpilation.
// Restore after FEINF-2744
const mock_resolve =
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  require('resolve') as unknown as typeof import('resolve');

const { currentEntry: mock_currentEntry } =
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  require('./user-settings') as unknown as typeof import('./user-settings');

type RTL18 = typeof import('@testing-library/react');

if (mock_currentEntry && mock_currentEntry.reactVersion === 17) {
  jest.mock('react', () => ({
    ...require('react-17'),
    default: require('react-17'),
    __esModule: true,
  }));
  jest.mock('react/jsx-dev-runtime', () => ({
    ...require('react-17/jsx-dev-runtime'),
    default: require('react-17/jsx-dev-runtime'),
    __esModule: true,
  }));
  jest.mock('react/jsx-runtime', () => ({
    ...require('react-17/jsx-runtime'),
    default: require('react-17/jsx-runtime'),
    __esModule: true,
  }));
  jest.mock('react-dom', () => ({
    ...require('react-dom-17'),
    default: require('react-dom-17'),
    __esModule: true,
  }));
  jest.mock('react-dom/server', () => ({
    ...require('react-dom-17/server'),
    default: require('react-dom-17/server'),
    __esModule: true,
  }));
  jest.mock('react-dom/test-utils', () => ({
    ...require('react-dom-17/test-utils'),
    default: require('react-dom-17/test-utils'),
    __esModule: true,
  }));
  jest.mock('react-test-renderer', () => ({
    ...require('react-test-renderer-17'),
    default: require('react-test-renderer-17'),
    __esModule: true,
  }));
  jest.mock('react-test-renderer/shallow', () => ({
    ...require('react-test-renderer-17/shallow'),
    default: require('react-test-renderer-17/shallow'),
    __esModule: true,
  }));

  jest.mock('@testing-library/react', () => {
    throw new Error('@testing-library/react should not be imported when running tests with React 17');
  });
} else {
  // React 18
  if (mock_currentEntry.rtlVersion === 14) {
    jest.mock('@testing-library/dom', () => {
      // Get the path for @testing-library/react v14
      const rtlPath = require.resolve('@testing-library/react');
      // Force resolving of @testing-library/dom relative to the v14 version of RTL
      // so that @testing-library/user-event resolves the same version of the package
      const domPath = mock_resolve.sync('@testing-library/dom', { basedir: rtlPath });
      return {
        ...require(domPath),
        default: require(domPath),
        __esModule: true,
      };
    });
  }

  // Eventually this needs to be turned on, but along with migrating to RTL 14.
  // Some tests still pass with @testing-library/react-hooks and React 18.
  // jest.mock('@testing-library/react-hooks', () => {
  //   throw new Error(
  //     '@testing-library/react-hooks should not be imported when running tests with React 18. Use @testing-library/react instead.',
  //   );
  // });

  const { configure: configureReactTestingLibraryForReact18 } = require('@testing-library/react');

  configureReactTestingLibraryForReact18({
    asyncUtilTimeout: process.env.BAZEL_YARN ? 30000 : 1000,
  });

  // All tests in a single suite (file) that run with React 18 should have IS_REACT_ACT_ENVIRONMENT=true
  // This is a React 18 specific global, see
  // https://react.dev/blog/2022/03/08/react-18-upgrade-guide#configuring-your-testing-environment
  // Normally, this is done by @testing-library/react itself but we don't load v14 in all tests.
  let prev_IS_REACT_ACT_ENVIRONMENT: boolean;
  beforeAll(() => {
    prev_IS_REACT_ACT_ENVIRONMENT = (global as any).IS_REACT_ACT_ENVIRONMENT;
    (global as any).IS_REACT_ACT_ENVIRONMENT = true;
    (global as any).USE_REACT_18_IN_TEST = true;
  });

  afterAll(() => {
    (global as any).IS_REACT_ACT_ENVIRONMENT = prev_IS_REACT_ACT_ENVIRONMENT;
    (global as any).USE_REACT_18_IN_TEST = false;
  });
}
