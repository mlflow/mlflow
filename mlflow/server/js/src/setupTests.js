/* eslint-disable no-undef -- FEINF-2715 - convert to TS */
import { configure } from 'enzyme';
import Adapter from '@wojtekmaj/enzyme-adapter-react-17';

const setupMockFetch = () => {
  // eslint-disable-next-line import/no-extraneous-dependencies, no-unreachable, global-require
  require('whatwg-fetch');
};

setupMockFetch();

configure({ adapter: new Adapter() });
// Included to mock local storage in JS tests, see docs at
// https://www.npmjs.com/package/jest-localstorage-mock#in-create-react-app
require('jest-localstorage-mock');

global.setImmediate = (cb) => {
  return setTimeout(cb, 0);
};
global.clearImmediate = (id) => {
  return clearTimeout(id);
};

// for plotly.js to work
//
window.URL.createObjectURL = function createObjectURL() {};

const testPath = expect.getState().testPath;
if (!testPath?.includes('.enzyme.')) {
  jest.mock('enzyme', () => {
    throw new Error('Enzyme is deprecated. Please use React Testing Library. go/deprecateenzyme');
  });
}

// Mock loadMessages which uses require.context from webpack which is unavailable in node.
jest.mock('./i18n/loadMessages', () => ({
  __esModule: true,
  DEFAULT_LOCALE: 'en',
  loadMessages: async (locale) => {
    if (locale.endsWith('unknown')) {
      return {};
    }
    return {
      // top-locale helps us see which merged message file has top precedence
      'top-locale': locale,
      [locale]: 'value',
    };
  },
}));

// Mock TelemetryClient which uses import.meta.url (not supported in Jest)
jest.mock('./telemetry/TelemetryClient', () => ({
  telemetryClient: {
    logEvent: jest.fn(),
    shutdown: jest.fn(),
    start: jest.fn(),
  },
}));

// Mock recharts components to avoid rendering issues in tests
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }) => <div data-testid="responsive-container">{children}</div>,
  LineChart: ({ children, data }) => (
    <div data-testid="line-chart" data-count={data?.length || 0}>
      {children}
    </div>
  ),
  BarChart: ({ children, data }) => (
    <div data-testid="bar-chart" data-count={data?.length || 0}>
      {children}
    </div>
  ),
  ComposedChart: ({ children, data }) => (
    <div data-testid="composed-chart" data-count={data?.length || 0}>
      {children}
    </div>
  ),
  AreaChart: ({ children, data }) => (
    <div data-testid="area-chart" data-count={data?.length || 0}>
      {children}
    </div>
  ),
  Line: ({ name }) => <div data-testid={name ? `line-${name}` : 'line'} />,
  Bar: ({ name }) => <div data-testid={name ? `bar-${name}` : 'bar'} />,
  Area: ({ name, dataKey }) => <div data-testid={`area-${dataKey}`} data-name={name} />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
  ReferenceLine: ({ label }) => <div data-testid="reference-line" data-label={label?.value} />,
}));

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Mock crypto API for tests
global.crypto = {
  randomUUID: () => 'test-uuid-' + Math.random().toString(36).substring(2, 15),
};

beforeEach(() => {
  // Prevent unit tests making actual fetch calls,
  // every test should explicitly mock all the API calls for the tested component.
  // Note: this needs to be mocked as a spy instead of a stub; otherwise we can't restore.
  // We need to restore fetch when testing graphql, otherwise Apollo throws an error before msw is
  // able to intercept the request.
  // Also note: jsdom does not have a global fetch, so we need to manually add a stub first.
  if (global.fetch === undefined) {
    global.fetch = () => {};
  }

  jest.spyOn(global, 'fetch').mockImplementation(() => {
    throw new Error('No API calls should be made from unit tests. Please explicitly mock all API calls.');
  });

  global.PerformanceObserver = class PerformanceObserver {
    callback;

    observe() {
      return null;
    }

    disconnect() {
      return null;
    }

    constructor(callback) {
      this.callback = callback;
    }
  };
});
