import { configure } from 'enzyme';
import Adapter from '@wojtekmaj/enzyme-adapter-react-17';

configure({ adapter: new Adapter() });
// Included to mock local storage in JS tests, see docs at
// https://www.npmjs.com/package/jest-localstorage-mock#in-create-react-app
require('jest-localstorage-mock');

// for plotly.js to work
//
window.URL.createObjectURL = function createObjectURL() {};

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

beforeEach(() => {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: jest.fn().mockImplementation((query) => ({
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
});
