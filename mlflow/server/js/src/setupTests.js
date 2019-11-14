import { configure } from 'enzyme';
import Adapter from 'enzyme-adapter-react-16';

configure({ adapter: new Adapter() });
// Included to mock local storage in JS tests, see docs at
// https://www.npmjs.com/package/jest-localstorage-mock#in-create-react-app
require('jest-localstorage-mock');

// for plotly.js to work
//
window.URL.createObjectURL = function createObjectURL() {};
