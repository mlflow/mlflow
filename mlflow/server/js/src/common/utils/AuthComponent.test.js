import React from 'react';
import { shallow } from 'enzyme';
import { AuthComponentImpl } from './AuthComponent';
import { setup_mock, teardown_mock } from '../../../__mocks__/xhr-mock';

let location;

beforeEach(() => {
  location = {};
  setup_mock();
});

afterEach(() => {
  teardown_mock();
});

const getAuthComponentMock = () => {
  return shallow(<AuthComponentImpl location={location} />);
};

test('Token should be in local storage after render', () => {
  location.search = 'code=code';
  XMLHttpRequest.mockImplementation(() => ({
    open: jest.fn(),
    send: jest.fn(),
    status: 200,
    getResponseHeader: jest.fn(() => 'token'),
  }));
  getAuthComponentMock().instance();
  expect(localStorage.getItem('token')).toEqual('token');
});
