import React from 'react';
import { mount } from 'enzyme';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import thunk from 'redux-thunk';
import RegisterModelButton from './RegisterModelButton';
import { mockAjax } from '../../common/utils/TestUtils';

describe('RegisterModelButton', () => {
  let wrapper;
  let minimalProps;
  let minimalStore;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    mockAjax();
    minimalProps = {
      disabled: false,
      runUuid: 'runUuid',
    };
    minimalStore = mockStore({
      entities: {
        modelByName: {},
      },
    });
  });

  test('should render with minimal props and store without exploding', () => {
    wrapper = mount(<RegisterModelButton {...minimalProps} store={minimalStore} />);
    expect(wrapper.find('button').length).toBe(1);
  });
});
