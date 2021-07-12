import React from 'react';
import { mount } from 'enzyme';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import CompareModelVersionsPage from './CompareModelVersionsPage';
import { mockAjax } from '../../common/utils/TestUtils';

describe('CompareModelVersionPage', () => {
  let wrapper;
  let minimalStore;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    mockAjax();
    minimalStore = mockStore({
      apis: {},
    });
  });

  test('should render with minimal props and store without exploding', () => {
    const props = {
      location: {
        search:
          '?name=' +
          encodeURI(JSON.stringify('normal-model-name')) +
          '&runs=' +
          JSON.stringify({
            1: '123',
            2: '234',
          }),
      },
    };
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <CompareModelVersionsPage {...props} />
        </BrowserRouter>
      </Provider>,
    );
  });

  test('should render with name with model name with special characters', () => {
    const props = {
      location: {
        search:
          '?name=' +
          encodeURI(JSON.stringify('funky?!@#$^*()_=name~[]')) +
          '&runs=' +
          JSON.stringify({
            1: '123',
            2: '234',
          }),
      },
    };
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <CompareModelVersionsPage {...props} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find(CompareModelVersionsPage).length).toBe(1);
  });
});
