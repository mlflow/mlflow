import React from 'react';
import { mount } from 'enzyme';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { mockModelVersionDetailed, mockRegisteredModelDetailed } from '../test-utils';
import { ModelVersionStatus, Stages } from '../constants';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import ModelListPage from './ModelListPage';
import { mockAjax } from '../../common/utils/TestUtils';

describe('ModelListPage', () => {
  let wrapper;
  let minimalProps;
  let minimalStore;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    mockAjax();
    const location = {};

    const history = {
      location: {
        pathName: '/models',
        search: '',
      },
      push: jest.fn(),
    };
    minimalProps = {
      models: [],
      searchRegisteredModelsApi: jest.fn(() => Promise.resolve({})),
      history,
      location,
    };
    const name = 'Model A';
    const versions = [
      mockModelVersionDetailed('Model A', 1, Stages.PRODUCTION, ModelVersionStatus.READY),
    ];
    minimalStore = mockStore({
      entities: {
        modelByName: {
          [name]: mockRegisteredModelDetailed(name, versions),
        },
      },
      apis: {},
    });
  });

  test('should render with minimal props and store without exploding', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelListPage {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find(ModelListPage).length).toBe(1);
  });
});
