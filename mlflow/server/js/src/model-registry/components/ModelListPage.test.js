import React from 'react';
import { mount } from 'enzyme';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { mockModelVersionDetailed, mockRegisteredModelDetailed, Stages,  stageTagComponents, modelStageNames } from '../test-utils';
import { ModelVersionStatus, REGISTERED_MODELS_SEARCH_TIMESTAMP_FIELD } from '../constants';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { ModelListPage, ModelListPageImpl } from './ModelListPage';

describe('ModelListPage', () => {
  let wrapper;
  let minimalProps;
  let minimalStore;
  let instance;
  let pushSpy;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    // TODO: remove global fetch mock by explicitly mocking all the service API calls
    global.fetch = jest.fn(() =>
      Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve('') }),
    );
    const location = {};

    pushSpy = jest.fn();
    const history = {
      location: {
        pathName: '/models',
        search: '',
      },
      push: pushSpy,
    };
    minimalProps = {
      models: [],
      searchRegisteredModelsApi: jest.fn(() => Promise.resolve({})),
      listEndpointsApi: jest.fn(() => Promise.resolve({})),
      listEndpointsV2Api: jest.fn(() => Promise.resolve({})),
      listModelStagesApi: jest.fn(() => Promise.resolve()),
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
        listModelStages: {
          'stageTagComponents': stageTagComponents(),
          'modelStageNames': modelStageNames
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

  test('updateUrlWithSearchFilter correctly pushes url with params', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelListPage {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    instance = wrapper.find(ModelListPageImpl).instance();
    instance.updateUrlWithSearchFilter(
      'name',
      'tag',
      REGISTERED_MODELS_SEARCH_TIMESTAMP_FIELD,
      false,
      2,
    );

    const expectedUrl = `/models?nameSearchInput=name&tagSearchInput=tag&orderByKey=timestamp&orderByAsc=false&page=2`;
    expect(pushSpy).toHaveBeenCalledWith(expectedUrl);
  });
});
