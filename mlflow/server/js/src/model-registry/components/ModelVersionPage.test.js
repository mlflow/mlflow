import React from 'react';
import { mount } from 'enzyme';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { mockModelVersionDetailed, mockRegisteredModelDetailed } from '../test-utils';
import { ModelVersionStatus, Stages } from '../constants';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { ModelVersionPage, ModelVersionPageImpl } from './ModelVersionPage';
import { ErrorView } from '../../common/components/ErrorView';
import { Spinner } from '../../common/components/Spinner';
import Utils from '../../common/utils/Utils';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import { ErrorCodes } from '../../common/constants';
import { getModelPageRoute } from '../routes';
import { mountWithIntl } from '../../common/utils/TestUtils';
import { getUUID } from '../../common/utils/ActionUtils';

jest.mock('../../common/utils/ActionUtils', () => ({
  getUUID: jest.fn(),
}));

describe('ModelVersionPage', () => {
  let wrapper;
  let instance;
  let minimalProps;
  let minimalStoreState;
  let minimalStore;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    // Simple mock of getUUID
    let counter = 0;
    getUUID.mockImplementation(() => `${counter++}`);

    // TODO: remove global fetch mock by explicitly mocking all the service API calls
    global.fetch = jest.fn(() =>
      Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve('') }),
    );

    minimalProps = {
      match: {
        params: {
          modelName: encodeURIComponent('Model A'),
          version: '1',
        },
      },
      history: {
        push: jest.fn(),
      },
    };
    const versions = [
      mockModelVersionDetailed('Model A', 1, Stages.PRODUCTION, ModelVersionStatus.READY),
    ];
    minimalStoreState = {
      entities: {
        runInfosByUuid: {},
        modelByName: {
          'Model A': mockRegisteredModelDetailed('Model A', versions),
        },
        modelVersionsByModel: {
          'Model A': {
            1: mockModelVersionDetailed(
              'Model A',
              '1',
              Stages.PRODUCTION,
              ModelVersionStatus.READY,
            ),
          },
        },
        activitiesByModelVersion: {},
        transitionRequestsByModelVersion: {},
        mlModelArtifactByModelVersion: {},
      },
    };
    minimalStore = mockStore({
      ...minimalStoreState,
      apis: {},
    });
  });

  test('should render with minimal props and store without exploding', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelVersionPage {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find(ModelVersionPage).length).toBe(1);
    expect(wrapper.find(Spinner).length).toBe(1);
  });

  test('should fetch new data when props are updated after mount', () => {
    // eslint-disable-next-line no-unused-vars
    const endpoint = 'ajax-api/2.0/mlflow/model-versions/get';

    const TestComponent = (
      { match = minimalProps.match }, // eslint-disable-line react/prop-types
    ) => (
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelVersionPage {...minimalProps} match={match} />
        </BrowserRouter>
      </Provider>
    );

    // Initial mount
    wrapper = mount(<TestComponent />);

    // Assert first (original) call for model version
    expect(global.fetch).toBeCalledWith(endpoint + '?name=Model+A&version=1', expect.anything());

    // Update the mocked match object with new params
    wrapper.setProps({
      match: {
        ...minimalProps.match,
        params: {
          ...minimalProps.match.params,
          version: '5',
        },
      },
    });

    // Assert second call for model version
    expect(global.fetch).toBeCalledWith(endpoint + '?name=Model+A&version=5', expect.anything());
  });

  test('should redirect to model page when model version is deleted', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ModelVersionPage {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    instance = wrapper.find(ModelVersionPageImpl).instance();
    const mockError = {
      getErrorCode() {
        return 'RESOURCE_DOES_NOT_EXIST';
      },
    };
    Utils.isBrowserTabVisible = jest.fn(() => true);
    instance.loadData = jest.fn(() => Promise.reject(mockError));
    expect(instance.props.modelName).toEqual('Model A');
    return instance.pollData().then(() => {
      expect(minimalProps.history.push).toHaveBeenCalledWith(getModelPageRoute('Model A'));
    });
  });

  test('should show ErrorView when resource is not found', () => {
    getUUID.mockImplementation(() => 'resource_not_found_error');
    // Populate store with failed model version get request
    const myStore = mockStore({
      ...minimalStoreState,
      apis: {
        resource_not_found_error: {
          id: 'resource_not_found_error',
          active: false,
          error: new ErrorWrapper(`{"error_code": "${ErrorCodes.RESOURCE_DOES_NOT_EXIST}"}`, 404),
        },
      },
    });

    wrapper = mountWithIntl(
      <Provider store={myStore}>
        <BrowserRouter>
          <ModelVersionPage {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find(ErrorView).length).toBe(1);
    expect(wrapper.find(ErrorView).prop('statusCode')).toBe(404);
    expect(wrapper.find(ErrorView).prop('subMessage')).toBe('Model Model A v1 does not exist');
  });

  test('should show ErrorView when resource conflict error is thrown', () => {
    const testMessage = 'Detected model version conflict';
    getUUID.mockImplementation(() => 'resource_conflict_id');
    // Populate store with failed model version get request
    const myStore = mockStore({
      ...minimalStoreState,
      apis: {
        resource_conflict_id: {
          id: 'resource_conflict_id',
          active: false,
          error: new ErrorWrapper(
            `{"error_code": "${ErrorCodes.RESOURCE_CONFLICT}", "message": "${testMessage}"}`,
            409,
          ),
        },
      },
    });

    wrapper = mountWithIntl(
      <Provider store={myStore}>
        <BrowserRouter>
          <ModelVersionPage {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find(ErrorView).length).toBe(1);
    expect(wrapper.find(ErrorView).prop('statusCode')).toBe(409);
    expect(wrapper.find(ErrorView).prop('subMessage')).toBe(testMessage);
  });
});
