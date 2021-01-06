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
import Utils from '../../common/utils/Utils';
import { getModelPageRoute } from '../routes';
import { mockAjax } from '../../common/utils/TestUtils';

describe('ModelVersionPage', () => {
  let wrapper;
  let instance;
  let minimalProps;
  let minimalStore;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    mockAjax();
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
    minimalStore = mockStore({
      entities: {
        runInfosByUuid: {},
        modelByName: {
          'Model A': mockRegisteredModelDetailed('Model A', versions),
        },
        modelVersionsByModel: {
          'Model A': {
            '1': mockModelVersionDetailed(
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
});
