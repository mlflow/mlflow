import React from 'react';
import { mountWithIntl, shallowWithIntl } from '../../common/utils/TestUtils';
import { ArtifactPageImpl, ConnectedArtifactPage } from './ArtifactPage';
import { ArtifactNode } from '../utils/ArtifactUtils';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { pending } from '../../common/utils/ActionUtils';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import { SEARCH_MODEL_VERSIONS } from '../../model-registry/actions';
import {
  ModelVersionStatus,
  MODEL_VERSION_STATUS_POLL_INTERVAL as POLL_INTERVAL,
} from '../../model-registry/constants';
import Utils from '../../common/utils/Utils';
import { mockModelVersionDetailed, Stages } from '../../model-registry/test-utils';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { ErrorCodes } from '../../common/constants';
import { ArtifactView } from './ArtifactView';
import { RunTag } from '../sdk/MlflowMessages';

describe('ArtifactPage', () => {
  let wrapper;
  let minimalProps;
  let minimalStore;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    // TODO: remove global fetch mock by explicitly mocking all the service API calls
    global.fetch = jest.fn(() =>
      Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve('') }),
    );
    const node = getTestArtifactNode();
    minimalProps = {
      runUuid: 'fakeUuid',
      runTags: {},
      artifactNode: node,
      artifactRootUri: 'test_root',
      listArtifactsApi: jest.fn(() => Promise.resolve({})),
      match: {
        params: {},
      },
    };

    minimalStore = mockStore({
      apis: {},
      entities: {
        artifactsByRunUuid: {
          fakeUuid: node,
        },
        artifactRootUriByRunUuid: {
          fakeUuid: '8',
        },
        modelVersionsByModel: {
          'Model A': {
            '1': mockModelVersionDetailed(
              'Model A',
              1,
              Stages.PRODUCTION,
              ModelVersionStatus.READY,
            ),
          },
        },
      },
    });
  });

  const getTestArtifactNode = () => {
    const rootNode = new ArtifactNode(true, undefined);
    rootNode.isLoaded = true;
    const dir1 = new ArtifactNode(false, { path: 'dir1', is_dir: true });
    const file1 = new ArtifactNode(false, { path: 'file1', is_dir: false, file_size: '159' });
    rootNode.children = { dir1, file1 };
    return rootNode;
  };

  const getArtifactPageInstance = () => {
    const mountedComponent = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ConnectedArtifactPage {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );

    return mountedComponent.find(ArtifactPageImpl).instance();
  };

  test('should render with minimal props without exploding', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ConnectedArtifactPage {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.length).toBe(1);
  });

  test('should render spinner while ListArtifacts API request is unresolved', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ConnectedArtifactPage {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find('.Spinner').length).toBe(1);
  });

  test('should make correct number of API requests if artifact path specified in url', (done) => {
    const mock = jest.fn();
    const props = {
      ...minimalProps,
      apis: {},
      listArtifactsApi: mock,
      initialSelectedArtifactPath: 'some/test/directory/file',
      searchModelVersionsApi: jest.fn(),
    };
    wrapper = shallowWithIntl(<ArtifactPageImpl {...props} />).dive();
    setImmediate(() => {
      expect(mock.mock.calls.length).toBe(5);
      done();
    });
  });

  test('ArtifactPage renders error message when listArtifacts request fails', () => {
    jest.spyOn(console, 'error').mockImplementation(() => {});
    const props = { ...minimalProps, apis: {}, searchModelVersionsApi: jest.fn() };
    wrapper = shallowWithIntl(<ArtifactPageImpl {...props} />).dive();
    const responseErrorWrapper = new ErrorWrapper(
      `{'error_code': '${ErrorCodes.PERMISSION_DENIED}', 'message': 'request failed'}`,
      403,
    );
    const artifactPageInstance = wrapper.instance();
    const listArtifactsErrorRequest = {
      id: artifactPageInstance.listArtifactsRequestId,
      active: false,
      error: responseErrorWrapper,
    };
    const artifactViewInstance = shallowWithIntl(
      artifactPageInstance.renderArtifactView(false, true, [listArtifactsErrorRequest]),
    );
    expect(artifactViewInstance.find('.mlflow-artifact-error').length).toBe(1);
    jest.clearAllMocks();
  });

  test('ArtifactPage renders ArtifactView when listArtifacts request succeeds', () => {
    const props = { ...minimalProps, apis: {}, searchModelVersionsApi: jest.fn() };
    wrapper = shallowWithIntl(<ArtifactPageImpl {...props} />).dive();
    const artifactPageInstance = wrapper.instance();
    const listArtifactsSuccessRequest = {
      id: artifactPageInstance.listArtifactsRequestId,
      active: true,
      payload: {},
    };
    const artifactViewInstance = shallowWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          {artifactPageInstance.renderArtifactView(false, false, [listArtifactsSuccessRequest])}
        </BrowserRouter>
      </Provider>,
    );
    expect(artifactViewInstance.find(ArtifactView).length).toBe(1);
  });

  test('should poll for model versions if registry is enabled and active node is directory', () => {
    jest.useFakeTimers();
    expect(Utils.isModelRegistryEnabled()).toEqual(true);

    getArtifactPageInstance().handleActiveNodeChange(true);
    jest.advanceTimersByTime(POLL_INTERVAL * 3);
    const expectedActions = minimalStore.getActions().filter((action) => {
      return action.type === pending(SEARCH_MODEL_VERSIONS);
    });
    expect(expectedActions).toHaveLength(3);
  });

  test('should not poll for model versions if registry is disabled', () => {
    jest.useFakeTimers();
    const enabledSpy = jest.spyOn(Utils, 'isModelRegistryEnabled').mockImplementation(() => false);
    expect(Utils.isModelRegistryEnabled()).toEqual(false);
    getArtifactPageInstance().handleActiveNodeChange(true);
    jest.advanceTimersByTime(POLL_INTERVAL * 3);
    const expectedActions = minimalStore.getActions().filter((action) => {
      return action.type === pending(SEARCH_MODEL_VERSIONS);
    });
    expect(expectedActions).toHaveLength(0);

    enabledSpy.mockRestore();
  });

  test('should not poll for model versions if active node is not directory', () => {
    jest.useFakeTimers();
    expect(getArtifactPageInstance().state.activeNodeIsDirectory).toEqual(false);

    jest.advanceTimersByTime(POLL_INTERVAL * 3);
    const expectedActions = minimalStore.getActions().filter((action) => {
      return action.type === pending(SEARCH_MODEL_VERSIONS);
    });
    expect(expectedActions).toHaveLength(0);
  });

  describe('autoselect logged model', () => {
    const generateLoggedModel = ({ time = '2021-05-01', path = 'someRunPath' } = {}) => ({
      run_id: `run-uuid`,
      artifact_path: path,
      utc_time_created: time,
      flavors: { keras: {}, python_function: {} },
    });

    const getLoggedModelRunTag = (models) =>
      models.length > 0
        ? {
            'mlflow.log-model.history': RunTag.fromJs({
              key: 'mlflow.log-model.history',
              value: JSON.stringify(models),
            }),
          }
        : {};

    const getInstance = ({ initialPath = '', models = [] } = {}) => {
      const props = {
        ...minimalProps,
        runTags: getLoggedModelRunTag(models),
        ...(initialPath && {
          match: {
            params: {
              initialSelectedArtifactPath: initialPath,
            },
          },
        }),
      };

      const artifactPageWrapper = mountWithIntl(
        <Provider store={minimalStore}>
          <BrowserRouter>
            <ConnectedArtifactPage {...props} />
          </BrowserRouter>
        </Provider>,
      );

      return artifactPageWrapper.find(ArtifactPageImpl).instance();
    };

    it('selects path from route when present', () => {
      const instance = getInstance({
        initialPath: 'passedInPath',
      });
      expect(instance.props['initialSelectedArtifactPath']).toBe('passedInPath');
    });

    it('autoselects from runtag if no path is present', () => {
      const instance = getInstance({
        models: [generateLoggedModel()],
      });
      expect(instance.props['initialSelectedArtifactPath']).toBe('someRunPath');
    });

    it('autoselects the most recent path', () => {
      const instance = getInstance({
        models: [
          generateLoggedModel({
            path: 'reallyOldRunPath',
            time: '1776-07-04',
          }),
          generateLoggedModel({
            path: 'moreRecentRunPath',
            time: '2021-07-04',
          }),
        ],
      });
      expect(instance.props['initialSelectedArtifactPath']).toBe('moreRecentRunPath');
    });
  });
});
