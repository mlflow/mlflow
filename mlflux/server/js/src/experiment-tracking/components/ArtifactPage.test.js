import React from 'react';
import { mountWithIntl, shallowWithIntl, mockAjax } from '../../common/utils/TestUtils';
import ArtifactPage, { ArtifactPageImpl } from './ArtifactPage';
import { ArtifactNode } from '../utils/ArtifactUtils';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { ErrorWrapper, pending } from '../../common/utils/ActionUtils';
import { SEARCH_MODEL_VERSIONS } from '../../model-registry/actions';
import {
  ModelVersionStatus,
  Stages,
  MODEL_VERSION_STATUS_POLL_INTERVAL as POLL_INTERVAL,
} from '../../model-registry/constants';
import Utils from '../../common/utils/Utils';
import { mockModelVersionDetailed } from '../../model-registry/test-utils';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { ErrorCodes } from '../../common/constants';
import { ArtifactView } from './ArtifactView';

describe('ArtifactPage', () => {
  let wrapper;
  let minimalProps;
  let minimalStore;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    mockAjax();
    const node = getTestArtifactNode();
    minimalProps = {
      runUuid: 'fakeUuid',
      artifactNode: node,
      artifactRootUri: 'test_root',
      listArtifactsApi: jest.fn(() => Promise.resolve({})),
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

  test('should render with minimal props without exploding', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ArtifactPage {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.length).toBe(1);
  });

  test('should render spinner while ListArtifacts API request is unresolved', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ArtifactPage {...minimalProps} />
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
    const props = { ...minimalProps, apis: {}, searchModelVersionsApi: jest.fn() };
    wrapper = shallowWithIntl(<ArtifactPageImpl {...props} />).dive();
    const responseErrorWrapper = new ErrorWrapper({
      responseText: `{'error_code': '${ErrorCodes.PERMISSION_DENIED}', 'message': 'request failed'}`,
    });
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

    const props = { ...minimalProps, store: minimalStore };
    wrapper = shallowWithIntl(<ArtifactPage {...props} />)
      .dive()
      .dive();
    wrapper.instance().handleActiveNodeChange(true);
    jest.runTimersToTime(POLL_INTERVAL * 3);
    const expectedActions = minimalStore.getActions().filter((action) => {
      return action.type === pending(SEARCH_MODEL_VERSIONS);
    });
    expect(expectedActions).toHaveLength(3);
  });

  test('should not poll for model versions if registry is disabled', () => {
    jest.useFakeTimers();
    const enabledSpy = jest.spyOn(Utils, 'isModelRegistryEnabled').mockImplementation(() => false);
    expect(Utils.isModelRegistryEnabled()).toEqual(false);

    const props = { ...minimalProps, store: minimalStore };
    wrapper = shallowWithIntl(<ArtifactPage {...props} />)
      .dive()
      .dive();
    wrapper.instance().handleActiveNodeChange(true);
    jest.runTimersToTime(POLL_INTERVAL * 3);
    const expectedActions = minimalStore.getActions().filter((action) => {
      return action.type === pending(SEARCH_MODEL_VERSIONS);
    });
    expect(expectedActions).toHaveLength(0);

    enabledSpy.mockRestore();
  });

  test('should not poll for model versions if active node is not directory', () => {
    jest.useFakeTimers();

    const props = { ...minimalProps, store: minimalStore };
    wrapper = shallowWithIntl(<ArtifactPage {...props} />)
      .dive()
      .dive();
    expect(wrapper.instance().state.activeNodeIsDirectory).toEqual(false);

    jest.runTimersToTime(POLL_INTERVAL * 3);
    const expectedActions = minimalStore.getActions().filter((action) => {
      return action.type === pending(SEARCH_MODEL_VERSIONS);
    });
    expect(expectedActions).toHaveLength(0);
  });
});
