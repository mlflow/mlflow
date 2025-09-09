/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallowWithIntl, mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';
import { ArtifactPageImpl, ConnectedArtifactPage } from './ArtifactPage';
import { ArtifactNode } from '../utils/ArtifactUtils';
import { Provider } from 'react-redux';
import { BrowserRouter } from '../../common/utils/RoutingUtils';
import { pending } from '../../common/utils/ActionUtils';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
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
import { RunTag } from '../sdk/MlflowMessages';

describe('ArtifactPage', () => {
  let wrapper;
  let minimalProps: any;
  let minimalStore: any;
  const mockStore = configureStore([thunk, promiseMiddleware()]);
  beforeEach(() => {
    // TODO: remove global fetch mock by explicitly mocking all the service API calls
    // @ts-expect-error TS(2322): Type 'Mock<Promise<{ ok: true; status: number; tex... Remove this comment to see the full error message
    global.fetch = jest.fn(() => Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve('') }));
    const node = getTestArtifactNode();
    minimalProps = {
      runUuid: 'fakeUuid',
      runTags: {},
      artifactNode: node,
      artifactRootUri: 'dbfs:/',
      listArtifactsApi: jest.fn(() => Promise.resolve({})),
      params: {},
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
            1: mockModelVersionDetailed('Model A', 1, Stages.PRODUCTION, ModelVersionStatus.READY),
          },
        },
      },
    });
  });
  const getTestArtifactNode = () => {
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
    const rootNode = new ArtifactNode(true, undefined);
    rootNode.isLoaded = true;
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
    const dir1 = new ArtifactNode(false, { path: 'dir1', is_dir: true });
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
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
    expect(wrapper.find('ArtifactViewBrowserSkeleton').length).toBe(1);
  });
  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
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
    expect(artifactViewInstance.find('[data-testid="artifact-view-error"]').length).toBe(1);
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
    const expectedActions = minimalStore.getActions().filter((action: any) => {
      return action.type === pending(SEARCH_MODEL_VERSIONS);
    });
    expect(expectedActions).toHaveLength(3);
  });
  test('should not poll for model versions if active node is not directory', () => {
    jest.useFakeTimers();
    expect(getArtifactPageInstance().state.activeNodeIsDirectory).toEqual(false);
    jest.advanceTimersByTime(POLL_INTERVAL * 3);
    const expectedActions = minimalStore.getActions().filter((action: any) => {
      return action.type === pending(SEARCH_MODEL_VERSIONS);
    });
    expect(expectedActions).toHaveLength(0);
  });
  test('should not report multiple errors', () => {
    jest.useFakeTimers();
    Utils.isModelRegistryEnabled = jest.fn().mockReturnValue(true);
    Utils.logErrorAndNotifyUser = jest.fn();
    expect(Utils.logErrorAndNotifyUser).toHaveBeenCalledTimes(0);
    const props = {
      ...minimalProps,
      apis: {},
      searchModelVersionsApi: jest.fn(() => {
        throw Error('err');
      }),
    };
    // Create our wrapper with the intial props
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ArtifactPageImpl {...props} />
        </BrowserRouter>
      </Provider>,
    );
    wrapper.find(ArtifactPageImpl).setState({ activeNodeIsDirectory: true });
    // Wait multiple poll intervals
    jest.advanceTimersByTime(POLL_INTERVAL * 3);
    // We should have only one error call
    expect(Utils.logErrorAndNotifyUser).toHaveBeenCalledTimes(1);
    // Let's change the run uuid now by changing the props
    // sadly, enzyme provides no convenient method to change
    // the deeply nested component props so we need to
    // improvise: https://github.com/enzymejs/enzyme/issues/1925
    wrapper.setProps({
      children: (
        <BrowserRouter>
          <ArtifactPageImpl {...props} runUuid="anotherFakeUuid" />
        </BrowserRouter>
      ),
    });
    // Wait another multiple poll intervals
    jest.advanceTimersByTime(POLL_INTERVAL * 5);
    // We should have only one more error call
    expect(Utils.logErrorAndNotifyUser).toHaveBeenCalledTimes(2);
    jest.clearAllMocks();
  });
  describe('autoselect logged model', () => {
    const generateLoggedModel = ({ time = '2021-05-01', path = 'someRunPath' } = {}) => ({
      run_id: `run-uuid`,
      artifact_path: path,
      utc_time_created: time,
      flavors: { keras: {}, python_function: {} },
    });
    const getLoggedModelRunTag = (models: any) =>
      models.length > 0
        ? {
            'mlflow.log-model.history': (RunTag as any).fromJs({
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
          params: {
            initialSelectedArtifactPath: initialPath,
          },
        }),
      };

      if (initialPath) {
        props.location = { pathname: `/artifactPath/${initialPath}` };
      }

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
        // @ts-expect-error TS(2322): Type '{ run_id: string; artifact_path: string; utc... Remove this comment to see the full error message
        models: [generateLoggedModel()],
      });
      expect(instance.props['initialSelectedArtifactPath']).toBe('someRunPath');
    });
    it('autoselects the most recent path', () => {
      const instance = getInstance({
        models: [
          // @ts-expect-error TS(2322): Type '{ run_id: string; artifact_path: string; utc... Remove this comment to see the full error message
          generateLoggedModel({
            path: 'reallyOldRunPath',
            time: '1776-07-04',
          }),
          // @ts-expect-error TS(2322): Type '{ run_id: string; artifact_path: string; utc... Remove this comment to see the full error message
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
