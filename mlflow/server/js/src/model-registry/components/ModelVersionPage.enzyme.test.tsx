/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { jest, describe, beforeEach, afterEach, test, expect } from '@jest/globals';
import { shallow } from 'enzyme';
import { waitFor } from '@testing-library/react';
import React from 'react';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { mockModelVersionDetailed, mockRegisteredModelDetailed } from '../test-utils';
import { ModelVersionStatus, Stages } from '../constants';
import { Provider } from 'react-redux';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { ModelVersionPage, ModelVersionPageImpl } from './ModelVersionPage';
import { ErrorView } from '../../common/components/ErrorView';
import { Spinner } from '../../common/components/Spinner';
import Utils from '../../common/utils/Utils';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import { ErrorCodes } from '../../common/constants';
import { ModelRegistryRoutes } from '../routes';
import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';
import { getUUID } from '../../common/utils/ActionUtils';
import { fetchModelVersionArtifact, getModelVersionApi } from '../actions';
import type { ModelVersionArtifactMetadata } from '../actions';
import { Services } from '../services';
import { MlflowService } from '../../experiment-tracking/sdk/MlflowService';
import * as PresignedArtifactUtils from '../../experiment-tracking/utils/PresignedArtifactUtils';
import * as ServerInfo from '../../experiment-tracking/hooks/useServerInfo';

jest.mock('../../common/utils/ActionUtils', () => ({
  getUUID: jest.fn(),
}));

jest.mock('../actions', () => ({
  ...jest.requireActual<typeof import('../actions')>('../actions'),
  getModelVersionApi: jest.fn(),
}));

describe('ModelVersionPage', () => {
  let wrapper: ReturnType<typeof mountWithIntl> | ReturnType<typeof shallow> | undefined;
  let instance;
  let minimalProps: any;
  let minimalStoreState: any;
  let minimalStore: any;
  const mockStore = configureStore([thunk, promiseMiddleware()]);
  const navigate = jest.fn();

  const mountComponent = (props = minimalProps, store = minimalStore) => {
    return mountWithIntl(
      <Provider store={store}>
        <MemoryRouter>
          <ModelVersionPage {...props} />
        </MemoryRouter>
      </Provider>,
    );
  };

  beforeEach(() => {
    // Simple mock of getUUID
    let counter = 0;
    (getUUID as any).mockImplementation(() => `${counter++}`);
    // TODO: remove global fetch mock by explicitly mocking all the service API calls
    jest
      .spyOn(global, 'fetch')
      // @ts-expect-error TS(2322): Type 'Mock<Promise<{ ok: true; status: number; tex... Remove this comment to see the full error message
      .mockImplementation(() => Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve('') }));
    jest
      .mocked(getModelVersionApi)
      .mockImplementation(jest.requireActual<typeof import('../actions')>('../actions').getModelVersionApi);
    minimalProps = {
      params: {
        modelName: encodeURIComponent('Model A'),
        version: '1',
      },
      navigate,
    };
    const versions = [mockModelVersionDetailed('Model A', 1, Stages.PRODUCTION, ModelVersionStatus.READY)];
    minimalStoreState = {
      entities: {
        runInfosByUuid: {},
        modelByName: {
          // @ts-expect-error TS(2345): Argument of type '{ name: any; creation_timestamp:... Remove this comment to see the full error message
          'Model A': mockRegisteredModelDetailed('Model A', versions),
        },
        modelVersionsByModel: {
          'Model A': {
            1: mockModelVersionDetailed('Model A', '1', Stages.PRODUCTION, ModelVersionStatus.READY),
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
  afterEach(() => {
    wrapper?.unmount();
    wrapper = undefined;
    jest.restoreAllMocks();
  });

  test('shares the page metadata requests with the artifact read on older servers', async () => {
    jest.spyOn(ServerInfo, 'getArtifactsPresignedOnlySync').mockReturnValue(false);
    jest.spyOn(ServerInfo, 'getModelArtifactPresignedDownloadsSupportedSync').mockReturnValue(false);
    const versionLookup = jest.spyOn(Services, 'getModelVersion');
    const runLookup = jest.spyOn(MlflowService, 'getRun');
    const read = jest.spyOn(PresignedArtifactUtils, 'fetchArtifactPreferPresigned').mockResolvedValue('contents');
    const getVersion = jest.fn(async () => ({
      value: { model_version: { run_id: 'run-id', source: 'runs:/run-id/model' } },
    }));
    const getRun = jest.fn(async () => ({
      value: { run: { info: { artifactUri: 'mlflow-artifacts:/12/run-id/artifacts' } } },
    }));
    const parse = jest.fn();
    const props: any = {
      modelName: 'Model A',
      version: '1',
      apis: {},
      getModelVersionApi: getVersion,
      getRunApi: getRun,
      getRegisteredModelApi: jest.fn(),
      parseMlModelFile: parse,
      getModelVersionArtifactApi: (
        name: string,
        version: string,
        _id: string,
        metadata: ModelVersionArtifactMetadata,
      ) => fetchModelVersionArtifact(name, version, metadata).then((value) => ({ value })),
    };
    wrapper = shallow(<ModelVersionPageImpl {...props} />);
    await waitFor(() => expect(parse).toHaveBeenCalledTimes(1));
    expect(getVersion).toHaveBeenCalledTimes(1);
    expect(getRun).toHaveBeenCalledTimes(1);
    expect(versionLookup).not.toHaveBeenCalled();
    expect(runLookup).not.toHaveBeenCalled();
    expect(read).toHaveBeenCalledTimes(1);
  });

  test.each([
    { marker: 'artifacts_presigned_only', message: 'Mode changed', expected: 'Reload this page', status: 403 },
    {
      marker: 'external_artifact_service',
      message: 'Configure a reverse proxy',
      expected: 'Configure a reverse proxy',
      status: 403,
    },
    {
      marker: 'invalid_artifact_root',
      message: 'Artifact root is invalid',
      expected: 'Artifact root is invalid',
      status: 400,
    },
    { marker: undefined, message: 'Optional artifact not found', expected: undefined, status: 404 },
    { marker: undefined, message: 'Bad request', expected: undefined, status: 400 },
  ])(
    'shows actionable artifact configuration failures while optional files remain quiet: $marker',
    async ({ marker, message, expected, status }) => {
      const notification = jest.spyOn(Utils, 'displayGlobalErrorNotification').mockImplementation(() => undefined);
      const error = new ErrorWrapper(
        {
          error_code:
            status === 403
              ? 'PERMISSION_DENIED'
              : status === 404
                ? 'RESOURCE_DOES_NOT_EXIST'
                : 'INVALID_PARAMETER_VALUE',
          message,
          ...(marker && { [marker]: true }),
        },
        status,
      );
      const props: any = {
        modelName: 'Model A',
        version: '1',
        apis: {},
        getModelVersionApi: jest.fn(async () => ({ value: { model_version: {} } })),
        getRegisteredModelApi: jest.fn(),
        getModelVersionArtifactApi: jest.fn(async () => {
          throw error;
        }),
      };
      wrapper = shallow(<ModelVersionPageImpl {...props} />);
      await waitFor(() => expect(wrapper?.state('criticalInitialRequestIds')).toHaveLength(1));
      if (expected) expect(notification).toHaveBeenCalledWith(expect.stringContaining(expected), 3);
      else expect(notification).not.toHaveBeenCalled();
    },
  );
  test('should render with minimal props and store without exploding', () => {
    wrapper = mountComponent();
    expect(wrapper.find(ModelVersionPage).length).toBe(1);
    expect(wrapper.find(Spinner).length).toBe(1);
  });
  test('should fetch new data when props are updated after mount', () => {
    const endpoint = 'ajax-api/2.0/mlflow/model-versions/get';
    const TestComponent = ({ params = minimalProps.params }) => (
      <Provider store={minimalStore}>
        <MemoryRouter>
          <ModelVersionPage {...minimalProps} params={params} />
        </MemoryRouter>
      </Provider>
    );
    // Initial mount
    wrapper = mountWithIntl(<TestComponent />);
    // Assert first (original) call for model version
    expect(global.fetch).toHaveBeenCalledWith(endpoint + '?name=Model+A&version=1', expect.anything());
    // Update the mocked params object with new params
    wrapper.setProps({
      params: {
        ...minimalProps.params,
        version: '5',
      },
    });
    // Assert second call for model version
    expect(global.fetch).toHaveBeenCalledWith(endpoint + '?name=Model+A&version=5', expect.anything());
  });
  test('should redirect to model page when model version is deleted', async () => {
    wrapper = mountComponent();
    instance = wrapper.find(ModelVersionPageImpl).instance();
    const mockError = {
      getErrorCode() {
        return 'RESOURCE_DOES_NOT_EXIST';
      },
    };
    jest.spyOn(Utils, 'isBrowserTabVisible').mockImplementation(() => true);
    jest.spyOn(instance, 'loadData').mockImplementation(() => Promise.reject(mockError));
    expect(instance.props.modelName).toEqual('Model A');
    await instance.pollData();
    expect(navigate).toHaveBeenCalledWith(ModelRegistryRoutes.getModelPageRoute('Model A'));
  });
  test('should show ErrorView when resource is not found', () => {
    (getUUID as any).mockImplementation(() => 'resource_not_found_error');
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
    wrapper = mountComponent(minimalProps, myStore);
    expect(wrapper.find(ErrorView).length).toBe(1);
    expect(wrapper.find(ErrorView).prop('statusCode')).toBe(404);
    expect(wrapper.find(ErrorView).prop('subMessage')).toBe('Model Model A v1 does not exist');
  });
  test('should not crash runtime when API call rejects', () => {
    const httpError = new ErrorWrapper(`{"error_code": "${ErrorCodes.RESOURCE_DOES_NOT_EXIST}"}`, 404);
    jest.mocked(getModelVersionApi).mockImplementation(() => {
      return {
        type: 'GET_MODEL_VERSION',
        payload: Promise.reject(httpError),
        meta: { id: 'abc', modelName: 'abc', version: '1' },
      };
    });
    (getUUID as any).mockImplementation(() => 'resource_not_found_error');
    const myStore = mockStore({
      ...minimalStoreState,
      apis: {
        resource_not_found_error: {
          id: 'resource_not_found_error',
          active: false,
          error: httpError,
        },
      },
    });

    // This test would fail if any unhandled promise rejection occurs
    expect(() => mountComponent(minimalProps, myStore)).not.toThrow();
  });
  test('should show ErrorView when resource conflict error is thrown', () => {
    const testMessage = 'Detected model version conflict';
    (getUUID as any).mockImplementation(() => 'resource_conflict_id');
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
    wrapper = mountComponent(minimalProps, myStore);
    expect(wrapper.find(ErrorView).length).toBe(1);
    expect(wrapper.find(ErrorView).prop('statusCode')).toBe(409);
    expect(wrapper.find(ErrorView).prop('subMessage')).toBe(testMessage);
  });
});
