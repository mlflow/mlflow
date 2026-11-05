/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { jest, describe, beforeEach, afterEach, test, expect } from '@jest/globals';
import { waitFor } from '@testing-library/react';
import React from 'react';
import { mount, shallow } from 'enzyme';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { CompareModelVersionsPageImpl, CompareModelVersionsPage } from './CompareModelVersionsPage';
import { fetchModelVersionArtifact } from '../actions';
import type { ModelVersionArtifactMetadata } from '../actions';
import { Services } from '../services';
import { MlflowService } from '../../experiment-tracking/sdk/MlflowService';
import * as PresignedArtifactUtils from '../../experiment-tracking/utils/PresignedArtifactUtils';
import * as ServerInfo from '../../experiment-tracking/hooks/useServerInfo';
import Utils from '../../common/utils/Utils';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';

describe('CompareModelVersionPage', () => {
  let wrapper: ReturnType<typeof mount> | ReturnType<typeof shallow> | undefined;
  let minimalStore: any;
  let minimalProps: any;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    // TODO: remove global fetch mock by explicitly mocking all the service API calls
    jest
      .spyOn(global, 'fetch')
      // @ts-expect-error TS(2322): Type 'Mock<Promise<{ ok: true; status: number; tex... Remove this comment to see the full error message
      .mockImplementation(() => Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve('') }));
    const modelName = 'normal-model-name';
    minimalProps = {
      location: {
        search:
          '?name=' +
          encodeURI(JSON.stringify(modelName)) +
          '&runs=' +
          JSON.stringify({
            1: '123',
            2: '234',
          }),
      },
      versionsToRuns: {
        1: '123',
        2: '234',
      },
      getRegisteredModelApi: jest.fn(),
      getModelVersionApi: jest.fn(),
      parseMlModelFile: jest.fn(),
    };
    minimalStore = mockStore({
      apis: {},
    });
  });

  afterEach(() => {
    wrapper?.unmount();
    wrapper = undefined;
    jest.restoreAllMocks();
  });

  test('shares each version lookup and a common run lookup with artifact reads on older servers', async () => {
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
      versionsToRuns: { 1: 'run-id', 2: 'run-id' },
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
    wrapper = shallow(<CompareModelVersionsPageImpl {...props} />);
    await waitFor(() => expect(parse).toHaveBeenCalledTimes(2));
    expect(getVersion).toHaveBeenCalledTimes(2);
    expect(getRun).toHaveBeenCalledTimes(1);
    expect(versionLookup).not.toHaveBeenCalled();
    expect(runLookup).not.toHaveBeenCalled();
    expect(read).toHaveBeenCalledTimes(2);
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
    'notifies actionable artifact failures without blocking comparison: $marker',
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
        versionsToRuns: { 1: null },
        getModelVersionApi: jest.fn(async () => ({ value: { model_version: {} } })),
        getRegisteredModelApi: jest.fn(),
        getModelVersionArtifactApi: jest.fn(async () => {
          throw error;
        }),
      };
      wrapper = shallow(<CompareModelVersionsPageImpl {...props} />);
      await waitFor(() => expect(wrapper?.state('requestIds')).toHaveLength(2));
      if (expected) expect(notification).toHaveBeenCalledWith(expect.stringContaining(expected), 3);
      else expect(notification).not.toHaveBeenCalled();
    },
  );

  // eslint-disable-next-line jest/expect-expect -- TODO(FEINF-1337)
  test('should render with minimal props and store without exploding', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <MemoryRouter>
          <CompareModelVersionsPage {...minimalProps} />
        </MemoryRouter>
      </Provider>,
    );
  });

  test('should render with name with model name with special characters', () => {
    const props: any = {
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
        <MemoryRouter>
          <CompareModelVersionsPage {...props} />
        </MemoryRouter>
      </Provider>,
    );
    expect(wrapper.find(CompareModelVersionsPageImpl).length).toBe(1);
  });

  test('should remove getRunRequest and getMlModelFileRequest api ids from state on 404', async () => {
    const mockError = {
      getErrorCode() {
        return 'RESOURCE_DOES_NOT_EXIST';
      },
    };
    const getRunApi = jest.fn().mockReturnValue(Promise.reject(mockError));
    const getModelVersionArtifactApi = jest.fn().mockReturnValue(Promise.reject(mockError));
    const myProps = {
      getRunApi: getRunApi,
      getModelVersionArtifactApi: getModelVersionArtifactApi,
      ...minimalProps,
    };
    const wrapper2 = shallow(<CompareModelVersionsPageImpl {...myProps} />);
    expect(wrapper2.state('requestIds').length).toBe(4);
    await expect(getRunApi).toHaveBeenCalled();
    await expect(getModelVersionArtifactApi).toHaveBeenCalled();
    expect(wrapper2.state('requestIds').length).toBe(2);
  });
});
