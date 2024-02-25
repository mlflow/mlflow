/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { mountWithIntl } from '../../common/utils/TestUtils.enzyme';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { Provider } from 'react-redux';
import { BrowserRouter } from '../../common/utils/RoutingUtils';
import { RunPage, RunPageImpl } from './RunPage';
import { ArtifactNode } from '../utils/ArtifactUtils';
import { mockModelVersionDetailed } from '../../model-registry/test-utils';
import { ModelVersionStatus, Stages } from '../../model-registry/constants';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import { ErrorCodes } from '../../common/constants';
import { RunNotFoundView } from './RunNotFoundView';

// mock this as feature-flags are hard-coded
jest.mock('../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual('../../common/utils/FeatureUtils'),
  shouldEnableDeepLearningUI: jest.fn(() => false),
}));

describe('RunPage', () => {
  let wrapper;
  let minimalProps: any;
  let minimalStore: any;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    // TODO: remove global fetch mock by explicitly mocking all the service API calls
    // @ts-expect-error TS(2322): Type 'Mock<Promise<{ ok: true; status: number; tex... Remove this comment to see the full error message
    global.fetch = jest.fn(() => Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve('') }));
    const modelVersion = mockModelVersionDetailed('Model A', 1, Stages.PRODUCTION, ModelVersionStatus.READY);
    const versions = [modelVersion];
    minimalProps = {
      match: {
        params: {
          runUuid: 'uuid-1234-5678-9012',
          experimentId: '12345',
        },
      },
      history: {
        push: jest.fn(),
      },
      modelVersions: versions,
      getRunApi: jest.fn(() => Promise.resolve({})),
      getExperimentApi: jest.fn(() => Promise.resolve({})),
      searchModelVersionsApi: jest.fn(() => Promise.resolve({})),
      setTagApi: jest.fn(() => Promise.resolve({})),
    };
    minimalStore = mockStore({
      entities: {
        runInfosByUuid: {
          'uuid-1234-5678-9012': {
            run_uuid: 'uuid-1234-5678-9012',
            experiment_id: '12345',
            user_id: 'me@me.com',
            status: 'RUNNING',
            start_time: 12345678990,
            end_time: 12345678999,
            artifact_uri: 'dbfs:/databricks/abc/uuid-1234-5678-9012',
            lifecycle_stage: 'active',
          },
        },
        // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
        artifactsByRunUuid: { 'uuid-1234-5678-9012': new ArtifactNode(true) },
        experimentsById: {
          12345: {
            experiment_id: '12345',
            name: 'my experiment',
            artifact_location: 'dbfs:/databricks/abc',
            lifecycle_stage: 'active',
            last_update_time: 12345678999,
            creation_time: 12345678900,
            tags: [],
          },
        },
        modelVersionsByModel: {
          'Model A': {
            1: modelVersion,
          },
        },
        tagsByRunUuid: { 'uuid-1234-5678-9012': {} },
      },
      apis: {},
    });
  });

  test('should render with minimal props and store without exploding', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <RunPage {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find(RunPage).length).toBe(1);
  });

  test('should display RunNotFoundView on RESOURCE_DOES_NOT_EXIST error', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <RunPage {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    ).find(RunPage);
    const runPageInstance = wrapper.find(RunPageImpl).instance();
    const responseErrorWrapper = new ErrorWrapper(
      `{"error_code": "${ErrorCodes.RESOURCE_DOES_NOT_EXIST}", "message": "Not found."}`,
      404,
    );
    const getRunErrorRequest = {
      id: runPageInstance.getRunRequestId,
      active: false,
      error: responseErrorWrapper,
    };
    expect(runPageInstance.renderRunView(false, true, [getRunErrorRequest]).type).toBe(RunNotFoundView);
  });
});
