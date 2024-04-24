/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { Provider } from 'react-redux';
import { BrowserRouter } from '../../common/utils/RoutingUtils';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { RunView, RunViewImpl } from './RunView';
import { Experiment, RunInfo, RunTag, Param } from '../sdk/MlflowMessages';
import { ArtifactNode } from '../utils/ArtifactUtils';
import { mockModelVersionDetailed } from '../../model-registry/test-utils';
import { ModelVersionStatus, Stages } from '../../model-registry/constants';
import { mountWithIntl } from 'common/utils/TestUtils.enzyme';

// mock this as feature-flags are hard-coded
jest.mock('../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual('../../common/utils/FeatureUtils'),
  shouldEnableDeepLearningUI: jest.fn(() => false),
}));

describe('RunView', () => {
  let minimalProps: any;
  let minimalStoreRaw: any;
  let minimalStore: any;
  let wrapper;
  const mockStore = configureStore([thunk, promiseMiddleware()]);
  beforeEach(() => {
    // TODO: remove global fetch mock by explicitly mocking all the service API calls
    // @ts-expect-error TS(2322): Type 'Mock<Promise<{ ok: true; status: number; tex... Remove this comment to see the full error message
    global.fetch = jest.fn(() => Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve('') }));
    minimalProps = {
      runUuid: 'uuid-1234-5678-9012',
      experimentId: '12345',
      getMetricPagePath: jest.fn(),
      handleSetRunTag: jest.fn(),
      setTagApi: jest.fn(),
      deleteTagApi: jest.fn(),
    };
    const modelVersion = mockModelVersionDetailed('Model A', 1, Stages.PRODUCTION, ModelVersionStatus.READY);
    minimalStoreRaw = {
      entities: {
        runInfosByUuid: {
          'uuid-1234-5678-9012': (RunInfo as any).fromJs({
            run_uuid: 'uuid-1234-5678-9012',
            experiment_id: '12345',
            user_id: 'me@me.com',
            status: 'RUNNING',
            artifact_uri: 'dbfs:/databricks/abc/uuid-1234-5678-9012',
            lifecycle_stage: 'active',
          }),
        },
        // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
        artifactsByRunUuid: { 'uuid-1234-5678-9012': new ArtifactNode(true) },
        experimentsById: {
          12345: (Experiment as any).fromJs({
            experiment_id: '12345',
            name: 'my experiment',
            artifact_location: 'dbfs:/databricks/abc',
            lifecycle_stage: 'active',
            last_update_time: 12345678999,
            creation_time: 12345678900,
            tags: [],
          }),
        },
        modelVersionsByModel: {
          'Model A': {
            1: modelVersion,
          },
        },
        tagsByRunUuid: { 'uuid-1234-5678-9012': {} },
        paramsByRunUuid: { 'uuid-1234-5678-9012': {} },
        runDatasetsByUuid: { 'uuid-1234-5678-9012': [] },
        latestMetricsByRunUuid: { 'uuid-1234-5678-9012': {} },
        artifactRootUriByRunUuid: { 'uuid-1234-5678-9012': 'root/uri' },
      },
      apis: {},
      compareExperiments: {},
    };
    minimalStore = mockStore(minimalStoreRaw);
  });
  test('should render with minimal props without exploding', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <RunView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find(RunView).length).toBe(1);
  });
  test('With no tags, params, duration - getRunCommand & metadata list', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <RunView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    ).find(RunView);
    expect(wrapper.html()).not.toContain('Git Commit');
    expect(wrapper.html()).not.toContain('Entry Point');
    expect(wrapper.html()).not.toContain('Duration');
    expect(wrapper.html()).not.toContain('Parent Run');
    expect(wrapper.html()).not.toContain('Job Output');
    expect(wrapper.html()).not.toContain('Run Command');
  });
  test('With non-empty tags, params, duration - getRunCommand & metadata list', () => {
    const store = mockStore({
      ...minimalStoreRaw,
      entities: {
        ...minimalStoreRaw.entities,
        runInfosByUuid: {
          ...minimalStoreRaw.entities.runInfosByUuid,
          'uuid-1234-5678-9012': (RunInfo as any).fromJs({
            run_uuid: 'uuid-1234-5678-9012',
            experiment_id: '12345',
            user_id: 'me@me.com',
            status: 'RUNNING',
            start_time: 12345678990,
            end_time: 12345678999,
            artifact_uri: 'dbfs:/databricks/abc/uuid-1234-5678-9012',
            lifecycle_stage: 'active',
          }),
        },
        tagsByRunUuid: {
          'uuid-1234-5678-9012': {
            'mlflow.source.type': (RunTag as any).fromJs({
              key: 'mlflow.source.type',
              value: 'PROJECT',
            }),
            'mlflow.source.name': (RunTag as any).fromJs({
              key: 'mlflow.source.name',
              value: 'notebook',
            }),
            'mlflow.source.git.commit': (RunTag as any).fromJs({
              key: 'mlflow.source.git.commit',
              value: 'abc',
            }),
            'mlflow.project.entryPoint': (RunTag as any).fromJs({
              key: 'mlflow.project.entryPoint',
              value: 'entry',
            }),
            'mlflow.project.backend': (RunTag as any).fromJs({
              key: 'mlflow.project.backend',
              value: 'databricks',
            }),
            'mlflow.parentRunId': (RunTag as any).fromJs({
              key: 'mlflow.parentRunId',
              value: 'run2-5656-7878-9090',
            }),
            'mlflow.databricks.runURL': (RunTag as any).fromJs({
              key: 'mlflow.databricks.runURL',
              value: 'https:/databricks.com/jobs_url/123',
            }),
          },
        },
        paramsByRunUuid: {
          'uuid-1234-5678-9012': {
            p1: (Param as any).fromJs({ key: 'p1', value: 'v1' }),
            p2: (Param as any).fromJs({ key: 'p2', value: 'v2' }),
          },
        },
      },
    });
    wrapper = mountWithIntl(
      <Provider store={store}>
        <BrowserRouter>
          <RunView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    ).find(RunView);

    expect(wrapper.html()).toContain('Git Commit');
    expect(wrapper.html()).toContain('Entry Point');
    expect(wrapper.html()).toContain('Duration');
    expect(wrapper.html()).toContain('Parent Run');
    expect(wrapper.html()).toContain('Job Output');
    expect(wrapper.html()).toContain('Run Command');
    expect(wrapper.html()).toContain('mlflow run notebook -v abc -e entry -b databricks -P p1=v1 -P p2=v2');
  });
  test('state: showNoteEditor false/true -> edit button shown/hidden', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <RunView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    ).find(RunView);
    expect(wrapper.html()).toContain('edit-description-button');
    wrapper.find('button[data-test-id="edit-description-button"]').simulate('click');
    expect(wrapper.html()).not.toContain('edit-description-button');
  });
  test('should set showRunRenameModal when Rename menu item is clicked', () => {
    wrapper = mountWithIntl(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <RunView {...minimalProps} />
        </BrowserRouter>
      </Provider>,
    );
    expect(wrapper.find(RunViewImpl).instance().state.showRunRenameModal).toBe(false);
    wrapper.find("button[data-test-id='overflow-menu-trigger']").simulate('click');
    wrapper.find('[data-test-id="overflow-rename-button"]').hostNodes().simulate('click');
    expect(wrapper.find(RunViewImpl).instance().state.showRunRenameModal).toBe(true);
  });
});
