import { DeepPartial } from 'redux';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { renderWithIntl, act, screen, within } from 'common/utils/TestUtils.react17';
import { RunViewOverviewV2 } from './RunViewOverviewV2';
import { ReduxState } from '../../../redux-types';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { cloneDeep, merge } from 'lodash';
import userEvent from '@testing-library/user-event';
import { getRunApi, setTagApi } from '../../actions';
import { NOTE_CONTENT_TAG } from '../../utils/NoteUtils';
import { DesignSystemProvider } from '@databricks/design-system';
import { EXPERIMENT_PARENT_ID_TAG } from '../experiment-page/utils/experimentPage.common-utils';
import type { KeyValueEntity } from '../../types';
import { shouldEnableDeepLearningUI } from 'common/utils/FeatureUtils';

jest.mock('../../../common/components/Prompt', () => ({
  Prompt: jest.fn(() => <div />),
}));

jest.mock('../../actions', () => ({
  setTagApi: jest.fn(() => ({ type: 'setTagApi', payload: Promise.resolve() })),
  getRunApi: jest.fn(() => ({ type: 'getRunApi', payload: Promise.resolve() })),
}));

jest.setTimeout(30000); // Larget timeout for integration testing

const testRunUuid = 'test-run-uuid';
const testRunName = 'Test run name';
const testExperimentId = '12345';

const testEntitiesState: Partial<ReduxState['entities']> = {
  tagsByRunUuid: {
    [testRunUuid]: {},
  },
  runInfosByUuid: {
    [testRunUuid]: {
      artifact_uri: 'file:/mlflow/tracking/12345/artifacts',
      start_time: 1672578000000, // 2023-01-01 14:00:00
      end_time: 1672578300000, // 2023-01-01 14:05:00
      experiment_id: testExperimentId,
      lifecycle_stage: 'active',
      run_name: testRunName,
      run_uuid: testRunUuid,
      status: 'FINISHED',
    } as any,
  },
  runDatasetsByUuid: {
    [testRunUuid]: [],
  },
  paramsByRunUuid: {
    [testRunUuid]: {
      param_a: { key: 'param_a', value: 'param_a_value' },
      param_b: { key: 'param_b', value: 'param_b_value' },
      param_c: { key: 'param_c', value: 'param_c_value' },
    } as any,
  },
  latestMetricsByRunUuid: {
    [testRunUuid]: {
      metric_1: { key: 'metric_1', value: 'metric_1_value' },
      metric_2: { key: 'metric_2', value: 'metric_2_value' },
      metric_3: { key: 'metric_3', value: 'metric_3_value' },
    } as any,
  },
  modelVersionsByRunUuid: {
    [testRunUuid]: [],
  },
};

describe('RunViewOverview integration', () => {
  const onRunDataUpdated = jest.fn();
  const renderComponent = (entities?: DeepPartial<ReduxState['entities']>) => {
    const state: DeepPartial<ReduxState> = {
      entities: merge(cloneDeep(testEntitiesState), entities),
    };

    return renderWithIntl(
      <DesignSystemProvider>
        <MockedReduxStoreProvider state={state}>
          <MemoryRouter>
            <RunViewOverviewV2 onRunDataUpdated={onRunDataUpdated} runUuid={testRunUuid} />
          </MemoryRouter>
        </MockedReduxStoreProvider>
      </DesignSystemProvider>,
    );
  };
  test('Render component in the simplest form and assert minimal set of values', () => {
    renderComponent();

    // Empty description
    expect(screen.getByText('No description')).toBeInTheDocument();

    // Start time
    expect(screen.getByRole('row', { name: /Created at\s+2023-01-01 13:00:00/ })).toBeInTheDocument();

    // Status
    expect(screen.getByRole('row', { name: /Status\s+Finished/ })).toBeInTheDocument();

    // Run ID
    expect(screen.getByRole('row', { name: /Run ID\s+test-run-uuid/ })).toBeInTheDocument();

    // Duration
    expect(screen.getByRole('row', { name: /Duration\s+5\.0min/ })).toBeInTheDocument();

    // Datasets
    expect(screen.getByRole('row', { name: /Datasets used\s+—/ })).toBeInTheDocument();

    // Logged models
    expect(screen.getByRole('row', { name: /Logged models\s+—/ })).toBeInTheDocument();

    // Registered models
    expect(screen.getByRole('row', { name: /Registered models\s+—/ })).toBeInTheDocument();
  });
  test('Render and change run description', async () => {
    renderComponent({
      tagsByRunUuid: {
        [testRunUuid]: {
          [NOTE_CONTENT_TAG]: { key: NOTE_CONTENT_TAG, value: 'existing description' },
        },
      },
    });

    // Non-empty description
    expect(screen.getByText('existing description')).toBeInTheDocument();

    await act(async () => {
      userEvent.click(screen.getByRole('button', { name: 'Edit description' }));
    });
    await act(async () => {
      userEvent.type(screen.getByTestId('text-area'), '{selectall}{backspace}hello');
    });
    await act(async () => {
      userEvent.click(screen.getByTestId('editable-note-save-button'));
    });

    expect(setTagApi).toBeCalledWith('test-run-uuid', NOTE_CONTENT_TAG, 'hello');
  });

  test.each([
    ['200ms', 30, 230],
    ['1.0s', 1000, 2000],
    ['1.5min', 500, 90500],
    ['1.3h', 10, 4500010],
  ])('Properly render %s formatted run duration', (expectedDuration, start_time, end_time) => {
    renderComponent({
      runInfosByUuid: {
        [testRunUuid]: { start_time, end_time },
      },
    });

    expect(screen.getByRole('cell', { name: expectedDuration })).toBeInTheDocument();
  });

  test("Render cell with run's author", () => {
    renderComponent({
      tagsByRunUuid: {
        [testRunUuid]: {
          'mlflow.user': { key: 'mlflow.user', value: 'test.joe@databricks.com' },
        },
      },
    });

    expect(screen.getByRole('cell', { name: 'test.joe@databricks.com' })).toBeInTheDocument();
  });

  test('Render cell with logged models and display dropdown menu', async () => {
    renderComponent({
      runInfosByUuid: {
        [testRunUuid]: {
          artifact_uri: 'file:/mlflow/tracking/12345/artifacts',
        },
      },
      tagsByRunUuid: {
        [testRunUuid]: {
          'mlflow.log-model.history': {
            key: 'mlflow.log-model.history',
            value: JSON.stringify([
              {
                artifact_path: 'path/to/model',
                flavors: { sklearn: {} },
                utc_time_created: 1672578000000,
              },
              {
                artifact_path: 'path/to/xgboost/model',
                flavors: { xgboost: {} },
                utc_time_created: 1672578000000,
              },
            ]),
          },
        },
      },
    });

    const modelsCell = screen.getByRole('cell', { name: /sklearn/ });

    // in the new tab-based run view, artifacts have a slightly different path
    // see getRunPageRoute() in src/experiment-tracking/routes.ts for more details
    const artifactStr = shouldEnableDeepLearningUI() ? 'artifacts' : 'artifactPath';
    const pathRegex = new RegExp(`test-run-uuid/${artifactStr}/path/to/model$`);
    expect(modelsCell).toBeInTheDocument();
    expect(within(modelsCell).getByRole('link')).toHaveAttribute('href', expect.stringMatching(pathRegex));
    expect(within(modelsCell).getByRole('button', { name: '+1' })).toBeInTheDocument();
    await act(async () => {
      userEvent.click(within(modelsCell).getByRole('button', { name: '+1' }));
    });

    expect(screen.getByText('xgboost')).toBeInTheDocument();
  });

  test('Render child run and check for the existing parent run link', () => {
    const testParentRunUuid = 'test-parent-run-uuid';
    const testParentRunName = 'Test parent run name';

    renderComponent({
      runInfosByUuid: {
        [testRunUuid]: {},
        [testParentRunUuid]: {
          experiment_id: testExperimentId,
          run_uuid: testParentRunUuid,
          run_name: testParentRunName,
        },
      },
      tagsByRunUuid: {
        [testRunUuid]: {
          [EXPERIMENT_PARENT_ID_TAG]: {
            key: EXPERIMENT_PARENT_ID_TAG,
            value: testParentRunUuid,
          } as KeyValueEntity,
        },
      },
    });

    expect(screen.getByRole('row', { name: /Parent run\s+Test parent run name/ })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Test parent run name/ })).toBeInTheDocument();
  });

  test('Render child run and load the parent run name if it does not exist', () => {
    const testParentRunUuid = 'test-parent-run-uuid';

    renderComponent({
      runInfosByUuid: {
        [testRunUuid]: {},
      },
      tagsByRunUuid: {
        [testRunUuid]: {
          [EXPERIMENT_PARENT_ID_TAG]: {
            key: EXPERIMENT_PARENT_ID_TAG,
            value: testParentRunUuid,
          } as KeyValueEntity,
        },
      },
    });

    expect(screen.getByText('Parent run name loading')).toBeInTheDocument();
    expect(getRunApi).toBeCalledWith(testParentRunUuid);
  });

  // TODO: expand integration tests when tags, params, metrics and models are complete
});
