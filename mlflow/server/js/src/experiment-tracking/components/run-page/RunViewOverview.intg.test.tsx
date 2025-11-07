import type { DeepPartial } from 'redux';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { waitFor, renderWithIntl, screen, within } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { RunViewOverview } from './RunViewOverview';
import type { ReduxState } from '../../../redux-types';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { cloneDeep, merge } from 'lodash';
import userEvent from '@testing-library/user-event';
import { setTagApi } from '../../actions';
import { useGetRunQuery } from './hooks/useGetRunQuery';
import { usePromptVersionsForRunQuery } from '../../pages/prompts/hooks/usePromptVersionsForRunQuery';
import { NOTE_CONTENT_TAG } from '../../utils/NoteUtils';
import { DesignSystemProvider } from '@databricks/design-system';
import { EXPERIMENT_PARENT_ID_TAG } from '../experiment-page/utils/experimentPage.common-utils';
import type { RunInfoEntity } from '../../types';
import type { KeyValueEntity } from '../../../common/types';
import { TestApolloProvider } from '../../../common/utils/TestApolloProvider';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import type { LoggedModelProto } from '../../types';
import { type RunPageModelVersionSummary } from './hooks/useUnifiedRegisteredModelVersionsSummariesForRun';
import { ExperimentKind, MLFLOW_LINKED_PROMPTS_TAG } from '../../constants';

jest.mock('../../../common/components/Prompt', () => ({
  Prompt: jest.fn(() => <div />),
}));

jest.mock('../../actions', () => ({
  setTagApi: jest.fn(() => ({ type: 'setTagApi', payload: Promise.resolve() })),
}));

jest.mock('./hooks/useGetRunQuery', () => ({
  useGetRunQuery: jest.fn(),
}));

const testPromptName = 'test-prompt';
const testPromptVersion = 1;
const testPromptName2 = 'test-prompt-2';

jest.mock('../../pages/prompts/hooks/usePromptVersionsForRunQuery', () => ({
  usePromptVersionsForRunQuery: jest.fn(() => ({
    data: {
      model_versions: [
        {
          name: testPromptName,
          version: testPromptVersion,
        },
      ],
    },
  })),
}));

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000); // Larget timeout for integration testing

const testRunUuid = 'test-run-uuid';
const testRunName = 'Test run name';
const testExperimentId = '12345';

const testRunInfo = {
  artifactUri: 'file:/mlflow/tracking/12345/artifacts',
  startTime: 1672578000000, // 2023-01-01 14:00:00
  endTime: 1672578300000, // 2023-01-01 14:05:00
  experimentId: testExperimentId,
  lifecycleStage: 'active',
  runName: testRunName,
  runUuid: testRunUuid,
  status: 'FINISHED' as const,
};

const testEntitiesState: Partial<ReduxState['entities']> = {
  tagsByRunUuid: {
    [testRunUuid]: {},
  },
  runInfosByUuid: {
    [testRunUuid]: testRunInfo,
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
  beforeEach(() => {
    jest.mocked(useGetRunQuery).mockReset();
  });

  const onRunDataUpdated = jest.fn();
  const renderComponent = ({
    tags = {},
    runInfo,
    reduxStoreEntities = {},
    loggedModelsV3,
    registeredModelVersionSummaries = [],
    experimentKind = ExperimentKind.CUSTOM_MODEL_DEVELOPMENT,
  }: {
    tags?: Record<string, KeyValueEntity>;
    reduxStoreEntities?: DeepPartial<ReduxState['entities']>;
    runInfo?: Partial<RunInfoEntity>;
    registeredModelVersionSummaries?: RunPageModelVersionSummary[];
    loggedModelsV3?: LoggedModelProto[] | undefined;
    experimentKind?: ExperimentKind;
  } = {}) => {
    const state: DeepPartial<ReduxState> = {
      entities: merge(
        cloneDeep(testEntitiesState),
        {
          tagsByRunUuid: {
            [testRunUuid]: tags,
          },
        },
        reduxStoreEntities,
      ),
    };

    const queryClient = new QueryClient();

    return renderWithIntl(
      <DesignSystemProvider>
        <QueryClientProvider client={queryClient}>
          <MockedReduxStoreProvider state={state}>
            <TestApolloProvider>
              <MemoryRouter>
                <RunViewOverview
                  onRunDataUpdated={onRunDataUpdated}
                  runUuid={testRunUuid}
                  latestMetrics={testEntitiesState.latestMetricsByRunUuid?.[testRunUuid] || {}}
                  params={testEntitiesState.paramsByRunUuid?.[testRunUuid] || {}}
                  runInfo={{ ...testRunInfo, ...runInfo }}
                  tags={merge({}, testEntitiesState.tagsByRunUuid?.[testRunUuid], tags) || {}}
                  registeredModelVersionSummaries={registeredModelVersionSummaries}
                  loggedModelsV3={loggedModelsV3}
                  experimentKind={experimentKind}
                />
              </MemoryRouter>
            </TestApolloProvider>
          </MockedReduxStoreProvider>
        </QueryClientProvider>
      </DesignSystemProvider>,
    );
  };

  test('Render component in the simplest form and assert minimal set of values', async () => {
    const { container } = renderComponent();

    // Empty description
    expect(screen.getByText('No description')).toBeInTheDocument();

    // Start time
    expect(container.textContent).toMatch(/Created at\s*01\/01\/2023, 01:00:00 PM/);

    // Status
    expect(container.textContent).toMatch(/Status\s*Finished/);

    // Run ID
    expect(container.textContent).toMatch(/Run ID\s*test-run-uuid/);

    // Duration
    expect(container.textContent).toMatch(/Duration\s*5\.0min/);

    // Experiment ID
    expect(container.textContent).toMatch(/Experiment ID\s*12345/);

    // Datasets
    expect(container.textContent).toMatch(/Datasets\s*None/);

    // Registered models
    expect(container.textContent).toMatch(/Registered models\s*None/);
  });
  test('Render and change run description', async () => {
    renderComponent({
      tags: {
        [NOTE_CONTENT_TAG]: { key: NOTE_CONTENT_TAG, value: 'existing description' },
      },
    });

    // Non-empty description
    expect(screen.getByText('existing description')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Edit description' }));
    await userEvent.clear(screen.getByTestId('text-area'));
    await userEvent.type(screen.getByTestId('text-area'), 'hello');
    await userEvent.click(screen.getByTestId('editable-note-save-button'));

    expect(setTagApi).toHaveBeenCalledWith('test-run-uuid', NOTE_CONTENT_TAG, 'hello');
  });

  test.each([
    ['200ms', 30, 230],
    ['1.0s', 1000, 2000],
    ['1.5min', 500, 90500],
    ['1.3h', 10, 4500010],
  ])('Properly render %s formatted run duration', (expectedDuration, startTime, endTime) => {
    const { container } = renderComponent({
      runInfo: { startTime, endTime },
    });

    expect(container.textContent).toMatch(expectedDuration);
  });

  test("Render cell with run's author", () => {
    const { container } = renderComponent({
      tags: {
        'mlflow.user': { key: 'mlflow.user', value: 'test.joe@databricks.com' },
      },
    });

    expect(container.textContent).toMatch('test.joe@databricks.com');
  });

  test('Render cell with logged models and display dropdown menu', async () => {
    renderComponent({
      runInfo: {
        artifactUri: 'file:/mlflow/tracking/12345/artifacts',
      },
      tags: {
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
    });

    await waitFor(() => {
      expect(screen.getByText(/sklearn/)).toBeInTheDocument();
    });

    const modelLink = screen.getByRole('link', { name: /sklearn/ });
    expect(modelLink).toHaveAttribute('href', expect.stringMatching(/test-run-uuid\/artifacts\/path\/to\/model$/));

    const moreButton = screen.getByRole('button', { name: '+1' });
    expect(moreButton).toBeInTheDocument();

    await userEvent.click(moreButton);
    expect(screen.getByText('xgboost')).toBeInTheDocument();
  });

  test('Render cell with registered models and display dropdown menu', async () => {
    renderComponent({
      runInfo: {
        artifactUri: 'file:/mlflow/tracking/12345/artifacts',
      },
      registeredModelVersionSummaries: [
        {
          displayedName: 'test-registered-model',
          version: '1',
          link: '/models/test-registered-model/versions/1',
          source: 'file:/mlflow/tracking/12345/artifacts',
          status: 'READY',
        },
        {
          displayedName: 'another-test-registered-model',
          version: '2',
          link: '/models/another-test-registered-model/versions/2',
          source: 'file:/mlflow/tracking/12345/artifacts',
          status: 'READY',
        },
      ],
      tags: {},
    });

    await waitFor(() => {
      expect(screen.getByText(/test-registered-model/)).toBeInTheDocument();
    });

    const modelLink = screen.getByRole('link', { name: /test-registered-model/ });
    expect(modelLink).toHaveAttribute('href', expect.stringMatching(/test-registered-model\/versions\/1$/));

    const moreButton = screen.getByRole('button', { name: '+1' });
    expect(moreButton).toBeInTheDocument();

    await userEvent.click(moreButton);
    expect(screen.getByText('another-test-registered-model')).toBeInTheDocument();
  });

  test('Render child run and check for the existing parent run link', () => {
    const testParentRunUuid = 'test-parent-run-uuid';
    const testParentRunName = 'Test parent run name';

    (useGetRunQuery as jest.Mock).mockReturnValue({
      data: { info: { runName: testParentRunName, runUuid: testParentRunUuid, experimentId: testExperimentId } },
      loading: false,
      apolloError: undefined,
      apiError: undefined,
      refetchRun: jest.fn(),
    });

    const { container } = renderComponent({
      tags: {
        [EXPERIMENT_PARENT_ID_TAG]: {
          key: EXPERIMENT_PARENT_ID_TAG,
          value: testParentRunUuid,
        } as KeyValueEntity,
      },
      runInfo: {},
      reduxStoreEntities: {
        runInfosByUuid: {
          [testParentRunUuid]: {
            experimentId: testExperimentId,
            runUuid: testParentRunUuid,
            runName: testParentRunName,
          },
        },
      },
    });

    expect(container.textContent).toMatch(/Parent run\s*Test parent run name/);
    expect(screen.getByRole('link', { name: /Test parent run name/ })).toBeInTheDocument();
  });

  test('Render child run and load the parent run name if it does not exist', async () => {
    const testParentRunUuid = 'test-parent-run-uuid';

    renderComponent({
      tags: {
        [EXPERIMENT_PARENT_ID_TAG]: {
          key: EXPERIMENT_PARENT_ID_TAG,
          value: testParentRunUuid,
        },
      },
    });

    await waitFor(() => {
      expect(screen.getByText('Parent run name loading')).toBeInTheDocument();
      expect(useGetRunQuery).toHaveBeenCalledWith({ runUuid: testParentRunUuid, disabled: false });
    });
  });

  test('Run overview contains prompts from run tags', async () => {
    renderComponent({
      tags: {
        [MLFLOW_LINKED_PROMPTS_TAG]: {
          key: MLFLOW_LINKED_PROMPTS_TAG,
          value: JSON.stringify([{ name: testPromptName2, version: testPromptVersion.toString() }]),
        },
      },
    });

    await waitFor(() => {
      expect(screen.getByText(`${testPromptName2} (v${testPromptVersion})`)).toBeInTheDocument();
    });
  });

  test('Run overview contains registered prompts', async () => {
    renderComponent();

    await waitFor(() => {
      expect(screen.getByText(`${testPromptName} (v${testPromptVersion})`)).toBeInTheDocument();
      expect(usePromptVersionsForRunQuery).toHaveBeenCalledWith({ runUuid: testRunUuid });
    });
  });
  // TODO: expand integration tests when tags, params, metrics and models are complete
});
