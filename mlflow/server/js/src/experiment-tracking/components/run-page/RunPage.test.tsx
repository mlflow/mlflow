import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { renderWithIntl, screen, waitFor, within } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { getExperimentApi, getRunApi, updateRunApi } from '../../actions';
import { searchModelVersionsApi } from '../../../model-registry/actions';
import { merge } from 'lodash';
import type { ReduxState } from '../../../redux-types';
import type { DeepPartial } from 'redux';
import { RunPage } from './RunPage';
import { testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import type { RunInfoEntity } from '../../types';
import userEvent from '@testing-library/user-event';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';
import { TestApolloProvider } from '../../../common/utils/TestApolloProvider';
import {
  shouldEnableGraphQLRunDetailsPage,
  shouldUseGetLoggedModelsBatchAPI,
} from '../../../common/utils/FeatureUtils';
import { setupServer } from '../../../common/utils/setup-msw';
import { graphql, rest } from 'msw';
import type { GetRun, GetRunVariables } from '../../../graphql/__generated__/graphql';
import { MlflowRunStatus } from '../../../graphql/__generated__/graphql';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import Utils from '../../../common/utils/Utils';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(90000); // Higher timeout due to integration testing and tables

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableGraphQLRunDetailsPage: jest.fn(),
  isModelsInUCEnabled: jest.fn(),
  isRegisterUCModelFromUIEnabled: jest.fn(),
  shouldUseGetLoggedModelsBatchAPI: jest.fn(),
}));

const mockAction = (id: string) => ({ type: 'action', payload: Promise.resolve(), meta: { id } });

jest.mock('../../actions', () => ({
  getExperimentApi: jest.fn(() => mockAction('experiment_request')),
  getRunApi: jest.fn(() => mockAction('run_request')),
  updateRunApi: jest.fn(() => mockAction('run_request')),
}));

jest.mock('../../../model-registry/actions', () => ({
  searchModelVersionsApi: jest.fn(() => mockAction('models_request')),
  searchRegisteredModelsApi: jest.fn(() => mockAction('search_registered_models_request')),
  ucRegisterModelVersionApi: jest.fn(() => mockAction('register_model_version_request')),
}));

const testRunUuid = 'test-run-uuid';
const testExperimentId = '12345';
const testRunInfo: Partial<RunInfoEntity> = {
  runName: 'Test run Name',
  experimentId: testExperimentId,
};

describe('RunPage (legacy redux + REST API)', () => {
  const server = setupServer();

  beforeEach(() => {
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
    jest.mocked(shouldEnableGraphQLRunDetailsPage).mockImplementation(() => false);
  });

  const mountComponent = (
    entities: DeepPartial<ReduxState['entities']> = {},
    apis: DeepPartial<ReduxState['apis']> = {},
  ) => {
    const state: DeepPartial<ReduxState> = {
      entities: merge(
        {
          artifactRootUriByRunUuid: {},
          runInfosByUuid: {},
          experimentsById: {},
          tagsByRunUuid: {},
          latestMetricsByRunUuid: {},
          runDatasetsByUuid: {},
          paramsByRunUuid: {},
          modelVersionsByRunUuid: {},
        },
        entities,
      ),
      apis: merge(
        {
          experiment_request: { active: true },
          run_request: { active: true },
        },
        apis,
      ),
    };

    const queryClient = new QueryClient();

    const renderResult = renderWithIntl(
      <TestApolloProvider>
        <QueryClientProvider client={queryClient}>
          <MockedReduxStoreProvider state={state}>
            <TestRouter
              initialEntries={[`/experiment/${testExperimentId}/run/${testRunUuid}`]}
              routes={[testRoute(<RunPage />, '/experiment/:experimentId/run/:runUuid')]}
            />
          </MockedReduxStoreProvider>
        </QueryClientProvider>
        ,
      </TestApolloProvider>,
    );

    return renderResult;
  };

  beforeEach(() => {
    jest.mocked(getRunApi).mockClear();
    jest.mocked(getExperimentApi).mockClear();
    jest.mocked(searchModelVersionsApi).mockClear();
    jest.mocked(updateRunApi).mockClear();
  });

  test('Start fetching run when store is empty and experiment and indicate loading state', async () => {
    mountComponent();

    await waitFor(() => {
      expect(screen.getByText('Run page loading')).toBeInTheDocument();
    });

    expect(getRunApi).toHaveBeenCalledWith(testRunUuid);
    expect(getExperimentApi).toHaveBeenCalledWith(testExperimentId);
    expect(searchModelVersionsApi).toHaveBeenCalledWith({ run_id: testRunUuid });
  });

  const entitiesWithMockRun = {
    runInfosByUuid: { [testRunUuid]: testRunInfo },
    experimentsById: {
      [testExperimentId]: { experimentId: testExperimentId, name: 'Test experiment name' },
    },
    tagsByRunUuid: { [testRunUuid]: {} },
    latestMetricsByRunUuid: {},
    runDatasetsByUuid: {},
    paramsByRunUuid: {},
    modelVersionsByRunUuid: {},
  };

  test('Do not display loading state when run and experiments are already loaded', async () => {
    mountComponent(entitiesWithMockRun);

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Test run Name/ })).toBeInTheDocument();
    });

    expect(getRunApi).not.toHaveBeenCalled();
    expect(getExperimentApi).not.toHaveBeenCalled();
    expect(searchModelVersionsApi).toHaveBeenCalled();
  });

  test('Attempt to rename the run', async () => {
    mountComponent(entitiesWithMockRun);

    await waitFor(() => {
      expect(screen.getByLabelText('Open header dropdown menu')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByLabelText('Open header dropdown menu'));
    await userEvent.click(screen.getByRole('menuitem', { name: 'Rename' }));
    await userEvent.clear(screen.getByTestId('rename-modal-input'));
    await userEvent.type(screen.getByTestId('rename-modal-input'), 'brand_new_run_name');
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));

    expect(updateRunApi).toHaveBeenCalledWith('test-run-uuid', 'brand_new_run_name', expect.anything());
  });

  test('Display 404 page in case of missing run', async () => {
    const runFetchError = new ErrorWrapper({ error_code: 'RESOURCE_DOES_NOT_EXIST' });

    mountComponent({}, { run_request: { active: false, error: runFetchError } });

    await waitFor(() => {
      expect(screen.getByText(/Run ID test-run-uuid does not exist/)).toBeInTheDocument();
    });
  });
});

describe('RunPage (GraphQL API)', () => {
  const server = setupServer();

  beforeEach(() => {
    jest.mocked(shouldEnableGraphQLRunDetailsPage).mockImplementation(() => true);
  });

  beforeEach(() => {
    server.use(
      graphql.query<GetRun, GetRunVariables>('GetRun', (req, res, ctx) => {
        if (req.variables.data.runId === 'invalid-run-uuid') {
          return res(
            ctx.data({
              mlflowGetRun: {
                __typename: 'MlflowGetRunResponse',
                apiError: {
                  __typename: 'ApiError',
                  helpUrl: null,
                  code: 'RESOURCE_DOES_NOT_EXIST',
                  message: 'Run not found',
                },
                run: null,
              },
            }),
          );
        }
        return res(
          ctx.data({
            mlflowGetRun: {
              __typename: 'MlflowGetRunResponse',
              apiError: null,
              run: {
                // Use 'any' type to satisfy multiple query implementations
                __typename: 'MlflowRun' as any,
                data: {
                  __typename: 'MlflowRunData',
                  metrics: [
                    {
                      __typename: 'MlflowMetricExtension',
                      key: 'test-metric',
                      value: 100,
                      step: '1',
                      timestamp: '1000',
                    },
                  ],
                  params: [{ __typename: 'MlflowParam', key: 'test-param', value: 'test-param-value' }],
                  tags: [
                    { __typename: 'MlflowRunTag', key: 'test-tag-a', value: 'test-tag-a-value' },
                    { __typename: 'MlflowRunTag', key: 'test-tag-b', value: 'test-tag-b-value' },
                  ],
                },
                experiment: {
                  __typename: 'MlflowExperiment',
                  artifactLocation: null,
                  experimentId: 'test-experiment',
                  lastUpdateTime: null,
                  lifecycleStage: null,
                  name: 'test experiment',
                  tags: [],
                },
                info: {
                  __typename: 'MlflowRunInfo',
                  artifactUri: null,
                  experimentId: 'test-experiment',
                  lifecycleStage: null,
                  runName: 'test run',
                  runUuid: 'test-run-uuid',
                  status: MlflowRunStatus.FINISHED,
                  userId: null,
                  startTime: '1672578000000',
                  endTime: '1672578300000',
                },
                inputs: {
                  __typename: 'MlflowRunInputs',
                  modelInputs: null,
                  datasetInputs: [
                    {
                      __typename: 'MlflowDatasetInput',
                      dataset: {
                        __typename: 'MlflowDataset',
                        digest: 'digest',
                        name: 'dataset-name',
                        profile: 'profile',
                        schema: 'schema',
                        source: 'source',
                        sourceType: 'sourceType',
                      },
                      tags: [{ __typename: 'MlflowInputTag', key: 'tag1', value: 'value1' }],
                    },
                  ],
                },
                outputs: {
                  __typename: 'MlflowRunOutputs',
                  modelOutputs: [{ __typename: 'MlflowModelOutput', modelId: 'test-model-id', step: '1' }],
                },
                modelVersions: [],
              },
            },
          }),
        );
      }),
    );
  });

  const mountComponent = (runUuid = testRunUuid) => {
    const queryClient = new QueryClient();

    const renderResult = renderWithIntl(
      <TestApolloProvider disableCache>
        <QueryClientProvider client={queryClient}>
          <MockedReduxStoreProvider
            state={{ entities: { modelVersionsByRunUuid: {}, tagsByRunUuid: {}, modelByName: {} } }}
          >
            <DesignSystemProvider>
              <TestRouter
                initialEntries={[`/experiment/${testExperimentId}/run/${runUuid}`]}
                routes={[testRoute(<RunPage />, '/experiment/:experimentId/run/:runUuid')]}
              />
            </DesignSystemProvider>
          </MockedReduxStoreProvider>
        </QueryClientProvider>
      </TestApolloProvider>,
    );

    return renderResult;
  };

  test('Properly fetch and display basic data', async () => {
    mountComponent();

    await waitFor(() => {
      expect(screen.getByText('test run')).toBeInTheDocument();
    });

    // Tags:
    expect(screen.getByText('test-tag-a')).toBeInTheDocument();
    expect(screen.getByText('test-tag-b')).toBeInTheDocument();

    // Params table:
    expect(screen.getByText('test-param')).toBeInTheDocument();
    expect(screen.getByText('test-param-value')).toBeInTheDocument();

    // Metrics table:
    expect(screen.getByRole('link', { name: 'test-metric' })).toBeInTheDocument();
    expect(screen.getByText('100')).toBeInTheDocument();

    // Dataset:
    expect(screen.getByText('dataset-name (digest)')).toBeInTheDocument();

    // Find metadata rows and verify their values
    const durationRow = screen.getByTestId('key-value-Duration');
    expect(durationRow.lastElementChild).toHaveTextContent('5.0min');

    const experimentIdRow = screen.getByTestId('key-value-Experiment ID');
    expect(experimentIdRow.lastElementChild).toHaveTextContent('test-experiment');

    const statusRow = screen.getByTestId('key-value-Status');
    expect(statusRow.lastElementChild).toHaveTextContent('Finished');
  });

  test('Properly display duration for ongoing run', async () => {
    // Mock run response with ongoing run and endTime set to 0
    server.resetHandlers(
      graphql.query<GetRun, GetRunVariables>('GetRun', (req, res, ctx) => {
        return res(
          ctx.data({
            mlflowGetRun: {
              __typename: 'MlflowGetRunResponse',
              apiError: null,
              run: {
                __typename: 'MlflowRun' as any,
                data: null,
                experiment: {
                  __typename: 'MlflowExperiment',
                  artifactLocation: null,
                  experimentId: 'test-experiment',
                  lastUpdateTime: null,
                  lifecycleStage: null,
                  name: 'test experiment',
                  tags: [],
                },
                info: {
                  __typename: 'MlflowRunInfo',
                  artifactUri: null,
                  experimentId: 'test-experiment',
                  lifecycleStage: null,
                  runName: 'test run',
                  runUuid: 'test-run-uuid',
                  status: MlflowRunStatus.RUNNING,
                  userId: null,
                  startTime: '1672578000000',
                  endTime: '0',
                },
                inputs: null,
                outputs: null,
                modelVersions: [],
              },
            },
          }),
        );
      }),
    );

    mountComponent();

    await waitFor(() => {
      expect(screen.getByText('test run')).toBeInTheDocument();
    });

    // For ongoing runs, duration row should be present but empty
    const durationRow = screen.getByTestId('key-value-Duration');
    expect(durationRow.lastElementChild).toBeEmptyDOMElement();

    // Status row should show the running status
    const statusRow = screen.getByTestId('key-value-Status');
    expect(statusRow.lastElementChild).toHaveTextContent('Running');
  });

  test('Display 404 page in case of missing run', async () => {
    mountComponent('invalid-run-uuid');

    await waitFor(() => {
      expect(screen.getByText(/Run ID invalid-run-uuid does not exist/)).toBeInTheDocument();
    });
  });
});
