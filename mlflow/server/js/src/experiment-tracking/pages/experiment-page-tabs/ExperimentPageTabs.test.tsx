import { DesignSystemProvider } from '@databricks/design-system';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { graphql, rest } from 'msw';
import { IntlProvider } from 'react-intl';
import { setupTestRouter, testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import { setupServer } from '../../../common/utils/setup-msw';
import { TestApolloProvider } from '../../../common/utils/TestApolloProvider';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { NOTE_CONTENT_TAG } from '../../utils/NoteUtils';
import ExperimentPageTabs from './ExperimentPageTabs';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { shouldEnableExperimentKindInference } from '../../../common/utils/FeatureUtils';
import { ExperimentKind } from '../../constants';
import { createMLflowRoutePath } from '../../../common/utils/RoutingUtils';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(60000); // Larger timeout for integration testing

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableExperimentKindInference: jest.fn(() => false),
}));

jest.mock('../experiment-logged-models/ExperimentLoggedModelListPage', () => ({
  // mock default export
  __esModule: true,
  default: () => <div>ExperimentLoggedModelListPage</div>,
}));

jest.mock('../experiment-traces/ExperimentTracesPage', () => ({
  // mock default export
  __esModule: true,
  default: () => <div>Experiment traces page</div>,
}));

describe('ExperimentLoggedModelListPage', () => {
  const { history } = setupTestRouter();
  const createTestExperiment = (id = '12345678', name = 'Test experiment name') => {
    return {
      __typename: 'MlflowExperiment',
      artifactLocation: null,
      name,
      creationTime: null,
      experimentId: id,
      lastUpdateTime: null,
      lifecycleStage: 'active',
      tags: [],
    };
  };

  const createTestExperimentResponse = (experiment: any) => ({
    mlflowGetExperiment: {
      __typename: 'MlflowGetExperimentResponse',
      apiError: null,
      experiment,
    },
  });

  const server = setupServer(
    graphql.query('MlflowGetExperimentQuery', (req, res, ctx) =>
      res(ctx.data(createTestExperimentResponse(createTestExperiment()))),
    ),
  );

  const renderTestComponent = () => {
    const queryClient = new QueryClient();
    return render(
      <TestApolloProvider disableCache>
        <MockedReduxStoreProvider state={{ entities: { experimentTagsByExperimentId: {}, experimentsById: {} } }}>
          <IntlProvider locale="en">
            <QueryClientProvider client={queryClient}>
              <DesignSystemProvider>
                <TestRouter
                  routes={[
                    testRoute(<ExperimentPageTabs />, createMLflowRoutePath('/experiments/:experimentId/:tabName')),
                  ]}
                  history={history}
                  initialEntries={[createMLflowRoutePath('/experiments/12345678/models')]}
                />
              </DesignSystemProvider>
            </QueryClientProvider>
          </IntlProvider>
        </MockedReduxStoreProvider>
      </TestApolloProvider>,
    );
  };

  beforeAll(() => {
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
    server.listen();
  });

  beforeEach(() => {
    server.resetHandlers();
    jest.mocked(shouldEnableExperimentKindInference).mockReturnValue(false);
  });

  test('should display experiment title when fetched', async () => {
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByText('Test experiment name')).toBeInTheDocument();
    });
  });

  test('should show error when experiment is missing in API response', async () => {
    server.resetHandlers(
      graphql.query('MlflowGetExperimentQuery', (req, res, ctx) =>
        res(
          ctx.data({
            mlflowGetExperiment: { experiment: null, apiError: { message: 'The requested resource was not found.' } },
          }),
        ),
      ),
      rest.post('/ajax-api/2.0/mlflow/logged-models/search', (req, res, ctx) => res(ctx.json({ models: [] }))),
    );
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByText(/The requested resource was not found./)).toBeInTheDocument();
    });
  });

  test('integration test: should display popover about inferred experiment kind', async () => {
    const confirmTagApiSpy = jest.fn();

    // Enable feature flags
    jest.mocked(shouldEnableExperimentKindInference).mockReturnValue(true);

    // Simulate experiment's traces so "GenAI" experiment kind is inferred
    server.use(
      rest.get('/ajax-api/2.0/mlflow/traces', (req, res, ctx) => {
        return res(ctx.json({ traces: [{ id: 'trace1' }] }));
      }),
      rest.post('/ajax-api/2.0/mlflow/runs/search', (req, res, ctx) => {
        return res(ctx.json({ runs: [{ info: { run_uuid: 'run1' } }] }));
      }),
      rest.post('/ajax-api/2.0/mlflow/experiments/set-experiment-tag', (req, res, ctx) => {
        confirmTagApiSpy(req.body);
        return res(ctx.json({}));
      }),
    );

    renderTestComponent();

    // Check that the popover is displayed
    expect(
      await screen.findByText(
        "We've automatically detected the experiment type to be 'GenAI apps & agents'. You can either confirm or change the type.",
      ),
    ).toBeInTheDocument();

    // Check we've been redirected to the Traces tab
    expect(await screen.findByText('Experiment traces page')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Confirm' }));

    await waitFor(() => {
      expect(confirmTagApiSpy).toHaveBeenCalledWith({
        experiment_id: '12345678',
        key: 'mlflow.experimentKind',
        value: ExperimentKind.GENAI_DEVELOPMENT,
      });
    });
  });

  test('integration test: should display modal with information about impossible experiment type inference', async () => {
    const confirmTagApiSpy = jest.fn();

    // Enable feature flags
    jest.mocked(shouldEnableExperimentKindInference).mockReturnValue(true);

    // Simulate experiment's traces so "GenAI" experiment kind is inferred
    server.use(
      rest.get('/ajax-api/2.0/mlflow/traces', (req, res, ctx) => {
        return res(ctx.json({ traces: [] }));
      }),
      rest.post('/ajax-api/2.0/mlflow/runs/search', (req, res, ctx) => {
        return res(ctx.json({ runs: [] }));
      }),
      rest.post('/ajax-api/2.0/mlflow/experiments/set-experiment-tag', (req, res, ctx) => {
        confirmTagApiSpy(req.body);
        return res(ctx.json({}));
      }),
    );

    renderTestComponent();

    expect(
      await screen.findByText(
        "We support multiple experiment types, each with its own set of features. Please select the type you'd like to use. You can change this later if needed.",
      ),
    ).toBeInTheDocument();

    const modal = screen.getByRole('dialog');

    await userEvent.click(within(modal).getByRole('radio', { name: 'GenAI apps & agents' }));
    await userEvent.click(within(modal).getByRole('button', { name: 'Confirm' }));

    await waitFor(() => {
      expect(confirmTagApiSpy).toHaveBeenCalledWith({
        experiment_id: '12345678',
        key: 'mlflow.experimentKind',
        value: ExperimentKind.GENAI_DEVELOPMENT,
      });
    });
  });
});
