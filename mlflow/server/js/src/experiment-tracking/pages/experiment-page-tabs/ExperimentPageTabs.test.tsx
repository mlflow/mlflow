/* eslint-disable jest/no-standalone-expect */
import { jest, describe, afterEach, beforeAll, beforeEach, test, expect } from '@jest/globals';
import React from 'react';
import { DesignSystemProvider } from '@databricks/design-system';

import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { graphql, rest } from 'msw';
import { IntlProvider } from 'react-intl';
import { setupTestRouter, TestRouter, waitForRoutesToBeRendered } from '../../../common/utils/RoutingTestUtils';
import { setupServer } from '../../../common/utils/setup-msw';
import { TestApolloProvider } from '../../../common/utils/TestApolloProvider';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { NOTE_CONTENT_TAG } from '../../utils/NoteUtils';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { ExperimentKind } from '../../constants';
import { createLazyRouteElement, createRouteElement, createMLflowRoutePath } from '../../../common/utils/RoutingUtils';
import { PageId, RoutePaths } from '../../routes';
import ExperimentPageTabs from './ExperimentPageTabs';
import { shouldEnableNewEvalRunsUi, shouldEnableWorkflowBasedNavigation } from '../../../common/utils/FeatureUtils';
import { WorkflowTypeProvider } from '../../../common/contexts/WorkflowTypeContext';
import ExperimentTracesPage from '../experiment-traces/ExperimentTracesPage';
import { useHeaderVisibility } from './ExperimentPageHeaderVisibilityContext';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(60000); // Larger timeout for integration testing

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableWorkflowBasedNavigation: jest.fn().mockReturnValue(false),
  shouldEnableNewEvalRunsUi: jest.fn().mockReturnValue(false),
}));

jest.mock('../experiment-evaluation-runs/ExperimentEvaluationRunsPage', () => ({
  // mock default export
  __esModule: true,
  default: () => <div>Experiment evaluation runs page</div>,
}));

jest.mock('../experiment-logged-models/ExperimentLoggedModelListPage', () => ({
  // mock default export
  __esModule: true,
  default: () => <div>ExperimentLoggedModelListPage</div>,
}));

jest.mock('../experiment-traces/ExperimentTracesPage', () => ({
  // mock default export
  __esModule: true,
  default: jest.fn(() => <div>Experiment traces page</div>),
}));

const ExperimentPageTabsWithWorkflowType = () => (
  <WorkflowTypeProvider>
    <ExperimentPageTabs />
  </WorkflowTypeProvider>
);

const FocusModeTracesPage = () => {
  const { setHeaderHidden } = useHeaderVisibility();
  React.useEffect(() => {
    setHeaderHidden(true);
    return () => setHeaderHidden(false);
  }, [setHeaderHidden]);
  return <div>Focused page</div>;
};

jest.mock('../experiment-overview/ExperimentGenAIOverviewPage', () => ({
  // mock default export
  __esModule: true,
  default: () => <div>Experiment overview page</div>,
}));

jest.mock('../experiment-runs/ExperimentRunsPage', () => ({
  // mock default export
  __esModule: true,
  default: () => <div>Experiment runs page</div>,
}));

describe('ExperimentLoggedModelListPage', () => {
  const { history } = setupTestRouter();
  const createTestExperiment = (
    id = '12345678',
    name = 'Test experiment name',
    tags = [{ key: 'mlflow.experimentKind', value: ExperimentKind.CUSTOM_MODEL_DEVELOPMENT }],
  ) => {
    return {
      __typename: 'MlflowExperiment',
      artifactLocation: null,
      name,
      creationTime: null,
      experimentId: id,
      lastUpdateTime: null,
      lifecycleStage: 'active',
      tags,
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
    // Add MSW handlers for experiment kind inference API calls
    rest.get('/ajax-api/2.0/mlflow/traces', (req, res, ctx) => res(ctx.json({ traces: [] }))),
    rest.post('/ajax-api/2.0/mlflow/runs/search', (req, res, ctx) => res(ctx.json({ runs: [] }))),
  );

  const renderTestComponent = (initialPath = '/experiments/12345678/models', { withWorkflowType = false } = {}) => {
    const queryClient = new QueryClient();
    return render(
      <TestApolloProvider disableCache>
        <MockedReduxStoreProvider state={{ entities: { experimentTagsByExperimentId: {}, experimentsById: {} } }}>
          <IntlProvider locale="en">
            <QueryClientProvider client={queryClient}>
              <DesignSystemProvider>
                <TestRouter
                  routes={[
                    {
                      path: RoutePaths.experimentPage,
                      pageId: PageId.experimentPage,
                      element: createRouteElement(
                        withWorkflowType ? ExperimentPageTabsWithWorkflowType : ExperimentPageTabs,
                      ),
                      children: [
                        {
                          path: RoutePaths.experimentPageTabOverview,
                          pageId: PageId.experimentPageTabOverview,
                          element: createLazyRouteElement(
                            () => import('../experiment-overview/ExperimentGenAIOverviewPage'),
                          ),
                        },
                        {
                          path: RoutePaths.experimentPageTabTraces,
                          pageId: PageId.experimentPageTabTraces,
                          element: createLazyRouteElement(() => import('../experiment-traces/ExperimentTracesPage')),
                        },
                        {
                          path: RoutePaths.experimentPageTabModels,
                          pageId: PageId.experimentPageTabModels,
                          element: createLazyRouteElement(
                            () => import('../experiment-logged-models/ExperimentLoggedModelListPage'),
                          ),
                        },
                        {
                          path: RoutePaths.experimentPageTabEvaluationRuns,
                          pageId: PageId.experimentPageTabEvaluationRuns,
                          element: createLazyRouteElement(
                            () => import('../experiment-evaluation-runs/ExperimentEvaluationRunsPage'),
                          ),
                        },
                        {
                          path: RoutePaths.experimentPageTabRuns,
                          pageId: PageId.experimentPageTabRuns,
                          element: createLazyRouteElement(() => import('../experiment-runs/ExperimentRunsPage')),
                        },
                      ],
                    },
                  ]}
                  history={history}
                  initialEntries={[createMLflowRoutePath(initialPath)]}
                />
              </DesignSystemProvider>
            </QueryClientProvider>
          </IntlProvider>
        </MockedReduxStoreProvider>
      </TestApolloProvider>,
    );
  };

  beforeAll(() => {
    jest.useRealTimers();
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
  });

  beforeEach(() => {
    server.resetHandlers();
  });

  test('should display experiment title when fetched', async () => {
    renderTestComponent();

    // Wait for lazy-loaded route components to finish loading and PageLoading skeleton to be removed.
    // First test in suite takes longer because lazy modules haven't been cached yet.
    await waitFor(() => waitForRoutesToBeRendered(), { timeout: 10000 });

    await waitFor(() => {
      expect(screen.getByText('Test experiment name')).toBeInTheDocument();
    });
  });

  test('integration test: should display popover about inferred experiment kind', async () => {
    const confirmTagApiSpy = jest.fn();

    server.resetHandlers(
      graphql.query('MlflowBatchGetExperimentsQuery', (req, res, ctx) =>
        res(ctx.data(createTestExperimentResponse([createTestExperiment('12345678', 'Test experiment name', [])]))),
      ),
      graphql.query('MlflowGetExperimentQuery', (req, res, ctx) =>
        res(ctx.data(createTestExperimentResponse(createTestExperiment('12345678', 'Test experiment name', [])))),
      ),
    );

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
        undefined,
      ),
    ).toBeInTheDocument();

    // Since we started on the /models tab, we should remain on the models tab (no redirect)
    expect(await screen.findByText('ExperimentLoggedModelListPage')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Confirm' }));

    expect(confirmTagApiSpy).toHaveBeenCalledWith({
      experiment_id: '12345678',
      key: 'mlflow.experimentKind',
      value: ExperimentKind.GENAI_DEVELOPMENT,
    });
  });

  test('integration test: should display modal with information about impossible experiment type inference', async () => {
    const confirmTagApiSpy = jest.fn();
    server.resetHandlers(
      graphql.query('MlflowGetExperimentQuery', (req, res, ctx) =>
        res(ctx.data(createTestExperimentResponse(createTestExperiment('12345678', 'Test experiment name', [])))),
      ),
    );

    // Simulate empty experiment so no experiment kind can be inferred
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
        undefined,
      ),
    ).toBeInTheDocument();

    const modal = screen.getByRole('dialog');

    // GenAI apps & agents is selected by default, just click Confirm
    await userEvent.click(within(modal).getByRole('button', { name: 'Confirm' }));

    await waitFor(() => {
      expect(confirmTagApiSpy).toHaveBeenCalledWith({
        experiment_id: '12345678',
        key: 'mlflow.experimentKind',
        value: ExperimentKind.GENAI_DEVELOPMENT,
      });
    });
  });

  test('integration test: should redirect to overview tab when GenAI experiment kind is inferred', async () => {
    server.resetHandlers(
      graphql.query('MlflowGetExperimentQuery', (req, res, ctx) =>
        res(ctx.data(createTestExperimentResponse(createTestExperiment('12345678', 'Test experiment name', [])))),
      ),
    );

    // Simulate experiment with traces (GenAI)
    server.use(
      rest.get('/ajax-api/2.0/mlflow/traces', (req, res, ctx) => {
        return res(ctx.json({ traces: [{ request_id: 'trace1' }] }));
      }),
      rest.post('/ajax-api/2.0/mlflow/runs/search', (req, res, ctx) => {
        return res(ctx.json({ runs: [] }));
      }),
    );

    // Start on experiment page WITHOUT a tab
    renderTestComponent('/experiments/12345678');

    // Should redirect to overview tab for GenAI experiments
    expect(await screen.findByText('Experiment overview page')).toBeInTheDocument();
  });

  test('integration test: should redirect to runs tab when ML experiment kind is inferred', async () => {
    server.resetHandlers(
      graphql.query('MlflowGetExperimentQuery', (req, res, ctx) =>
        res(ctx.data(createTestExperimentResponse(createTestExperiment('12345678', 'Test experiment name', [])))),
      ),
    );

    // Simulate experiment with runs but no traces (ML)
    server.use(
      rest.get('/ajax-api/2.0/mlflow/traces', (req, res, ctx) => {
        return res(ctx.json({ traces: [] }));
      }),
      rest.post('/ajax-api/2.0/mlflow/runs/search', (req, res, ctx) => {
        return res(ctx.json({ runs: [{ info: { run_uuid: 'run1' } }] }));
      }),
    );

    // Start on experiment page WITHOUT a tab
    renderTestComponent('/experiments/12345678');

    // Should redirect to runs tab for ML experiments
    expect(await screen.findByText('Experiment runs page')).toBeInTheDocument();
  });

  test('integration test: should redirect based on existing kind tag without waiting for inference', async () => {
    server.resetHandlers(
      graphql.query('MlflowGetExperimentQuery', (req, res, ctx) =>
        res(
          ctx.data(
            createTestExperimentResponse(
              createTestExperiment('12345678', 'Test experiment name', [
                { key: 'mlflow.experimentKind', value: ExperimentKind.GENAI_DEVELOPMENT },
              ]),
            ),
          ),
        ),
      ),
      rest.get('/ajax-api/3.0/mlflow/server-info', (req, res, ctx) =>
        res(ctx.json({ store_type: 'SqlAlchemyStore', workspaces_enabled: false })),
      ),
    );

    // Start on experiment page WITHOUT a tab
    renderTestComponent('/experiments/12345678');

    // Should redirect to overview tab based on existing tag
    expect(await screen.findByText('Experiment overview page')).toBeInTheDocument();
  });

  describe('GenAI experiment page shell', () => {
    const genAIExperiment = createTestExperiment('12345678', 'Test experiment name', [
      { key: 'mlflow.experimentKind', value: ExperimentKind.GENAI_DEVELOPMENT },
    ]);

    beforeEach(() => {
      server.use(
        graphql.query('MlflowGetExperimentQuery', (req, res, ctx) =>
          res(ctx.data(createTestExperimentResponse(genAIExperiment))),
        ),
      );
    });

    test('flag off: keeps the experiment title header', async () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
      renderTestComponent('/experiments/12345678/evaluation-runs');

      expect(await screen.findByText('Experiment evaluation runs page')).toBeInTheDocument();
      expect(await screen.findByTestId('experiment-view-header-back-button')).toBeInTheDocument();
      expect(screen.queryByRole('navigation', { name: 'Breadcrumb' })).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: 'Copy link to this experiment' })).not.toBeInTheDocument();
    });

    test('flag on: shows the breadcrumb header with the run evaluation action on the evaluation runs tab', async () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
      renderTestComponent('/experiments/12345678/evaluation-runs');

      const breadcrumb = await screen.findByRole('navigation', { name: 'Breadcrumb' });
      expect(within(breadcrumb).getByRole('link', { name: 'Test experiment name' })).toBeInTheDocument();
      expect(within(breadcrumb).getByText('Evaluation runs')).toHaveAttribute('aria-current', 'page');
      expect(screen.getByRole('button', { name: 'Copy link to this experiment' })).toBeInTheDocument();
      expect(await screen.findByRole('button', { name: 'Run evaluation' })).toBeInTheDocument();
      expect(screen.getByRole('main')).toContainElement(screen.getByText('Experiment evaluation runs page'));
      expect(screen.queryByTestId('experiment-view-header-back-button')).not.toBeInTheDocument();
    });

    test('flag on: shows no page action outside the evaluation runs tab', async () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
      renderTestComponent('/experiments/12345678/traces');

      const breadcrumb = await screen.findByRole('navigation', { name: 'Breadcrumb' });
      expect(within(breadcrumb).getByText('Traces')).toHaveAttribute('aria-current', 'page');
      expect(screen.queryByRole('button', { name: 'Run evaluation' })).not.toBeInTheDocument();
    });

    test('flag on: keeps the experiment info, Edit / Delete menu and archival badge in the breadcrumb header', async () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
      server.use(
        graphql.query('MlflowGetExperimentQuery', (req, res, ctx) =>
          res(ctx.data(createTestExperimentResponse({ ...genAIExperiment, effectiveTraceArchivalRetention: '30d' }))),
        ),
      );
      renderTestComponent('/experiments/12345678/traces');

      const main = within(await screen.findByRole('main'));
      expect(await main.findByText('Archive after: 30 days')).toBeInTheDocument();
      await userEvent.click(main.getByRole('button', { name: 'Info' }));
      expect(await screen.findByTestId('experiment-view-header-info-tooltip-content')).toHaveTextContent(
        'Experiment ID: 12345678',
      );

      await userEvent.click(main.getByRole('button', { name: 'Open header dropdown menu' }));
      expect(await screen.findByRole('menuitem', { name: 'Edit experiment' })).toBeInTheDocument();
      expect(screen.getByRole('menuitem', { name: 'Delete' })).toBeInTheDocument();
      expect(screen.queryByTestId('experiment-view-header-back-button')).not.toBeInTheDocument();
    });

    test('flag on: Edit experiment from the evaluation runs breadcrumb header opens the experiment editor', async () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
      renderTestComponent('/experiments/12345678/evaluation-runs');

      const main = within(await screen.findByRole('main'));
      expect(await main.findByRole('button', { name: 'Run evaluation' })).toBeInTheDocument();
      expect(main.getByRole('button', { name: 'Info' })).toBeInTheDocument();
      await userEvent.click(main.getByRole('button', { name: 'Open header dropdown menu' }));
      await userEvent.click(await screen.findByRole('menuitem', { name: 'Edit experiment' }));
      expect(await screen.findByRole('dialog', { name: 'Edit experiment' })).toBeInTheDocument();
    });

    test('flag on: shows the experiment description under the breadcrumb row, not in it', async () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
      server.use(
        graphql.query('MlflowGetExperimentQuery', (req, res, ctx) =>
          res(
            ctx.data(
              createTestExperimentResponse({
                ...genAIExperiment,
                tags: [...genAIExperiment.tags, { key: NOTE_CONTENT_TAG, value: 'Nightly support agent evals' }],
              }),
            ),
          ),
        ),
      );
      renderTestComponent('/experiments/12345678/traces');

      const description = await screen.findByText('Nightly support agent evals');
      const breadcrumb = screen.getByRole('navigation', { name: 'Breadcrumb' });
      const breadcrumbRow = screen.getByRole('main').firstElementChild;
      expect(breadcrumbRow).toContainElement(breadcrumb);
      expect(breadcrumbRow).not.toContainElement(description);
      expect(screen.getByRole('main')).toContainElement(description);

      await userEvent.click(screen.getByTestId('experiment-metadata-editor-edit-button'));
      expect(await screen.findByRole('dialog', { name: 'Edit experiment' })).toBeInTheDocument();
    });

    test('flag on: shows no description area when the experiment has no description', async () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
      renderTestComponent('/experiments/12345678/traces');

      expect(await screen.findByRole('navigation', { name: 'Breadcrumb' })).toBeInTheDocument();
      expect(screen.queryByTestId('experiment-metadata-editor-edit-button')).not.toBeInTheDocument();
    });

    test('flag on: keeps the experiment title header for non-GenAI experiments', async () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
      server.use(
        graphql.query('MlflowGetExperimentQuery', (req, res, ctx) =>
          res(ctx.data(createTestExperimentResponse(createTestExperiment()))),
        ),
      );
      renderTestComponent('/experiments/12345678/runs');

      expect(await screen.findByTestId('experiment-view-header-back-button')).toBeInTheDocument();
      expect(screen.queryByRole('navigation', { name: 'Breadcrumb' })).not.toBeInTheDocument();
    });

    test('flag on: hides the breadcrumb row while a page hides the header', async () => {
      jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
      jest.mocked(ExperimentTracesPage).mockImplementation(FocusModeTracesPage);
      try {
        renderTestComponent('/experiments/12345678/traces');

        expect(await screen.findByRole('main')).toContainElement(await screen.findByText('Focused page'));
        expect(screen.queryByRole('navigation', { name: 'Breadcrumb' })).not.toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Copy link to this experiment' })).not.toBeInTheDocument();
      } finally {
        jest.mocked(ExperimentTracesPage).mockImplementation(() => <div>Experiment traces page</div>);
      }
    });

    describe('with workflow-based navigation (the OSS default)', () => {
      beforeEach(() => {
        jest.mocked(shouldEnableWorkflowBasedNavigation).mockReturnValue(true);
        jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
        localStorage.clear();
      });

      afterEach(() => {
        jest.mocked(shouldEnableWorkflowBasedNavigation).mockReturnValue(false);
      });

      test('shows the shell in GenAI mode', async () => {
        renderTestComponent('/experiments/12345678/traces', { withWorkflowType: true });

        const breadcrumb = await screen.findByRole('navigation', { name: 'Breadcrumb' });
        expect(within(breadcrumb).getByText('Traces')).toHaveAttribute('aria-current', 'page');
        expect(screen.getByRole('main')).toContainElement(screen.getByText('Experiment traces page'));
        expect(screen.queryByTestId('experiment-view-header-back-button')).not.toBeInTheDocument();
      });

      test('keeps the experiment title header in Model training mode', async () => {
        renderTestComponent('/experiments/12345678/runs?workflowType=machine_learning', { withWorkflowType: true });

        expect(await screen.findByText('Experiment runs page')).toBeInTheDocument();
        expect(await screen.findByTestId('experiment-observatory-link')).toBeInTheDocument();
        expect(screen.queryByRole('navigation', { name: 'Breadcrumb' })).not.toBeInTheDocument();
        expect(screen.queryByRole('main')).not.toBeInTheDocument();
      });
    });
  });
});
