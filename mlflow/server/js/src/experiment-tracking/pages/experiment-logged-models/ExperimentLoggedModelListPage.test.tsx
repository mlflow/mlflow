import { graphql, rest } from 'msw';
import { setupServer } from '../../../common/utils/setup-msw';
import { render, screen, waitFor, within } from '@testing-library/react';
import ExperimentLoggedModelListPage from './ExperimentLoggedModelListPage';
import { setupTestRouter, testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import { TestApolloProvider } from '../../../common/utils/TestApolloProvider';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import userEvent from '@testing-library/user-event';
import { LoggedModelStatusProtoEnum } from '../../types';
import { first, orderBy } from 'lodash';
import { RunsChartsBarCardConfig } from '../../components/runs-charts/runs-charts.types';
import { RunsChartsRunData } from '../../components/runs-charts/components/RunsCharts.common';
import { createMLflowRoutePath } from '../../../common/utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

jest.setTimeout(90000); // increase timeout due to testing heavier tables and charts

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  isExperimentLoggedModelsUIEnabled: jest.fn(() => true),
}));

// Mock the chart component to save some resources while easily assert that the correct chart is rendered
jest.mock('../../components/runs-charts/components/RunsChartsDraggableCard', () => ({
  RunsChartsDraggableCard: ({
    cardConfig,
    chartRunData,
  }: {
    cardConfig: RunsChartsBarCardConfig;
    chartRunData: RunsChartsRunData[];
  }) => (
    <div data-testid="test-chart">
      <span>this is chart for {cardConfig.metricKey}</span>
      <span>
        displaying data for models:{' '}
        {chartRunData
          .filter(({ hidden }) => !hidden)
          .map((dataTrace) => dataTrace.uuid)
          .join(',')}
      </span>
    </div>
  ),
}));

describe('ExperimentLoggedModelListPage', () => {
  const { history } = setupTestRouter();

  // Simulate API returning logged models in particular order, configured by "ascending" flag
  const createTestLoggedModelsResponse = (
    experiment_id = 'test-experiment',
    { ascending = false, pageToken }: { ascending?: boolean; pageToken?: string } = {},
  ) => {
    const allResults = 6;
    const pageSize = 3;
    const page = Number(pageToken) || 1;

    const allModels = Array.from({ length: allResults }, (_, i) => ({
      info: {
        experiment_id,
        model_id: `m-${i + 1}`,
        model_type: 'Agent',
        name: `Test model ${i + 1}`,
        registrations: [],
        source_run_id: 'test-run',
        status: LoggedModelStatusProtoEnum.LOGGED_MODEL_PENDING,
        status_message: 'Pending',
        tags: [],
        creation_timestamp_ms: i * 1000,
      },
      data: {
        metrics: [
          {
            key: 'common-metric',
            timestamp: i,
            step: i,
            value: i,
          },
          {
            key: 'test-metric-for-model-' + (i + 1),
            timestamp: i,
            step: i,
            value: i,
          },
        ],
      },
    }));

    const next_page_token = page * pageSize < allResults ? (page + 1).toString() : null;

    const orderedModels = orderBy(allModels, [({ info }) => info.creation_timestamp_ms], [ascending ? 'asc' : 'desc']);

    return {
      models: orderedModels.slice((page - 1) * pageSize, page * pageSize),
      next_page_token,
    };
  };

  const server = setupServer(
    rest.get('/ajax-api/2.0/mlflow/runs/get', (req, res, ctx) => {
      return res(ctx.json({ run: { info: { run_uuid: 'test-run' } } }));
    }),
    rest.post<any>('/ajax-api/2.0/mlflow/logged-models/search', (req, res, ctx) => {
      const responsePayload = createTestLoggedModelsResponse('test-experiment', {
        ascending: (req?.body as any)?.order_by?.[0]?.ascending,
        pageToken: (req?.body as any)?.page_token,
      });

      return res(ctx.json(responsePayload));
    }),
  );

  const renderTestComponent = () => {
    const queryClient = new QueryClient();
    return render(
      <TestApolloProvider disableCache>
        <QueryClientProvider client={queryClient}>
          <MockedReduxStoreProvider state={{ entities: { experimentTagsByExperimentId: {} } }}>
            <IntlProvider locale="en">
              <DesignSystemProvider>
                <TestRouter
                  routes={[testRoute(<ExperimentLoggedModelListPage />, '/experiments/:experimentId')]}
                  history={history}
                  initialEntries={['/experiments/test-experiment']}
                />
              </DesignSystemProvider>
            </IntlProvider>
          </MockedReduxStoreProvider>
        </QueryClientProvider>
      </TestApolloProvider>,
    );
  };

  beforeAll(() => {
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
    server.listen();
  });

  beforeEach(() => {
    server.resetHandlers();
  });

  test('should display experiment title and test logged models when fetched', async () => {
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByText('Test model 6')).toBeInTheDocument();
      expect(screen.getByText('Test model 5')).toBeInTheDocument();
      expect(screen.getByText('Test model 4')).toBeInTheDocument();
    });
  });

  test('should change sort order for runs', async () => {
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByText('Test model 6')).toBeInTheDocument();
    });

    expect(screen.getAllByRole('link', { name: /Test model \d/ }).map((cell) => cell.textContent)).toEqual([
      'Test model 6',
      'Test model 5',
      'Test model 4',
    ]);

    await userEvent.click(screen.getByRole('button', { name: 'Sort: Created' }));

    await waitFor(() => {
      expect(screen.getAllByRole('link', { name: /Test model \d/ }).map((cell) => cell.textContent)).toEqual([
        'Test model 1',
        'Test model 2',
        'Test model 3',
      ]);
    });
  });

  test('should load more runs', async () => {
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByText('Test model 6')).toBeInTheDocument();
    });

    expect(screen.getAllByRole('link', { name: /Test model \d/ }).map((cell) => cell.textContent)).toEqual([
      'Test model 6',
      'Test model 5',
      'Test model 4',
    ]);

    await userEvent.click(screen.getByRole('button', { name: 'Load more' }));

    await waitFor(() => {
      expect(screen.getAllByRole('link', { name: /Test model \d/ }).map((cell) => cell.textContent)).toEqual([
        'Test model 6',
        'Test model 5',
        'Test model 4',
        'Test model 3',
        'Test model 2',
        'Test model 1',
      ]);
    });

    expect(screen.queryByRole('button', { name: 'Load more' })).not.toBeInTheDocument();
  });

  test('should fetch and display run names', async () => {
    server.use(
      rest.get('/ajax-api/2.0/mlflow/runs/get', (req, res, ctx) => {
        return res(ctx.json({ run: { info: { run_uuid: 'test-run', run_name: 'Test run name' } } }));
      }),
    );

    renderTestComponent();

    // We should see run names as links
    await waitFor(() => {
      expect(screen.getAllByRole('link', { name: 'Test run name' }).length).toBeGreaterThan(0);
    });

    // The link should point to the run page
    expect(first(screen.getAllByRole('link', { name: 'Test run name' }))).toHaveAttribute(
      'href',
      createMLflowRoutePath('/experiments/test-experiment/runs/test-run'),
    );
  });

  describe('ExperimentLoggedModelListPage: charts', () => {
    test('should display charts and filter them by metric name', async () => {
      renderTestComponent();

      await waitFor(() => {
        expect(screen.getByRole('radio', { name: 'Chart view' })).toBeInTheDocument();
      });

      await userEvent.click(screen.getByRole('radio', { name: 'Chart view' }));

      await waitFor(() => {
        expect(screen.getAllByTestId('test-chart').length).toBe(4);
      });

      expect(screen.getByText('this is chart for common-metric')).toBeInTheDocument();
      expect(screen.getByText('this is chart for test-metric-for-model-6')).toBeInTheDocument();
      expect(screen.getByText('this is chart for test-metric-for-model-5')).toBeInTheDocument();
      expect(screen.getByText('this is chart for test-metric-for-model-4')).toBeInTheDocument();

      await userEvent.type(screen.getByPlaceholderText('Search metric charts'), 'common-metric');

      await waitFor(() => {
        expect(screen.getAllByTestId('test-chart').length).toBe(1);
      });

      expect(screen.getByText('this is chart for common-metric')).toBeInTheDocument();
    });

    test('should reflect visibility of logged models on charts', async () => {
      renderTestComponent();

      await waitFor(() => {
        expect(screen.getByRole('radio', { name: 'Chart view' })).toBeInTheDocument();
      });

      await userEvent.click(screen.getByRole('radio', { name: 'Chart view' }));

      await waitFor(() => {
        expect(screen.getAllByText('displaying data for models: m-6,m-5,m-4').length).toBeGreaterThan(0);
      });

      await waitFor(() => {
        expect(screen.getByText('Test model 5')).toBeInTheDocument();
      });

      // Click visibility button for model 5
      await userEvent.click(
        within(screen.getByText('Test model 5').closest('[role=row]') as HTMLElement).getByRole('button'),
      );

      await waitFor(() => {
        expect(screen.getAllByText('displaying data for models: m-6,m-4').length).toBeGreaterThan(0);
      });

      // Hide all runs
      await userEvent.click(screen.getByLabelText('Toggle visibility of rows'));
      await userEvent.click(screen.getByText('Hide all runs'));

      // Click visibility button for model 4
      await userEvent.click(
        within(screen.getByText('Test model 4').closest('[role=row]') as HTMLElement).getByRole('button'),
      );

      await waitFor(() => {
        expect(screen.getAllByText('displaying data for models: m-4').length).toBeGreaterThan(0);
      });
    });
  });
});
