import { graphql, rest } from 'msw';
import { setupServer } from '../../../common/utils/setup-msw';
import { render, screen, waitFor, within, act } from '@testing-library/react';
import ExperimentLoggedModelListPage from './ExperimentLoggedModelListPage';
import { setupTestRouter, testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import { TestApolloProvider } from '../../../common/utils/TestApolloProvider';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import userEvent from '@testing-library/user-event';
import { LoggedModelStatusProtoEnum } from '../../types';
import { first, orderBy } from 'lodash';
import type { RunsChartsBarCardConfig } from '../../components/runs-charts/runs-charts.types';
import type { RunsChartsRunData } from '../../components/runs-charts/components/RunsCharts.common';
import { createMLflowRoutePath } from '../../../common/utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(90000); // increase timeout due to testing heavier tables and charts

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  isLoggedModelsFilteringAndSortingEnabled: jest.fn(() => true),
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
    {
      ascending = false,
      pageToken,
      filter,
    }: {
      ascending?: boolean;
      pageToken?: string;
      // Filter string, the only supported one is "metrics.common-metric" one
      filter?: string;
    } = {},
  ) => {
    const allResults = 6;
    const pageSize = 3;
    const page = Number(pageToken) || 1;

    const modelsWithRegisteredVersions = ['m-6', 'm-4'];
    // Parse filter string to extract the value of "metrics.common-metric"
    const parsedFilterMetric = filter ? filter.match(/metrics\.common-metric = (.+)/)?.[1] : undefined;

    if (filter && !parsedFilterMetric) {
      throw new Error('Parse error: invalid filter string');
    }

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
        tags: modelsWithRegisteredVersions.includes(`m-${i + 1}`)
          ? [
              {
                key: 'mlflow.modelVersions',
                value: JSON.stringify([
                  {
                    name: 'registered-model-name-' + (i + 1),
                    version: 1,
                  },
                ]),
              },
            ]
          : [],
        creation_timestamp_ms: i * 1000,
      },
      data: {
        metrics: [
          {
            key: 'common-metric',
            timestamp: i,
            step: i,
            value: i,
            dataset_digest: undefined as string | undefined,
            dataset_name: undefined as string | undefined,
          },
          {
            key: 'test-metric-for-model-' + (i + 1),
            timestamp: i,
            step: i,
            value: i,
            dataset_digest: undefined as string | undefined,
            dataset_name: undefined as string | undefined,
          },
        ],
      },
    })).filter(
      (model) =>
        !parsedFilterMetric ||
        model.data.metrics.some(
          (metric) => metric.key === 'common-metric' && metric.value === Number(parsedFilterMetric),
        ),
    );

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
      try {
        const responsePayload = createTestLoggedModelsResponse('test-experiment', {
          ascending: (req?.body as any)?.order_by?.[0]?.ascending,
          pageToken: (req?.body as any)?.page_token,
          filter: (req?.body as any)?.filter,
        });

        return res(ctx.json(responsePayload));
      } catch (error: any) {
        return res(ctx.status(400), ctx.json({ message: error.message, error_code: 400 }));
      }
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

  test('should display registered model versions', async () => {
    renderTestComponent();

    // Wait for the data to be loaded
    await waitFor(() => {
      expect(screen.getByText('Test model 6')).toBeInTheDocument();
    });

    // Expect model 6 to have a link to the registered model version
    expect(screen.getByRole('link', { name: /registered-model-name-6 v1/ })).toHaveAttribute(
      'href',
      createMLflowRoutePath('/models/registered-model-name-6/versions/1'),
    );

    // Expect model 5 to not have any registered model version links
    expect(screen.queryByRole('link', { name: /registered-model-name-5/ })).not.toBeInTheDocument();
  });

  test('should use search query filter and display filtered results', async () => {
    renderTestComponent();

    // We should see run names as links
    await waitFor(() => {
      expect(screen.getByPlaceholderText('metrics.rmse >= 0.8')).toBeInTheDocument();
    });

    await userEvent.type(screen.getByPlaceholderText('metrics.rmse >= 0.8'), 'metrics.common-metric = 5{enter}');

    await waitFor(() => {
      expect(screen.getByText('Test model 6')).toBeInTheDocument();
    });

    expect(screen.queryByText('Test model 5')).not.toBeInTheDocument();
    expect(screen.queryByText('Test model 4')).not.toBeInTheDocument();
    expect(screen.queryByText('Test model 3')).not.toBeInTheDocument();
    expect(screen.queryByText('Test model 2')).not.toBeInTheDocument();
    expect(screen.queryByText('Test model 1')).not.toBeInTheDocument();
  });

  test('should use search query filter and display empty results with relevant message', async () => {
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByPlaceholderText('metrics.rmse >= 0.8')).toBeInTheDocument();
    });

    await userEvent.type(screen.getByPlaceholderText('metrics.rmse >= 0.8'), 'metrics.common-metric = 55{enter}');

    await waitFor(() => {
      expect(screen.getByText('No models found')).toBeInTheDocument();
    });
  });

  test('should use malformed search query filter and display error message', async () => {
    jest.spyOn(console, 'error').mockImplementation(() => {});
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByPlaceholderText('metrics.rmse >= 0.8')).toBeInTheDocument();
    });

    await userEvent.type(screen.getByPlaceholderText('metrics.rmse >= 0.8'), 'metr.malformed_query="{enter}');

    await waitFor(() => {
      expect(screen.getByText('Parse error: invalid filter string')).toBeInTheDocument();
    });

    jest.restoreAllMocks();
  });

  test('should allow filtering by datasets', async () => {
    const requestSpy = jest.fn();
    server.use(
      rest.post<any>('/ajax-api/2.0/mlflow/logged-models/search', (req, res, ctx) => {
        requestSpy(req.body);
        const responsePayload = createTestLoggedModelsResponse('test-experiment');
        const firstModelResult = responsePayload.models[0];
        const secondModelResult = responsePayload.models[1];

        for (const metric of firstModelResult.data.metrics) {
          metric.dataset_digest = '123456';
          metric.dataset_name = 'train_dataset';
        }
        for (const metric of secondModelResult.data.metrics) {
          metric.dataset_digest = '987654';
          metric.dataset_name = 'test_dataset';
        }

        return res(ctx.json(responsePayload));
      }),
    );

    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByText('Test model 6')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole('button', { name: 'Datasets' }));

    await waitFor(() => {
      expect(screen.getByRole('option', { name: 'train_dataset (#123456)' })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'test_dataset (#987654)' })).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole('option', { name: 'train_dataset (#123456)' }));

    await waitFor(() => {
      expect(requestSpy).toHaveBeenLastCalledWith(
        expect.objectContaining({
          datasets: [
            {
              dataset_digest: '123456',
              dataset_name: 'train_dataset',
            },
          ],
        }),
      );
    });
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
