import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl } from '../../../../common/utils/TestUtils.react18';
import { TopUsersTable } from './TopUsersTable';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import {
  AggregationType,
  SpanMetricKey,
  TraceDimensionKey,
  TraceMetricKey,
} from '@databricks/web-shared/model-trace-explorer';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';
import { OverviewChartProvider } from '../OverviewChartContext';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';

const userDataPoint = (metricName: string, user: string, values: Record<string, number>, extra = {}) => ({
  metric_name: metricName,
  dimensions: { [TraceDimensionKey.TRACE_USER]: user, ...extra },
  values,
});

const DATA_POINTS_BY_METRIC = {
  [TraceMetricKey.TRACE_COUNT]: [
    userDataPoint(TraceMetricKey.TRACE_COUNT, 'alice', { [AggregationType.COUNT]: 3 }, { trace_status: 'OK' }),
    userDataPoint(TraceMetricKey.TRACE_COUNT, 'alice', { [AggregationType.COUNT]: 1 }, { trace_status: 'ERROR' }),
    userDataPoint(TraceMetricKey.TRACE_COUNT, 'bob', { [AggregationType.COUNT]: 2 }, { trace_status: 'OK' }),
  ],
  [TraceMetricKey.SESSION_COUNT]: [
    userDataPoint(TraceMetricKey.SESSION_COUNT, 'alice', { [AggregationType.COUNT]: 2 }),
    userDataPoint(TraceMetricKey.SESSION_COUNT, 'bob', { [AggregationType.COUNT]: 1 }),
  ],
  [TraceMetricKey.TOTAL_TOKENS]: [
    userDataPoint(TraceMetricKey.TOTAL_TOKENS, 'alice', { [AggregationType.SUM]: 1500 }),
    userDataPoint(TraceMetricKey.TOTAL_TOKENS, 'bob', { [AggregationType.SUM]: 700 }),
  ],
  [TraceMetricKey.LATENCY]: [
    userDataPoint(TraceMetricKey.LATENCY, 'alice', { [AggregationType.AVG]: 250 }),
    userDataPoint(TraceMetricKey.LATENCY, 'bob', { [AggregationType.AVG]: 100 }),
  ],
  [SpanMetricKey.TOTAL_COST]: [
    userDataPoint(SpanMetricKey.TOTAL_COST, 'alice', { [AggregationType.SUM]: 1.25 }),
    userDataPoint(SpanMetricKey.TOTAL_COST, 'bob', { [AggregationType.SUM]: 0.5 }),
  ],
} satisfies Record<string, any[]>;

describe('TopUsersTable', () => {
  const startTimeMs = new Date('2025-12-22T10:00:00Z').getTime();
  const endTimeMs = new Date('2025-12-22T12:00:00Z').getTime();

  const contextProps = {
    experimentIds: ['test-experiment-123'],
    startTimeMs,
    endTimeMs,
    timeIntervalSeconds: 3600,
    timeBuckets: [startTimeMs, startTimeMs + 3600000, endTimeMs],
  };

  const server = setupServer();
  let requestBodies: any[] = [];

  const renderComponent = (props: React.ComponentProps<typeof TopUsersTable> = {}) => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return renderWithIntl(
      <QueryClientProvider client={queryClient}>
        <DesignSystemProvider>
          <OverviewChartProvider {...contextProps}>
            <TopUsersTable {...props} />
          </OverviewChartProvider>
        </DesignSystemProvider>
      </QueryClientProvider>,
    );
  };

  const setupTraceMetricsHandler = (dataPointsByMetric: Record<string, any[]>) => {
    server.use(
      rest.post(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), async (req, res, ctx) => {
        const body = await req.json();
        requestBodies.push(body);
        const metricName: string = body.metric_name ?? body.metric_names?.[0];
        return res(ctx.json({ data_points: dataPointsByMetric[metricName] ?? [] }));
      }),
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    requestBodies = [];
    setupTraceMetricsHandler(DATA_POINTS_BY_METRIC);
  });

  it('groups every metric query by the trace_user dimension', async () => {
    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('Top Users')).toBeInTheDocument();
    });

    expect(requestBodies).toHaveLength(5);
    for (const body of requestBodies) {
      expect(body.dimensions).toContain(TraceDimensionKey.TRACE_USER);
    }
  });

  it('renders one row per user with combined metrics', async () => {
    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('alice')).toBeInTheDocument();
    });

    expect(screen.getByText('bob')).toBeInTheDocument();
    expect(screen.getByText('$1.25')).toBeInTheDocument();
    expect(screen.getByText('$0.50')).toBeInTheDocument();
    // alice: 1 error out of 4 traces
    expect(screen.getByText('25.00%')).toBeInTheDocument();
    expect(screen.getByText('0.00%')).toBeInTheDocument();
  });

  it('sorts users by trace count descending by default', async () => {
    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('alice')).toBeInTheDocument();
    });

    const names = screen.getAllByText(/^(alice|bob)$/).map((element) => element.textContent);
    expect(names).toEqual(['alice', 'bob']);
  });

  it('calls onSelectUser when a user name is clicked', async () => {
    const onSelectUser = jest.fn();
    renderComponent({ onSelectUser });

    await waitFor(() => {
      expect(screen.getByText('bob')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('bob'));

    expect(onSelectUser).toHaveBeenCalledWith('bob');
  });

  it('renders the empty state when no trace has a user', async () => {
    setupTraceMetricsHandler({});
    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('No data available for the selected time range')).toBeInTheDocument();
    });
  });

  it('renders the error state when the API call fails', async () => {
    server.use(
      rest.post(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), (_req, res, ctx) =>
        res(ctx.status(500), ctx.json({ error: 'API Error' })),
      ),
    );
    renderComponent();

    await waitFor(() => {
      expect(screen.getByText('Failed to load chart data')).toBeInTheDocument();
    });
  });
});
