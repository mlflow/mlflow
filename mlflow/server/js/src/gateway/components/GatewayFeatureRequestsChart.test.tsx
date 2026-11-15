import { describe, it, expect } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEventGlobal, { PointerEventsCheckLevel } from '@testing-library/user-event';
import { rest } from 'msw';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { renderWithIntl } from '../../common/utils/TestUtils.react18';
import { setupServer } from '../../common/utils/setup-msw';
import { OverviewChartProvider } from '../../experiment-tracking/pages/experiment-overview/OverviewChartContext';
import { GatewayFeatureRequestsChart } from './GatewayFeatureRequestsChart';

const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

const BUCKET = '2026-11-13T10:00:00Z';

describe('GatewayFeatureRequestsChart', () => {
  const server = setupServer();

  const respondWith = (byCaller: Record<string, number>, total: number) => {
    server.use(
      rest.post(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), async (req, res, ctx) => {
        const body = await req.json();
        const dataPoints = body.dimensions.includes('gateway_caller')
          ? Object.entries(byCaller).map(([caller, count]) => ({
              metric_name: 'trace_count',
              dimensions: { time_bucket: BUCKET, gateway_caller: caller },
              values: { COUNT: count },
            }))
          : [{ metric_name: 'trace_count', dimensions: { time_bucket: BUCKET }, values: { COUNT: total } }];
        return res(ctx.json({ data_points: dataPoints }));
      }),
    );
  };

  const renderComponent = () =>
    renderWithIntl(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <DesignSystemProvider>
          <OverviewChartProvider
            experimentIds={['exp-1']}
            startTimeMs={new Date(BUCKET).getTime()}
            endTimeMs={new Date(BUCKET).getTime() + 3600 * 1000}
            timeIntervalSeconds={3600}
            timeBuckets={[new Date(BUCKET).getTime()]}
          >
            <GatewayFeatureRequestsChart />
          </OverviewChartProvider>
        </DesignSystemProvider>
      </QueryClientProvider>,
    );

  it('stacks a bar for each MLflow feature that made requests, and one for other requests', async () => {
    respondWith({ judge: 3, issue_detection: 2, OpenAI: 1 }, 7);

    renderComponent();

    expect(await screen.findByText('Requests by Feature')).toBeInTheDocument();
    expect(screen.getByTestId('bar-Judges')).toBeInTheDocument();
    expect(screen.getByTestId('bar-Issue detection')).toBeInTheDocument();
    expect(screen.getByTestId('bar-Other')).toBeInTheDocument();
    expect(screen.queryByTestId('bar-Conversation simulator')).not.toBeInTheDocument();
    expect(screen.getByText('All features')).toBeInTheDocument();
  });

  it('hides the features deselected in the selector', async () => {
    respondWith({ judge: 3, issue_detection: 2 }, 5);

    renderComponent();

    await userEvent.click(await screen.findByTestId('gateway-feature-selector'));
    await userEvent.click(screen.getByRole('checkbox', { name: 'Judges' }));

    await waitFor(() => expect(screen.queryByTestId('bar-Judges')).not.toBeInTheDocument());
    expect(screen.getByTestId('bar-Issue detection')).toBeInTheDocument();
    expect(screen.getByText('1 selected')).toBeInTheDocument();
  });

  it('shows the empty state when the gateway has no requests', async () => {
    respondWith({}, 0);

    renderComponent();

    expect(await screen.findByText('No data available for the selected time range')).toBeInTheDocument();
    expect(screen.queryByTestId('gateway-feature-selector')).not.toBeInTheDocument();
  });
});
