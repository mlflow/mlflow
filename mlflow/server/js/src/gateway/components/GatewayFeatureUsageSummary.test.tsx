import { describe, it, expect } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { renderWithIntl } from '../../common/utils/TestUtils.react18';
import { setupServer } from '../../common/utils/setup-msw';
import { OverviewChartProvider } from '../../experiment-tracking/pages/experiment-overview/OverviewChartContext';
import { GatewayFeatureUsageSummary } from './GatewayFeatureUsageSummary';

describe('GatewayFeatureUsageSummary', () => {
  const server = setupServer();

  const respondWith = (valuesByMetric: Record<string, Record<string, number>>) => {
    server.use(
      rest.post(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), async (req, res, ctx) => {
        const body = await req.json();
        const aggregation = body.aggregations[0].aggregation_type;
        return res(
          ctx.json({
            data_points: Object.entries(valuesByMetric[body.metric_name] ?? {}).map(([caller, value]) => ({
              metric_name: body.metric_name,
              dimensions: { gateway_caller: caller },
              values: { [aggregation]: value },
            })),
          }),
        );
      }),
    );
  };

  const renderComponent = () =>
    renderWithIntl(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <DesignSystemProvider>
          <OverviewChartProvider experimentIds={['exp-1']} timeIntervalSeconds={3600} timeBuckets={[]}>
            <GatewayFeatureUsageSummary />
          </OverviewChartProvider>
        </DesignSystemProvider>
      </QueryClientProvider>,
    );

  const rowTexts = () =>
    screen
      .getAllByText(/^(Judges|Conversation simulator|Issue detection|Prompt optimization|Other)$/)
      .map((cell) => cell.textContent);

  it('lists usage for each MLflow feature, with other callers last', async () => {
    respondWith({
      trace_count: { judge: 12, issue_detection: 30, OpenAI: 50 },
      total_tokens: { judge: 1800, issue_detection: 4500, OpenAI: 7500 },
      total_cost: { judge: 0.5, issue_detection: 1.25, OpenAI: 2 },
    });

    renderComponent();

    expect(await screen.findByText('Usage by Feature')).toBeInTheDocument();
    // Sorted by requests, but "Other" stays last even though it has the most.
    expect(rowTexts()).toEqual(['Issue detection', 'Judges', 'Other']);

    expect(screen.getByText('30')).toBeInTheDocument();
    expect(screen.getByText('4.50K')).toBeInTheDocument();
    expect(screen.getByText('≈$1.25')).toBeInTheDocument();
    expect(screen.queryByText(/gateway:\/<endpoint-name>/)).not.toBeInTheDocument();
  });

  it('re-sorts features from a column header', async () => {
    respondWith({
      trace_count: { judge: 12, issue_detection: 30 },
      total_tokens: { judge: 1800, issue_detection: 4500 },
    });

    renderComponent();
    await screen.findByText('Usage by Feature');

    await userEvent.click(screen.getByText('Feature'));

    expect(rowTexts()).toEqual(['Judges', 'Issue detection']);
  });

  it('explains how to route MLflow features through the gateway when none have used it', async () => {
    respondWith({ trace_count: { OpenAI: 50 }, total_tokens: { OpenAI: 7500 } });

    renderComponent();

    expect(await screen.findByText('Other')).toBeInTheDocument();
    expect(screen.getByText(/as the model for judges, the conversation simulator/)).toBeInTheDocument();
  });
});
