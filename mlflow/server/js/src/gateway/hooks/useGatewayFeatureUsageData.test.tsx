import { describe, it, expect, beforeEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { rest } from 'msw';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { setupServer } from '../../common/utils/setup-msw';
import { OverviewChartProvider } from '../../experiment-tracking/pages/experiment-overview/OverviewChartContext';
import { OTHER_CALLERS, useGatewayFeatureUsageData } from './useGatewayFeatureUsageData';

type ValuesByCaller = Record<string, number>;

describe('useGatewayFeatureUsageData', () => {
  const server = setupServer();
  let capturedRequests: any[] = [];

  const respondWith = (valuesByMetric: Record<string, ValuesByCaller>) => {
    server.use(
      rest.post(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), async (req, res, ctx) => {
        const body = await req.json();
        capturedRequests.push(body);
        const aggregation = body.aggregations[0].aggregation_type;
        const values: ValuesByCaller = valuesByMetric[body.metric_name] ?? {};
        return res(
          ctx.json({
            data_points: Object.entries(values).map(([caller, value]) => ({
              metric_name: body.metric_name,
              dimensions: { gateway_caller: caller },
              values: { [aggregation]: value },
            })),
          }),
        );
      }),
    );
  };

  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <OverviewChartProvider
        experimentIds={['exp-1']}
        startTimeMs={1000}
        endTimeMs={2000}
        timeIntervalSeconds={3600}
        timeBuckets={[]}
        filters={['trace.status = "OK"']}
      >
        {children}
      </OverviewChartProvider>
    </QueryClientProvider>
  );

  beforeEach(() => {
    capturedRequests = [];
  });

  it('breaks usage down by MLflow feature and combines every other caller', async () => {
    respondWith({
      trace_count: { simulator: 4, judge: 12, OpenAI: 3, Mozilla: 2 },
      total_tokens: { simulator: 600, judge: 1800, OpenAI: 450, Mozilla: 300 },
      total_cost: { simulator: 0.125, judge: 0.5, OpenAI: 0.25, Mozilla: 0.5 },
    });

    const { result } = renderHook(() => useGatewayFeatureUsageData(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.usage).toEqual([
      { feature: 'judge', requests: 12, tokens: 1800, cost: 0.5 },
      { feature: 'simulator', requests: 4, tokens: 600, cost: 0.125 },
      { feature: OTHER_CALLERS, requests: 5, tokens: 750, cost: 0.75 },
    ]);
    expect(result.current.hasFeatureUsage).toBe(true);
  });

  it('groups requests, tokens and cost by gateway caller within the chart filters', async () => {
    respondWith({});

    const { result } = renderHook(() => useGatewayFeatureUsageData(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(
      capturedRequests.map(({ view_type, metric_name, dimensions, filters }) => ({
        view_type,
        metric_name,
        dimensions,
        filters,
      })),
    ).toEqual(
      expect.arrayContaining([
        {
          view_type: 'TRACES',
          metric_name: 'trace_count',
          dimensions: ['gateway_caller'],
          filters: ['trace.status = "OK"'],
        },
        {
          view_type: 'TRACES',
          metric_name: 'total_tokens',
          dimensions: ['gateway_caller'],
          filters: ['trace.status = "OK"'],
        },
        {
          view_type: 'SPANS',
          metric_name: 'total_cost',
          dimensions: ['gateway_caller'],
          filters: ['trace.status = "OK"'],
        },
      ]),
    );
    expect(result.current.usage).toEqual([]);
  });

  it('reports no feature usage when only other callers used the gateway', async () => {
    respondWith({ trace_count: { OpenAI: 3 }, total_tokens: { OpenAI: 450 } });

    const { result } = renderHook(() => useGatewayFeatureUsageData(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.usage).toEqual([{ feature: OTHER_CALLERS, requests: 3, tokens: 450, cost: 0 }]);
    expect(result.current.hasFeatureUsage).toBe(false);
  });
});
