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

  // Queries grouped by caller get a data point per caller. Ungrouped totals default to the sum over
  // the callers; `totals` overrides them to include traces that have no caller.
  const respondWith = (valuesByMetric: Record<string, ValuesByCaller>, totals: Record<string, number> = {}) => {
    server.use(
      rest.post(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), async (req, res, ctx) => {
        const body = await req.json();
        capturedRequests.push(body);
        const aggregation = body.aggregations[0].aggregation_type;
        const values: ValuesByCaller = valuesByMetric[body.metric_name] ?? {};
        if (!body.dimensions?.includes('gateway_caller')) {
          const total = totals[body.metric_name] ?? Object.values(values).reduce((sum, value) => sum + value, 0);
          return res(
            ctx.json({
              data_points: [{ metric_name: body.metric_name, dimensions: {}, values: { [aggregation]: total } }],
            }),
          );
        }
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

  it('counts gateway traces without a caller as other usage', async () => {
    // Grouping by caller leaves out traces without one, such as requests sent with neither a caller
    // header nor a User-Agent, but the totals the other charts show include them.
    respondWith(
      {
        trace_count: { judge: 2, OpenAI: 1 },
        total_tokens: { judge: 300, OpenAI: 150 },
        total_cost: { judge: 0.5, OpenAI: 0.25 },
      },
      { trace_count: 5, total_tokens: 750, total_cost: 1.25 },
    );

    const { result } = renderHook(() => useGatewayFeatureUsageData(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.usage).toEqual([
      { feature: 'judge', requests: 2, tokens: 300, cost: 0.5 },
      { feature: OTHER_CALLERS, requests: 3, tokens: 450, cost: 0.75 },
    ]);
  });

  it('leaves out other usage when MLflow features made every request', async () => {
    respondWith({ trace_count: { judge: 2 }, total_tokens: { judge: 300 }, total_cost: { judge: 0.1 } });

    const { result } = renderHook(() => useGatewayFeatureUsageData(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.usage).toEqual([{ feature: 'judge', requests: 2, tokens: 300, cost: 0.1 }]);
  });

  it('queries usage by gateway caller and the totals within the chart filters', async () => {
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
        {
          view_type: 'TRACES',
          metric_name: 'trace_count',
          dimensions: ['trace_status'],
          filters: ['trace.status = "OK"'],
        },
        { view_type: 'TRACES', metric_name: 'total_tokens', dimensions: undefined, filters: ['trace.status = "OK"'] },
        { view_type: 'SPANS', metric_name: 'total_cost', dimensions: undefined, filters: ['trace.status = "OK"'] },
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
