import { describe, it, expect, beforeEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { rest } from 'msw';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { setupServer } from '../../common/utils/setup-msw';
import { OverviewChartProvider } from '../../experiment-tracking/pages/experiment-overview/OverviewChartContext';
import { OTHER_CALLERS } from '../utils/gatewayFeatures';
import { useGatewayFeatureRequestsChartData } from './useGatewayFeatureRequestsChartData';

const FIRST_BUCKET = '2026-11-13T10:00:00Z';
const SECOND_BUCKET = '2026-11-13T11:00:00Z';

type CountsByBucket = Record<string, Record<string, number>>;

describe('useGatewayFeatureRequestsChartData', () => {
  const server = setupServer();
  let capturedRequests: any[] = [];

  // Requests per caller in each bucket, and each bucket's total, which can include traces that have no caller.
  const respondWith = (byCaller: CountsByBucket, totals: Record<string, number>) => {
    server.use(
      rest.post(getAjaxUrl('ajax-api/3.0/mlflow/traces/metrics'), async (req, res, ctx) => {
        const body = await req.json();
        capturedRequests.push(body);
        const dataPoints = body.dimensions.includes('gateway_caller')
          ? Object.entries(byCaller).flatMap(([bucket, counts]) =>
              Object.entries(counts).map(([caller, count]) => ({
                metric_name: 'trace_count',
                dimensions: { time_bucket: bucket, gateway_caller: caller },
                values: { COUNT: count },
              })),
            )
          : Object.entries(totals).map(([bucket, count]) => ({
              metric_name: 'trace_count',
              dimensions: { time_bucket: bucket, trace_status: 'OK' },
              values: { COUNT: count },
            }));
        return res(ctx.json({ data_points: dataPoints }));
      }),
    );
  };

  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <OverviewChartProvider
        experimentIds={['exp-1']}
        startTimeMs={new Date(FIRST_BUCKET).getTime()}
        endTimeMs={new Date(SECOND_BUCKET).getTime()}
        timeIntervalSeconds={3600}
        timeBuckets={[new Date(FIRST_BUCKET).getTime(), new Date(SECOND_BUCKET).getTime()]}
        filters={['trace.status = "OK"']}
      >
        {children}
      </OverviewChartProvider>
    </QueryClientProvider>
  );

  beforeEach(() => {
    capturedRequests = [];
  });

  it('breaks each time bucket down by MLflow feature, with the rest of its requests as other', async () => {
    respondWith(
      {
        [FIRST_BUCKET]: { judge: 2, simulator: 1, OpenAI: 1 },
        [SECOND_BUCKET]: { judge: 1 },
      },
      // The first bucket has 2 traces without a caller on top of the 4 grouped above.
      { [FIRST_BUCKET]: 6, [SECOND_BUCKET]: 1 },
    );

    const { result } = renderHook(() => useGatewayFeatureRequestsChartData(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.series).toEqual(['judge', 'simulator', OTHER_CALLERS]);
    expect(result.current.chartData.map(({ timestampMs, values }) => ({ timestampMs, values }))).toEqual([
      {
        timestampMs: new Date(FIRST_BUCKET).getTime(),
        values: { judge: 2, simulator: 1, issue_detection: 0, prompt_optimization: 0, [OTHER_CALLERS]: 3 },
      },
      {
        timestampMs: new Date(SECOND_BUCKET).getTime(),
        values: { judge: 1, simulator: 0, issue_detection: 0, prompt_optimization: 0, [OTHER_CALLERS]: 0 },
      },
    ]);
    expect(result.current.hasData).toBe(true);
  });

  it('queries requests per caller and per bucket within the chart filters', async () => {
    respondWith({}, {});

    const { result } = renderHook(() => useGatewayFeatureRequestsChartData(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(
      capturedRequests.map(({ metric_name, dimensions, time_interval_seconds, filters }) => ({
        metric_name,
        dimensions,
        time_interval_seconds,
        filters,
      })),
    ).toEqual(
      expect.arrayContaining([
        {
          metric_name: 'trace_count',
          dimensions: ['gateway_caller'],
          time_interval_seconds: 3600,
          filters: ['trace.status = "OK"'],
        },
        {
          metric_name: 'trace_count',
          dimensions: ['trace_status'],
          time_interval_seconds: 3600,
          filters: ['trace.status = "OK"'],
        },
      ]),
    );
    expect(result.current.series).toEqual([]);
    expect(result.current.hasData).toBe(false);
  });
});
