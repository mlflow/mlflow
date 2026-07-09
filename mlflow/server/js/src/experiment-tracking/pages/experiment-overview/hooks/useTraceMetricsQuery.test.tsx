import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useTraceMetricsQuery } from './useTraceMetricsQuery';
import { MetricViewType, AggregationType, TraceMetricKey } from '@databricks/web-shared/model-trace-explorer';
import type { ReactNode } from 'react';
import { setupServer } from '../../../../common/utils/setup-msw';
import { rest } from 'msw';

const mockShouldUseTracesV4API = jest.fn<() => boolean>();
jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  shouldUseTracesV4API: () => mockShouldUseTracesV4API(),
}));

const mockShouldEnableBatchedTokenMetricQueries = jest.fn<() => boolean>();
jest.mock('../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<Record<string, unknown>>('../../../../common/utils/FeatureUtils'),
  shouldEnableBatchedTokenMetricQueries: () => mockShouldEnableBatchedTokenMetricQueries(),
}));

const mockWarehouseId = jest.fn<() => string | undefined | null>();
jest.mock('../../experiment-page-tabs/SqlWarehouseContext', () => ({
  useSqlWarehouseContextSafe: () => {
    const id = mockWarehouseId();
    return id !== undefined ? { warehouseId: id } : null;
  },
}));

describe('useTraceMetricsQuery', () => {
  const server = setupServer();

  const defaultParams = {
    experimentIds: ['test-exp-1'],
    startTimeMs: 1000000,
    endTimeMs: 2000000,
    viewType: MetricViewType.TRACES,
    metricName: TraceMetricKey.TRACE_COUNT,
    aggregations: [{ aggregation_type: AggregationType.COUNT }],
  };

  const createQueryClient = () => new QueryClient({ defaultOptions: { queries: { retry: false } } });

  const createWrapper = () => {
    const queryClient = createQueryClient();
    return ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockShouldUseTracesV4API.mockReturnValue(false);
    mockWarehouseId.mockReturnValue(undefined);
    mockShouldEnableBatchedTokenMetricQueries.mockReturnValue(false);
  });

  describe('OSS mode (V4 disabled)', () => {
    it('should call the 3.0 endpoint with experiment_ids', async () => {
      let capturedBody: any = null;
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useTraceMetricsQuery(defaultParams), { wrapper: createWrapper() });

      await waitFor(() => expect(capturedBody).not.toBeNull());

      expect(capturedBody.experiment_ids).toEqual(['test-exp-1']);
      expect(capturedBody.view_type).toBe(MetricViewType.TRACES);
      expect(capturedBody.metric_name).toBe(TraceMetricKey.TRACE_COUNT);
      expect(capturedBody.metric_names).toBeUndefined();
      expect(capturedBody.start_time_ms).toBe(1000000);
      expect(capturedBody.end_time_ms).toBe(2000000);
      // Should NOT have V4-specific fields
      expect(capturedBody.locations).toBeUndefined();
      expect(capturedBody.sql_warehouse_id).toBeUndefined();
    });

    it('should return data from the 3.0 endpoint', async () => {
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', (_req, res, ctx) =>
          res(ctx.json({ data_points: [{ metric_name: 'trace_count', dimensions: {}, values: { COUNT: 42 } }] })),
        ),
      );

      const { result } = renderHook(() => useTraceMetricsQuery(defaultParams), { wrapper: createWrapper() });

      await waitFor(() => expect(result.current.isLoading).toBe(false));

      expect(result.current.data?.data_points).toHaveLength(1);
      expect(result.current.data?.data_points[0].values['COUNT']).toBe(42);
    });

    it('should allow queries without start/end time in OSS', async () => {
      let capturedBody: any = null;
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useTraceMetricsQuery({ ...defaultParams, startTimeMs: undefined, endTimeMs: undefined }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => expect(capturedBody).not.toBeNull());
      // Query should still fire in OSS even without time range
      expect(capturedBody.experiment_ids).toEqual(['test-exp-1']);
    });
  });

  describe('V4 mode (Databricks)', () => {
    beforeEach(() => {
      mockShouldUseTracesV4API.mockReturnValue(true);
      mockWarehouseId.mockReturnValue('warehouse-123');
    });

    it('should call the 4.0 endpoint with locations format', async () => {
      let capturedBody: any = null;
      server.use(
        rest.post('ajax-api/4.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useTraceMetricsQuery(defaultParams), { wrapper: createWrapper() });

      await waitFor(() => expect(capturedBody).not.toBeNull());

      expect(capturedBody.locations).toEqual([
        { type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'test-exp-1' } },
      ]);
      expect(capturedBody.sql_warehouse_id).toBe('warehouse-123');
      // Should NOT have OSS-specific field
      expect(capturedBody.experiment_ids).toBeUndefined();
    });

    it('should include all query params in V4 request', async () => {
      let capturedBody: any = null;
      server.use(
        rest.post('ajax-api/4.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(
        () =>
          useTraceMetricsQuery({
            ...defaultParams,
            timeIntervalSeconds: 3600,
            filters: ['trace.status = "OK"'],
            dimensions: ['assessment_name'],
          }),
        { wrapper: createWrapper() },
      );

      await waitFor(() => expect(capturedBody).not.toBeNull());

      expect(capturedBody.view_type).toBe(MetricViewType.TRACES);
      expect(capturedBody.metric_name).toBe(TraceMetricKey.TRACE_COUNT);
      expect(capturedBody.metric_names).toBeUndefined();
      expect(capturedBody.start_time_ms).toBe(1000000);
      expect(capturedBody.end_time_ms).toBe(2000000);
      expect(capturedBody.time_interval_seconds).toBe(3600);
      expect(capturedBody.filters).toEqual(['trace.status = "OK"']);
      expect(capturedBody.dimensions).toEqual(['assessment_name']);
    });

    it('should convert multiple experiment IDs to locations', async () => {
      let capturedBody: any = null;
      server.use(
        rest.post('ajax-api/4.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useTraceMetricsQuery({ ...defaultParams, experimentIds: ['exp-1', 'exp-2', 'exp-3'] }), {
        wrapper: createWrapper(),
      });

      await waitFor(() => expect(capturedBody).not.toBeNull());

      expect(capturedBody.locations).toHaveLength(3);
      expect(capturedBody.locations[0].mlflow_experiment.experiment_id).toBe('exp-1');
      expect(capturedBody.locations[1].mlflow_experiment.experiment_id).toBe('exp-2');
      expect(capturedBody.locations[2].mlflow_experiment.experiment_id).toBe('exp-3');
    });

    it('should disable query when sql_warehouse_id is not available', async () => {
      mockWarehouseId.mockReturnValue(null);

      const requestMade = jest.fn();
      server.use(
        rest.post('ajax-api/4.0/mlflow/traces/metrics', async (req, res, ctx) => {
          requestMade();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      const { result } = renderHook(() => useTraceMetricsQuery(defaultParams), { wrapper: createWrapper() });

      // Wait a tick to ensure query would have fired if enabled
      await new Promise((r) => setTimeout(r, 50));

      expect(requestMade).not.toHaveBeenCalled();
      expect(result.current.isLoading).toBe(false);
    });

    it('should disable query when start_time_ms is missing', async () => {
      const requestMade = jest.fn();
      server.use(
        rest.post('ajax-api/4.0/mlflow/traces/metrics', async (req, res, ctx) => {
          requestMade();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      const { result } = renderHook(() => useTraceMetricsQuery({ ...defaultParams, startTimeMs: undefined }), {
        wrapper: createWrapper(),
      });

      // Wait a tick to ensure query would have fired if enabled
      await new Promise((r) => setTimeout(r, 50));

      expect(requestMade).not.toHaveBeenCalled();
      expect(result.current.isLoading).toBe(false);
    });

    it('should disable query when end_time_ms is missing', async () => {
      const requestMade = jest.fn();
      server.use(
        rest.post('ajax-api/4.0/mlflow/traces/metrics', async (req, res, ctx) => {
          requestMade();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      const { result } = renderHook(() => useTraceMetricsQuery({ ...defaultParams, endTimeMs: undefined }), {
        wrapper: createWrapper(),
      });

      await new Promise((r) => setTimeout(r, 50));

      expect(requestMade).not.toHaveBeenCalled();
      expect(result.current.isLoading).toBe(false);
    });
  });

  describe('metric_names support', () => {
    it('should send metric_names when metricNames param is provided', async () => {
      let capturedBody: any = null;
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(
        () =>
          useTraceMetricsQuery({
            experimentIds: ['test-exp-1'],
            startTimeMs: 1000000,
            endTimeMs: 2000000,
            viewType: MetricViewType.TRACES,
            metricNames: [TraceMetricKey.INPUT_TOKENS, TraceMetricKey.OUTPUT_TOKENS],
            aggregations: [{ aggregation_type: AggregationType.SUM }],
          }),
        { wrapper: createWrapper() },
      );

      await waitFor(() => expect(capturedBody).not.toBeNull());

      expect(capturedBody.metric_names).toEqual([TraceMetricKey.INPUT_TOKENS, TraceMetricKey.OUTPUT_TOKENS]);
    });

    it('should prefer metricNames over metricName when both are provided', async () => {
      let capturedBody: any = null;
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(
        () =>
          useTraceMetricsQuery({
            experimentIds: ['test-exp-1'],
            startTimeMs: 1000000,
            endTimeMs: 2000000,
            viewType: MetricViewType.TRACES,
            metricName: TraceMetricKey.TRACE_COUNT,
            metricNames: [TraceMetricKey.INPUT_TOKENS, TraceMetricKey.OUTPUT_TOKENS],
            aggregations: [{ aggregation_type: AggregationType.SUM }],
          }),
        { wrapper: createWrapper() },
      );

      await waitFor(() => expect(capturedBody).not.toBeNull());

      // metricNames should take precedence
      expect(capturedBody.metric_names).toEqual([TraceMetricKey.INPUT_TOKENS, TraceMetricKey.OUTPUT_TOKENS]);
    });

    it('should disable query when neither metricName nor metricNames is provided', async () => {
      const requestMade = jest.fn();
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          requestMade();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      const { result } = renderHook(
        () =>
          useTraceMetricsQuery({
            experimentIds: ['test-exp-1'],
            startTimeMs: 1000000,
            endTimeMs: 2000000,
            viewType: MetricViewType.TRACES,
            aggregations: [{ aggregation_type: AggregationType.COUNT }],
          }),
        { wrapper: createWrapper() },
      );

      await new Promise((r) => setTimeout(r, 50));

      expect(requestMade).not.toHaveBeenCalled();
      expect(result.current.isLoading).toBe(false);
    });
  });

  describe('auto-promotion of metricName to metricNames', () => {
    it('should send metric_names when flag is on and only metricName is provided', async () => {
      mockShouldEnableBatchedTokenMetricQueries.mockReturnValue(true);
      let capturedBody: any = null;
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useTraceMetricsQuery(defaultParams), { wrapper: createWrapper() });

      await waitFor(() => expect(capturedBody).not.toBeNull());

      expect(capturedBody.metric_names).toEqual([TraceMetricKey.TRACE_COUNT]);
      expect(capturedBody.metric_name).toBeUndefined();
    });

    it('should send metric_name when flag is off and only metricName is provided', async () => {
      mockShouldEnableBatchedTokenMetricQueries.mockReturnValue(false);
      let capturedBody: any = null;
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(() => useTraceMetricsQuery(defaultParams), { wrapper: createWrapper() });

      await waitFor(() => expect(capturedBody).not.toBeNull());

      expect(capturedBody.metric_name).toBe(TraceMetricKey.TRACE_COUNT);
      expect(capturedBody.metric_names).toBeUndefined();
    });

    it('should use metricNames as-is when flag is on and metricNames is already provided', async () => {
      mockShouldEnableBatchedTokenMetricQueries.mockReturnValue(true);
      let capturedBody: any = null;
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          capturedBody = await req.json();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      renderHook(
        () =>
          useTraceMetricsQuery({
            ...defaultParams,
            metricName: undefined,
            metricNames: [TraceMetricKey.INPUT_TOKENS, TraceMetricKey.OUTPUT_TOKENS],
          }),
        { wrapper: createWrapper() },
      );

      await waitFor(() => expect(capturedBody).not.toBeNull());

      expect(capturedBody.metric_names).toEqual([TraceMetricKey.INPUT_TOKENS, TraceMetricKey.OUTPUT_TOKENS]);
      expect(capturedBody.metric_name).toBeUndefined();
    });
  });

  describe('disabled state', () => {
    it('should not fetch when enabled is false', async () => {
      const requestMade = jest.fn();
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          requestMade();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      const { result } = renderHook(() => useTraceMetricsQuery({ ...defaultParams, enabled: false }), {
        wrapper: createWrapper(),
      });

      await new Promise((r) => setTimeout(r, 50));

      expect(requestMade).not.toHaveBeenCalled();
      expect(result.current.isLoading).toBe(false);
    });

    it('should not fetch when experimentIds is empty', async () => {
      const requestMade = jest.fn();
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/metrics', async (req, res, ctx) => {
          requestMade();
          return res(ctx.json({ data_points: [] }));
        }),
      );

      const { result } = renderHook(() => useTraceMetricsQuery({ ...defaultParams, experimentIds: [] }), {
        wrapper: createWrapper(),
      });

      await new Promise((r) => setTimeout(r, 50));

      expect(requestMade).not.toHaveBeenCalled();
      expect(result.current.isLoading).toBe(false);
    });
  });
});
