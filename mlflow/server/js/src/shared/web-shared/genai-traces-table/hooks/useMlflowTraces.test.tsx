import { renderHook, waitFor } from '@testing-library/react';
import React from 'react';

import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

import { useGenAiTraceEvaluationArtifacts } from './useGenAiTraceEvaluationArtifacts';
import {
  invalidateMlflowSearchTracesCache,
  useMlflowTraces,
  useMlflowTracesTableMetadata,
  useSearchMlflowTraces,
} from './useMlflowTraces';
import { EXECUTION_DURATION_COLUMN_ID, SESSION_COLUMN_ID, LOGGED_MODEL_COLUMN_ID } from './useTableColumns';
import { FilterOperator, TracesTableColumnGroup, TracesTableColumnType } from '../types';
import type { TraceInfoV3, RunEvaluationTracesDataEntry } from '../types';
import { shouldEnableUnifiedEvalTab } from '../utils/FeatureUtils';
import { fetchFn } from '../utils/FetchUtils';

// Mock shouldEnableUnifiedEvalTab
jest.mock('../utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../utils/FeatureUtils')>('../utils/FeatureUtils'),
  shouldEnableUnifiedEvalTab: jest.fn(),
  getMlflowTracesSearchPageSize: jest.fn().mockReturnValue(10000),
}));

// Mock the artifact hook
jest.mock('./useGenAiTraceEvaluationArtifacts', () => ({
  useGenAiTraceEvaluationArtifacts: jest.fn(),
}));

// Mock fetchFn
jest.mock('../utils/FetchUtils', () => ({
  fetchFn: jest.fn(),
  getAjaxUrl: jest.fn().mockImplementation((relativeUrl) => '/' + relativeUrl),
}));

// Mock global window.fetch
global.fetch = jest.fn();

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false, // Turn off retries to simplify testing
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <IntlProvider locale="en">
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </IntlProvider>
  );
}

describe('useMlflowTracesTableMetadata', () => {
  test('returns empty data and isLoading = false when disabled is true', async () => {
    const { result } = renderHook(
      () => useMlflowTracesTableMetadata({ experimentId: 'some-experiment', disabled: true }),
      {
        wrapper: createWrapper(),
      },
    );

    expect(result.current.isLoading).toBe(false);
  });
});

describe('useSearchMlflowTraces', () => {
  test('returns empty data and isLoading = false when disabled is true', async () => {
    const { result } = renderHook(() => useSearchMlflowTraces({ experimentId: 'some-experiment', disabled: true }), {
      wrapper: createWrapper(),
    });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toEqual([]);
  });

  test('makes network call to fetch traces when enabled', async () => {
    jest.mocked(fetchFn).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        traces: [
          {
            trace_id: 'trace_1',
            request: '{"input": "value"}',
            response: '{"output": "value"}',
          },
        ],
        next_page_token: undefined,
      }),
    } as any);

    const { result } = renderHook(() => useSearchMlflowTraces({ experimentId: 'experiment-xyz' }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(fetchFn).toHaveBeenCalledTimes(1); // only one page
    expect(result.current.data).toHaveLength(1);
    expect(result.current.data?.[0]).toEqual({
      trace_id: 'trace_1',
      request: '{"input": "value"}',
      response: '{"output": "value"}',
    });
  });

  test('uses filters to fetch traces', async () => {
    jest.mocked(fetchFn).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        traces: [
          {
            trace_id: 'trace_1',
            request: '{"input": "value"}',
            response: '{"output": "value"}',
          },
        ],
        next_page_token: undefined,
      }),
    } as any);

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          experimentId: 'experiment-xyz',
          runUuid: 'run-xyz',
          timeRange: {
            startTime: '100',
            endTime: '200',
          },
          filters: [
            // assessment is client side so should get filtered out from network filters
            {
              column: TracesTableColumnGroup.ASSESSMENT,
              key: 'overall',
              operator: FilterOperator.EQUALS,
              value: 'success',
            },
            {
              column: TracesTableColumnGroup.TAG,
              operator: FilterOperator.EQUALS,
              value: 'user_1',
              key: 'user',
            },
            // no tag key so should be ignored
            {
              column: TracesTableColumnGroup.TAG,
              operator: FilterOperator.EQUALS,
              value: 'user_1',
            },
            {
              column: TracesTableColumnGroup.TAG,
              operator: FilterOperator.EQUALS,
              value: 'user_2',
              key: 'user',
            },
            {
              column: 'execution_duration',
              operator: FilterOperator.GREATER_THAN,
              value: 1000,
            },
            {
              column: 'user',
              operator: FilterOperator.EQUALS,
              value: 'user_3',
              key: 'user',
            },
            {
              column: 'run_name',
              operator: FilterOperator.EQUALS,
              value: 'run_1',
              key: 'run_name',
            },
            {
              column: LOGGED_MODEL_COLUMN_ID,
              operator: FilterOperator.EQUALS,
              value: 'version_1',
              key: 'version',
            },
            {
              column: 'state',
              operator: FilterOperator.EQUALS,
              value: 'OK',
            },
            {
              column: 'trace_name',
              operator: FilterOperator.EQUALS,
              value: 'trace_1',
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(fetchFn).toHaveBeenCalledWith(
      '/ajax-api/3.0/mlflow/traces/search',
      expect.objectContaining({
        body: JSON.stringify({
          locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
          filter: `request_metadata."mlflow.sourceRun" = 'run-xyz' AND attributes.timestamp_ms > 100 AND attributes.timestamp_ms < 200 AND tags.user = 'user_1' AND tags.user = 'user_2' AND attributes.execution_time_ms > 1000 AND request_metadata."mlflow.trace.user" = 'user_3' AND request_metadata."mlflow.sourceRun" = 'run_1' AND request_metadata."mlflow.modelId" = 'version_1' AND attributes.status = 'OK' AND attributes.name = 'trace_1'`,
          max_results: 10000,
        }),
      }),
    );
  });

  test('handles custom metadata filters correctly', async () => {
    jest.mocked(fetchFn).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        traces: [
          {
            trace_id: 'trace_1',
            request: '{"input": "value"}',
            response: '{"output": "value"}',
            trace_metadata: {
              user_id: 'user123',
              environment: 'production',
              'mlflow.internal.key': 'internal_value', // Should be excluded
            },
          },
        ],
        next_page_token: undefined,
      }),
    } as any);

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          experimentId: 'experiment-xyz',
          filters: [
            {
              column: 'custom_metadata:user_id',
              operator: FilterOperator.EQUALS,
              value: 'user123',
            },
            {
              column: 'custom_metadata:environment',
              operator: FilterOperator.EQUALS,
              value: 'production',
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(fetchFn).toHaveBeenCalledWith(
      '/ajax-api/3.0/mlflow/traces/search',
      expect.objectContaining({
        body: JSON.stringify({
          locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
          filter: `request_metadata.user_id = 'user123' AND request_metadata.environment = 'production'`,
          max_results: 10000,
        }),
      }),
    );
  });

  test('excludes MLflow internal keys from custom metadata filters', async () => {
    jest.mocked(fetchFn).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        traces: [
          {
            trace_id: 'trace_1',
            request: '{"input": "value"}',
            response: '{"output": "value"}',
            trace_metadata: {
              user_id: 'user123',
              'mlflow.run_id': 'run123', // Should be excluded
              'mlflow.internal.key': 'internal_value', // Should be excluded
            },
          },
        ],
        next_page_token: undefined,
      }),
    } as any);

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          experimentId: 'experiment-xyz',
          filters: [
            {
              column: 'custom_metadata:user_id',
              operator: FilterOperator.EQUALS,
              value: 'user123',
            },
            {
              column: 'custom_metadata:mlflow.run_id', // This should be ignored
              operator: FilterOperator.EQUALS,
              value: 'run123',
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(fetchFn).toHaveBeenCalledWith(
      '/ajax-api/3.0/mlflow/traces/search',
      expect.objectContaining({
        body: JSON.stringify({
          locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
          filter: `request_metadata.user_id = 'user123' AND request_metadata.mlflow.run_id = 'run123'`,
          max_results: 10000,
        }),
      }),
    );
  });

  test('uses order_by to fetch traces for server sortable column', async () => {
    jest.mocked(fetchFn).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        traces: [
          {
            trace_id: 'trace_1',
            request: '{"input": "value"}',
            response: '{"output": "value"}',
          },
        ],
        next_page_token: undefined,
      }),
    } as any);

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          experimentId: 'experiment-xyz',
          runUuid: 'run-xyz',
          timeRange: {
            startTime: '100',
            endTime: '200',
          },
          tableSort: {
            key: EXECUTION_DURATION_COLUMN_ID,
            asc: false,
            type: TracesTableColumnType.INPUT,
          },
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(fetchFn).toHaveBeenCalledWith(
      '/ajax-api/3.0/mlflow/traces/search',
      expect.objectContaining({
        body: JSON.stringify({
          locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
          filter: `request_metadata."mlflow.sourceRun" = 'run-xyz' AND attributes.timestamp_ms > 100 AND attributes.timestamp_ms < 200`,
          max_results: 10000,
          order_by: ['execution_time DESC'],
        }),
      }),
    );
  });

  test('Does not use order_by to fetch traces for non-server sortable column', async () => {
    jest.mocked(fetchFn).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        traces: [
          {
            trace_id: 'trace_1',
            request: '{"input": "value"}',
            response: '{"output": "value"}',
          },
        ],
        next_page_token: undefined,
      }),
    } as any);

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          experimentId: 'experiment-xyz',
          runUuid: 'run-xyz',
          timeRange: {
            startTime: '100',
            endTime: '200',
          },
          tableSort: {
            key: SESSION_COLUMN_ID,
            asc: false,
            type: TracesTableColumnType.INPUT,
          },
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(fetchFn).toHaveBeenCalledWith(
      '/ajax-api/3.0/mlflow/traces/search',
      expect.objectContaining({
        body: JSON.stringify({
          locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
          filter: `request_metadata."mlflow.sourceRun" = 'run-xyz' AND attributes.timestamp_ms > 100 AND attributes.timestamp_ms < 200`,
          max_results: 10000,
          order_by: [],
        }),
      }),
    );
  });

  test("use loggedModelId and sqlWarehouseId to fetch model's online traces", async () => {
    jest.mocked(fetchFn).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        traces: [
          {
            trace_id: 'trace_1',
            request: '{"input": "value"}',
            response: '{"output": "value"}',
          },
        ],
        next_page_token: undefined,
      }),
    } as any);

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          experimentId: 'experiment-xyz',
          runUuid: 'run-xyz',
          loggedModelId: 'model-123',
          sqlWarehouseId: 'warehouse-456',
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(fetchFn).toHaveBeenCalledWith(
      '/ajax-api/3.0/mlflow/traces/search',
      expect.objectContaining({
        body: JSON.stringify({
          locations: [{ mlflow_experiment: { experiment_id: 'experiment-xyz' }, type: 'MLFLOW_EXPERIMENT' }],
          filter: `request_metadata."mlflow.sourceRun" = 'run-xyz'`,
          max_results: 10000,
          model_id: 'model-123',
          sql_warehouse_id: 'warehouse-456',
        }),
      }),
    );
  });

  it('handles client side filters', async () => {
    jest.mocked(fetchFn).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        traces: [
          {
            trace_id: 'trace_1',
            request: '{"input": "value"}',
            response: '{"output": "value"}',
            assessments: [
              {
                assessment_id: 'overall_assessment',
                assessment_name: 'overall_assessment',
                trace_id: 'trace_1',
                feedback: {
                  value: 'pass',
                },
              },
              {
                assessment_id: 'correctness',
                assessment_name: 'correctness_assessment',
                trace_id: 'trace_1',
                feedback: {
                  value: 'pass',
                },
              },
            ],
          },
          {
            trace_id: 'trace_2',
            request: '{"input": "value"}',
            response: '{"output": "value"}',
            assessments: [
              {
                assessment_id: 'overall_assessment',
                assessment_name: 'overall_assessment',
                trace_id: 'trace_2',
                feedback: {
                  value: 'fail',
                },
              },
              {
                assessment_id: 'correctness',
                assessment_name: 'correctness_assessment',
                trace_id: 'trace_2',
                feedback: {
                  value: 'fail',
                },
              },
            ],
          },
        ] as TraceInfoV3[],
        next_page_token: undefined,
      }),
    } as any);

    const { result } = renderHook(
      () =>
        useSearchMlflowTraces({
          currentRunDisplayName: 'run-xyz',
          experimentId: 'experiment-xyz',
          runUuid: 'run-xyz',
          timeRange: {
            startTime: '100',
            endTime: '200',
          },
          filters: [
            {
              column: TracesTableColumnGroup.ASSESSMENT,
              key: 'overall_assessment',
              operator: FilterOperator.EQUALS,
              value: 'pass',
            },
          ],
        }),
      {
        wrapper: createWrapper(),
      },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.data).toHaveLength(1);
    expect(result.current.data?.[0].trace_id).toBe('trace_1');
    expect(result.current.data?.[0].assessments).toHaveLength(2);
    expect(result.current.data?.[0].assessments?.[0].assessment_id).toBe('overall_assessment');
    expect(result.current.data?.[0].assessments?.[0]?.feedback?.value).toBe('pass');
    expect(result.current.data?.[0].assessments?.[1].assessment_id).toBe('correctness');
    expect(result.current.data?.[0].assessments?.[1]?.feedback?.value).toBe('pass');
  });
});

describe('useMlflowTraces', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.mocked(shouldEnableUnifiedEvalTab).mockReturnValue(true);
  });

  test('returns empty data and isLoading = false when disabled is true', async () => {
    jest.mocked(useGenAiTraceEvaluationArtifacts).mockReturnValue({
      data: [],
      isLoading: false,
    } as any);
    // Render the hook with disabled = true
    const { result } = renderHook(() => useMlflowTraces('some-experiment', 'some-run-uuid', [], true), {
      wrapper: createWrapper(),
    });

    expect(result.current.isLoading).toBe(false);
    expect(result.current.data).toEqual([]);
  });

  test('calls artifact hook if runUuid is provided', async () => {
    jest.mocked(useGenAiTraceEvaluationArtifacts).mockReturnValue({
      data: [{ evaluationId: 'artifact-entry-1' } as RunEvaluationTracesDataEntry],
      isLoading: false,
    } as any);

    jest.mocked(fetchFn).mockResolvedValue({
      ok: true,
      json: async () => ({ traces: [], next_page_token: undefined }),
    } as any);

    const { result } = renderHook(() => useMlflowTraces('experiment-123', 'run-abc', []), { wrapper: createWrapper() });

    await waitFor(() => {
      // The fetch is triggered by React Query in the background
      expect(result.current.isLoading).toBe(false);
    });

    // The artifact hook was called
    expect(useGenAiTraceEvaluationArtifacts).toHaveBeenCalledWith(
      {
        artifacts: [],
        runUuid: 'run-abc',
      },
      { disabled: false },
    );

    // The final search data should be empty (search returned no traces),
    // so only artifact data is present after merging.
    expect(result.current.data).toHaveLength(1);
    expect(result.current.data[0].evaluationId).toBe('artifact-entry-1');
  });

  test('does not call search api if feature flag is not enabled', async () => {
    jest.mocked(shouldEnableUnifiedEvalTab).mockReturnValue(false);
    jest.mocked(useGenAiTraceEvaluationArtifacts).mockReturnValue({
      data: [{ evaluationId: 'artifact-entry-1' } as RunEvaluationTracesDataEntry],
      isLoading: false,
    } as any);

    const { result } = renderHook(() => useMlflowTraces('experiment-xyz', 'run-xyz'), { wrapper: createWrapper() });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(fetchFn).toHaveBeenCalledTimes(0); // should not be called
    expect(result.current.data).toHaveLength(1);
    expect(result.current.data[0].evaluationId).toBe('artifact-entry-1');
  });

  test('makes network call to fetch traces when enabled', async () => {
    jest.mocked(useGenAiTraceEvaluationArtifacts).mockReturnValue({
      data: [],
      isLoading: false,
    } as any);

    jest.mocked(fetchFn).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        traces: [
          {
            trace_id: 'trace_1',
            request: '{"input": "value"}',
            response: '{"output": "value"}',
          },
        ],
        next_page_token: undefined,
      }),
    } as any);

    const { result } = renderHook(() => useMlflowTraces('experiment-xyz', 'run-xyz'), { wrapper: createWrapper() });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(fetchFn).toHaveBeenCalledTimes(1); // only one page
    expect(result.current.data).toHaveLength(1);
    expect(result.current.data[0].inputs).toEqual({});
    expect(result.current.data[0].outputs).toEqual({});
    expect(result.current.data[0].traceInfo).toEqual({
      trace_id: 'trace_1',
      request: '{"input": "value"}',
      response: '{"output": "value"}',
    });
  });

  test('respects pagination until next_page_token is missing', async () => {
    jest.mocked(useGenAiTraceEvaluationArtifacts).mockReturnValue({
      data: [],
      isLoading: false,
    } as any);

    jest
      .mocked(fetchFn)
      // First call returns 1 trace and a next_page_token
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          traces: [{ trace_id: 'trace_page_1' }],
          next_page_token: 'abc',
        }),
      } as any)
      // Second call returns 1 trace and a next_page_token
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          traces: [{ trace_id: 'trace_page_2' }],
          next_page_token: 'cde',
        }),
      } as any)
      // Third call returns 1 trace and no next_page_token
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          traces: [{ trace_id: 'trace_page_3' }],
          next_page_token: undefined,
        }),
      } as any);

    const { result } = renderHook(() => useMlflowTraces('experiment-paginated', 'run-paginated'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    // We expect 2 network calls
    expect(fetchFn).toHaveBeenCalledTimes(3);

    // Should have 3 traces total
    expect(result.current.data).toHaveLength(3);
    expect(result.current.data[0]?.traceInfo?.trace_id).toBe('trace_page_1');
    expect(result.current.data[1]?.traceInfo?.trace_id).toBe('trace_page_2');
    expect(result.current.data[2]?.traceInfo?.trace_id).toBe('trace_page_3');
  });

  test('falls back to artifact data if search fails', async () => {
    // Return some artifact data
    jest.mocked(useGenAiTraceEvaluationArtifacts).mockReturnValue({
      data: [
        {
          evaluationId: 'artifact-trace-1',
        },
      ],
      isLoading: false,
    } as any);

    // Search call fails
    jest.mocked(fetchFn).mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({}),
    } as any);

    const { result } = renderHook(() => useMlflowTraces('experiment-fail', 'run-fail'), { wrapper: createWrapper() });

    // The request should fail, but we wait for the hook to settle
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    // Fallback to artifact data
    expect(result.current.data).toHaveLength(1);
    expect(result.current.data[0].evaluationId).toBe('artifact-trace-1');
  });

  test('uses artifact data if any paginated call fails mid-way', async () => {
    // Mock artifact data
    jest.mocked(useGenAiTraceEvaluationArtifacts).mockReturnValue({
      data: [
        {
          evaluationId: 'artifact-partial-error-1',
        },
      ],
      isLoading: false,
    } as any);

    // First fetch call succeeds, returning a valid page plus a next_page_token
    jest
      .mocked(fetchFn)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          traces: [{ trace_id: 'trace_page_1' }],
          next_page_token: 'page2',
        }),
      } as any)
      // Second fetch call fails
      .mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: async () => ({}),
      } as any);

    const { result } = renderHook(() => useMlflowTraces('experiment-partial-error', 'run-partial-error'), {
      wrapper: createWrapper(),
    });

    // Wait for the query to finish loading
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    // Since the second page fails, the entire query is considered an error,
    // and the hook falls back to artifact data instead of partial data.
    expect(fetchFn).toHaveBeenCalledTimes(2);

    // The final data must come from artifacts only, ignoring the partial page-1 data
    // because the query throws an error on page 2.
    expect(result.current.data).toHaveLength(1);
    expect(result.current.data[0].evaluationId).toBe('artifact-partial-error-1');
  });

  test('skips artifact fetch if runUuid is not provided', async () => {
    // In this scenario, we pretend the user wants to see all experiment traces.
    // So we won't call the artifact hook.
    jest.mocked(useGenAiTraceEvaluationArtifacts).mockReturnValue({
      data: [],
      isLoading: false,
    } as any);

    jest.mocked(fetchFn).mockResolvedValue({
      ok: true,
      json: async () => ({
        traces: [
          {
            trace_id: 'trace_1',
            request: '{"input": "value"}',
            response: '{"output": "value"}',
          },
        ],
        next_page_token: undefined,
      }),
    } as any);

    const { result } = renderHook(() => useMlflowTraces('experiment-all'), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    // The artifact hook is disabled because runUuid is `undefined`.
    expect(useGenAiTraceEvaluationArtifacts).toHaveBeenCalledWith(
      { artifacts: undefined, runUuid: '' },
      { disabled: true },
    );

    expect(result.current.data).toHaveLength(1);
    expect(result.current.data[0]?.traceInfo?.trace_id).toBe('trace_1');
    expect(result.current.data[0].inputs).toEqual({});
    expect(result.current.data[0].outputs).toEqual({});
  });

  test('use artifact data if both artifact and search data exist', async () => {
    jest.mocked(useGenAiTraceEvaluationArtifacts).mockReturnValue({
      data: [
        {
          evaluationId: 'artifact-trace-1',
        },
      ],
      isLoading: false,
    } as any);

    jest.mocked(fetchFn).mockResolvedValue({
      ok: true,
      json: async () => ({
        traces: [
          {
            trace_id: 'trace_1',
            request: '{"input": "value"}',
            response: '{"output": "value"}',
          },
        ],
        next_page_token: undefined,
      }),
    } as any);

    const { result } = renderHook(() => useMlflowTraces('experiment-xyz', 'run-xyz'), { wrapper: createWrapper() });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    // search should still be called but return data should be from artifacts
    expect(fetchFn).toHaveBeenCalledTimes(1);

    expect(result.current.data).toHaveLength(1);
    expect(result.current.data[0].evaluationId).toBe('artifact-trace-1');
  });
});

describe('invalidateMlflowSearchTracesCache', () => {
  test('invalidates queries with searchMlflowTraces key', () => {
    const queryClient = new QueryClient();
    const invalidateQueriesSpy = jest.spyOn(queryClient, 'invalidateQueries');

    // Call the function
    invalidateMlflowSearchTracesCache({ queryClient });

    // Verify that invalidateQueries was called with the correct key
    expect(invalidateQueriesSpy).toHaveBeenCalledTimes(1);
    expect(invalidateQueriesSpy).toHaveBeenCalledWith({ queryKey: ['searchMlflowTraces'] });
  });
});
