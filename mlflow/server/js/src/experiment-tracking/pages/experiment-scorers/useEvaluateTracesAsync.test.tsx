import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { useEvaluateTracesAsync } from './useEvaluateTracesAsync';
import type { ModelTrace, ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { setupServer } from '../../../common/utils/setup-msw';
import { rest } from 'msw';
import { ScorerEvaluationScope } from './constants';
import { isSessionJudgeEvaluationResult, JudgeEvaluationResult } from './useEvaluateTraces.common';

jest.useFakeTimers();

/**
 * Helper to create mock trace data with proper V3 structure
 */
function createMockTrace(traceId: string, sessionId?: string): ModelTrace {
  return {
    info: {
      trace_id: traceId,
      // trace_location is required for isV3ModelTraceInfo to return true
      trace_location: {
        type: 'MLFLOW_EXPERIMENT',
        mlflow_experiment: { experiment_id: 'exp-123' },
      },
      request_time: Date.now(),
      state: 'OK',
      ...(sessionId && { trace_metadata: { 'mlflow.trace.session': sessionId } }),
    } as unknown as ModelTraceInfoV3,
    data: {
      spans: [
        {
          trace_id: traceId,
          span_id: `span-${traceId}`,
          name: 'root',
          trace_state: '',
          parent_span_id: null,
          start_time_unix_nano: '0',
          end_time_unix_nano: '1000000',
          status: { code: 'STATUS_CODE_UNSET' },
          attributes: {
            'mlflow.spanInputs': `input-${traceId}`,
            'mlflow.spanOutputs': `output-${traceId}`,
          },
        },
      ],
    },
  };
}

/**
 * Helper to setup MSW handlers for trace fetching endpoints
 */
function setupTraceFetchHandlers(server: ReturnType<typeof setupServer>, traces: Map<string, ModelTrace>) {
  // Handler for trace info endpoint: GET ajax-api/3.0/mlflow/traces/:traceId
  server.use(
    rest.get('ajax-api/3.0/mlflow/traces/:traceId', (req, res, ctx) => {
      const { traceId } = req.params;
      const trace = traces.get(traceId as string);
      return res(
        ctx.json({
          trace: {
            trace_info: trace?.info || {},
          },
        }),
      );
    }),
  );

  // Handler for trace artifact/data endpoint: GET ajax-api/3.0/mlflow/get-trace-artifact
  server.use(
    rest.get('ajax-api/3.0/mlflow/get-trace-artifact', (req, res, ctx) => {
      const requestId = req.url.searchParams.get('request_id');
      const trace = traces.get(requestId || '');
      return res(ctx.json(trace?.data || {}));
    }),
  );
}

describe('useEvaluateTracesAsync', () => {
  const server = setupServer();
  let queryClient: QueryClient;
  let wrapper: React.ComponentType<{ children: React.ReactNode }>;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
        mutations: {
          retry: false,
        },
      },
    });

    wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );

    jest.clearAllMocks();
  });

  describe('Happy Path - Job Initiation and Job Succeeds', () => {
    it('should successfully start evaluation job and return results when job succeeds', async () => {
      const traceId = 'trace-1';
      const experimentId = 'exp-123';
      const jobId = 'job-123';
      const serializedScorer = 'mock-serialized-scorer';

      const mockTrace = createMockTrace(traceId);
      const traces = new Map([[traceId, mockTrace]]);

      // Setup trace search handler
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/search', (_req, res, ctx) => {
          return res(
            ctx.json({
              traces: [{ trace_id: traceId }],
              next_page_token: undefined,
            }),
          );
        }),
      );

      // Setup scorer invoke handler - returns job ID
      server.use(
        rest.post('ajax-api/3.0/mlflow/scorer/invoke', (_req, res, ctx) => {
          return res(
            ctx.json({
              jobs: [{ job_id: jobId }],
            }),
          );
        }),
      );

      // Setup job status handler - first call returns RUNNING, second returns SUCCEEDED
      let jobStatusCallCount = 0;
      server.use(
        rest.get(`ajax-api/3.0/jobs/${jobId}`, (_req, res, ctx) => {
          jobStatusCallCount++;
          if (jobStatusCallCount === 1) {
            return res(ctx.json({ status: 'RUNNING' }));
          }
          return res(
            ctx.json({
              status: 'SUCCEEDED',
              result: {
                [traceId]: {
                  assessments: [
                    {
                      assessment_name: 'test_scorer',
                      numeric_value: 0.9,
                      rationale: 'Test passed',
                    },
                  ],
                },
              },
            }),
          );
        }),
      );

      // Setup trace fetching handlers using MSW
      setupTraceFetchHandlers(server, traces);

      const onScorerFinished = jest.fn();

      const { result } = renderHook(
        () =>
          useEvaluateTracesAsync({
            onScorerFinished,
          }),
        { wrapper },
      );

      // Initial state should be idle
      expect(result.current[1].latestEvaluation).toBeNull();
      expect(result.current[1].isLoading).toBe(false);
      expect(result.current[1].error).toBeNull();

      // Start evaluation
      act(() => {
        const [evaluateFunction] = result.current;
        evaluateFunction({
          itemIds: ['trace-1'],
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          experimentId,
          judgeInstructions: 'Test instructions',
          serializedScorer,
        });
      });

      // Wait for job to complete and results to be available
      await waitFor(() => {
        expect(result.current[1].isLoading).toBe(false);
        expect(result.current[1].latestEvaluation).not.toBeNull();
      });

      const [, finalState] = result.current;

      // Verify final state
      expect(finalState.latestEvaluation).toHaveLength(1);
      expect(finalState.latestEvaluation?.[0]).toEqual({
        trace: mockTrace,
        results: [
          {
            assessment_name: 'test_scorer',
            numeric_value: 0.9,
            rationale: 'Test passed',
          },
        ],
        error: null,
      });
      expect(finalState.error).toBeNull();

      // Verify onScorerFinished was called with SUCCEEDED status
      expect(onScorerFinished).toHaveBeenCalledWith(expect.objectContaining({ status: 'SUCCEEDED' }));
    });
  });

  describe('Bad Path - Job Creation Fails', () => {
    it('should handle error when scorer invoke endpoint fails', async () => {
      jest.spyOn(console, 'error').mockImplementation(() => {});
      const traceId = 'trace-1';
      const experimentId = 'exp-123';
      const serializedScorer = 'mock-serialized-scorer';

      const mockTrace = createMockTrace(traceId);
      const traces = new Map([[traceId, mockTrace]]);

      // Setup trace search handler
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/search', (_req, res, ctx) => {
          return res(
            ctx.json({
              traces: [{ trace_id: traceId }],
              next_page_token: undefined,
            }),
          );
        }),
      );

      // Setup scorer invoke handler to return an error
      server.use(
        rest.post('ajax-api/3.0/mlflow/scorer/invoke', (_req, res, ctx) => {
          return res(ctx.status(500), ctx.json({ error_code: 'INTERNAL_ERROR', message: 'Failed to create job' }));
        }),
      );

      // Setup trace fetching handlers using MSW
      setupTraceFetchHandlers(server, traces);

      const { result } = renderHook(() => useEvaluateTracesAsync({}), { wrapper });

      await act(async () => {
        const [evaluateFunction] = result.current;
        evaluateFunction({
          itemIds: ['trace-1'],
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          experimentId,
          judgeInstructions: 'Test instructions',
          serializedScorer,
        });
      });

      // Wait for the error to propagate and loading to finish
      await waitFor(() => {
        expect(result.current[1].error).not.toBeNull();
        expect(result.current[1].isLoading).toBe(false);
      });

      const [, state] = result.current;

      // Should have error state
      expect(state.error).toBeTruthy();
      expect(state.isLoading).toBe(false);
      jest.restoreAllMocks();
    });
  });

  describe('Bad Path - Job Fails After Two Polls', () => {
    it('should handle job failure after initial PENDING and RUNNING states', async () => {
      const traceId = 'trace-1';
      const experimentId = 'exp-123';
      const jobId = 'job-failing';
      const serializedScorer = 'mock-serialized-scorer';

      const mockTrace = createMockTrace(traceId);
      const traces = new Map([[traceId, mockTrace]]);

      // Setup trace search handler
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/search', (_req, res, ctx) => {
          return res(
            ctx.json({
              traces: [{ trace_id: traceId }],
              next_page_token: undefined,
            }),
          );
        }),
      );

      // Setup scorer invoke handler
      server.use(
        rest.post('ajax-api/3.0/mlflow/scorer/invoke', (_req, res, ctx) => {
          return res(ctx.json({ jobs: [{ job_id: jobId }] }));
        }),
      );

      // Setup job status handler:
      // Poll 1: PENDING
      // Poll 2: RUNNING
      // Poll 3: FAILED
      let jobStatusCallCount = 0;
      server.use(
        rest.get(`ajax-api/3.0/jobs/${jobId}`, (_req, res, ctx) => {
          jobStatusCallCount++;
          if (jobStatusCallCount === 1) {
            return res(ctx.json({ status: 'PENDING' }));
          }
          if (jobStatusCallCount === 2) {
            return res(ctx.json({ status: 'RUNNING' }));
          }
          // Third poll - job fails
          return res(
            ctx.json({
              status: 'FAILED',
              result: 'Scorer execution failed: Model endpoint unavailable',
            }),
          );
        }),
      );

      // Setup trace fetching handlers using MSW
      setupTraceFetchHandlers(server, traces);

      const { result } = renderHook(() => useEvaluateTracesAsync({}), { wrapper });

      // Start evaluation
      act(() => {
        const [evaluateTracesFn] = result.current;
        evaluateTracesFn({
          itemIds: ['trace-1'],
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          experimentId,
          judgeInstructions: 'Test instructions',
          serializedScorer,
        });
      });

      // Wait for job to fail - error is set when job fails
      await waitFor(() => {
        expect(result.current[1].isLoading).toBe(false);
        expect(result.current[1].error?.message).toBe('Scorer execution failed: Model endpoint unavailable');
      });

      // Verify job was polled at least 3 times
      expect(jobStatusCallCount).toBeGreaterThanOrEqual(3);
    }, 30000);

    it('should handle job failure with detailed error message', async () => {
      const traceId = 'trace-1';
      const experimentId = 'exp-123';
      const jobId = 'job-error';
      const serializedScorer = 'mock-serialized-scorer';
      const errorMessage = 'Rate limit exceeded: Too many requests to the scoring endpoint';

      const mockTrace = createMockTrace(traceId);
      const traces = new Map([[traceId, mockTrace]]);

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/search', (_req, res, ctx) => {
          return res(ctx.json({ traces: [{ trace_id: traceId }], next_page_token: undefined }));
        }),
        rest.post('ajax-api/3.0/mlflow/scorer/invoke', (_req, res, ctx) => {
          return res(ctx.json({ jobs: [{ job_id: jobId }] }));
        }),
      );

      // Poll 1: RUNNING, Poll 2: FAILED
      let pollCount = 0;
      server.use(
        rest.get(`ajax-api/3.0/jobs/${jobId}`, (_req, res, ctx) => {
          pollCount++;
          if (pollCount === 1) {
            return res(ctx.json({ status: 'RUNNING' }));
          }
          return res(ctx.json({ status: 'FAILED', result: errorMessage }));
        }),
      );

      // Setup trace fetching handlers using MSW
      setupTraceFetchHandlers(server, traces);

      const { result } = renderHook(() => useEvaluateTracesAsync({}), {
        wrapper,
      });

      act(() => {
        const [evaluateTracesFn] = result.current;
        evaluateTracesFn({
          itemIds: ['trace-1'],
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          experimentId,
          judgeInstructions: 'Test',
          serializedScorer,
        });
      });

      // Wait for job to fail - error is set when job fails
      await waitFor(() => {
        expect(result.current[1].isLoading).toBe(false);
        expect(result.current[1].error?.message).toBe(errorMessage);
      });
    }, 30000);
  });

  describe('Reset Functionality', () => {
    it('should reset state when reset is called', async () => {
      const traceId = 'trace-1';
      const experimentId = 'exp-123';
      const jobId = 'job-reset';
      const serializedScorer = 'mock-serialized-scorer';

      const mockTrace = createMockTrace(traceId);
      const traces = new Map([[traceId, mockTrace]]);

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/search', (_req, res, ctx) => {
          return res(ctx.json({ traces: [{ trace_id: traceId }], next_page_token: undefined }));
        }),
        rest.post('ajax-api/3.0/mlflow/scorer/invoke', (_req, res, ctx) => {
          return res(ctx.json({ jobs: [{ job_id: jobId }] }));
        }),
        rest.get(`ajax-api/3.0/jobs/${jobId}`, (_req, res, ctx) => {
          return res(
            ctx.json({
              status: 'SUCCEEDED',
              result: { [traceId]: { assessments: [{ assessment_name: 'test', numeric_value: 1 }] } },
            }),
          );
        }),
      );

      // Setup trace fetching handlers using MSW
      setupTraceFetchHandlers(server, traces);

      const { result } = renderHook(() => useEvaluateTracesAsync({}), {
        wrapper,
      });

      // Run evaluation
      await act(async () => {
        const [evaluateTracesFn] = result.current;
        evaluateTracesFn({
          itemIds: ['trace-1'],
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          experimentId,
          judgeInstructions: 'Test',
          serializedScorer,
        });
      });

      // Wait for results
      await waitFor(() => {
        expect(result.current[1].latestEvaluation).not.toBeNull();
      });

      // Reset
      act(() => {
        result.current[1].reset();
      });

      // Verify state is reset
      await waitFor(() => {
        expect(result.current[1].latestEvaluation).toBeNull();
        expect(result.current[1].isLoading).toBe(false);
      });
    });
  });

  describe('Job Success with Trace Failures', () => {
    it('should return failure error messages when job succeeds but contains trace-level failures', async () => {
      const traceId = 'trace-1';
      const experimentId = 'exp-123';
      const jobId = 'job-with-failures';
      const serializedScorer = 'mock-serialized-scorer';

      const mockTrace = createMockTrace(traceId);
      const traces = new Map([[traceId, mockTrace]]);

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/search', (_req, res, ctx) => {
          return res(ctx.json({ traces: [{ trace_id: traceId }], next_page_token: undefined }));
        }),
        rest.post('ajax-api/3.0/mlflow/scorer/invoke', (_req, res, ctx) => {
          return res(ctx.json({ jobs: [{ job_id: jobId }] }));
        }),
        rest.get(`ajax-api/3.0/jobs/${jobId}`, (_req, res, ctx) => {
          return res(
            ctx.json({
              status: 'SUCCEEDED',
              result: {
                [traceId]: {
                  assessments: [],
                  failures: [{ error_code: 'SCORER_ERROR', error_message: 'Scorer failed to process trace' }],
                },
              },
            }),
          );
        }),
      );

      setupTraceFetchHandlers(server, traces);

      const { result } = renderHook(() => useEvaluateTracesAsync({}), {
        wrapper,
      });

      act(() => {
        const [evaluateTracesFn] = result.current;
        evaluateTracesFn({
          itemIds: ['trace-1'],
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          experimentId,
          judgeInstructions: 'Test',
          serializedScorer,
        });
      });

      await waitFor(() => {
        expect(result.current[1].latestEvaluation).not.toBeNull();
        expect(result.current[1].isLoading).toBe(false);
      });

      const [, state] = result.current;

      // Job succeeded overall
      expect(state.error).toBeNull();
      // But individual trace has failure error
      expect(state.latestEvaluation).toHaveLength(1);
      expect(state.latestEvaluation?.[0].error).toBe('Scorer failed to process trace');
      expect(state.latestEvaluation?.[0].results).toEqual([]);
    });
  });

  describe('Polling Behavior', () => {
    it('should stop polling when job completes', async () => {
      const traceId = 'trace-1';
      const experimentId = 'exp-123';
      const jobId = 'job-poll-stop';
      const serializedScorer = 'mock-serialized-scorer';

      const mockTrace = createMockTrace(traceId);
      const traces = new Map([[traceId, mockTrace]]);

      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/search', (_req, res, ctx) => {
          return res(ctx.json({ traces: [{ trace_id: traceId }], next_page_token: undefined }));
        }),
        rest.post('ajax-api/3.0/mlflow/scorer/invoke', (_req, res, ctx) => {
          return res(ctx.json({ jobs: [{ job_id: jobId }] }));
        }),
      );

      let pollCount = 0;
      server.use(
        rest.get(`ajax-api/3.0/jobs/${jobId}`, (_req, res, ctx) => {
          pollCount++;
          if (pollCount <= 2) {
            return res(ctx.json({ status: 'RUNNING' }));
          }
          return res(
            ctx.json({
              status: 'SUCCEEDED',
              result: { [traceId]: { assessments: [] } },
            }),
          );
        }),
      );

      // Setup trace fetching handlers using MSW
      setupTraceFetchHandlers(server, traces);

      const { result } = renderHook(() => useEvaluateTracesAsync({}), {
        wrapper,
      });

      await act(async () => {
        const [evaluateTracesFn] = result.current;
        evaluateTracesFn({
          itemIds: ['trace-1'],
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          experimentId,
          judgeInstructions: 'Test',
          serializedScorer,
        });
      });

      await waitFor(() => {
        expect(result.current[1].latestEvaluation).not.toBeNull();
      });

      const pollCountAfterSuccess = pollCount;

      // Wait a bit more and verify polling has stopped
      act(() => {
        jest.advanceTimersByTime(10000);
      });

      // Poll count should not increase significantly after job succeeded
      // Allow for one additional poll that might have been in-flight
      expect(pollCount).toBeLessThanOrEqual(pollCountAfterSuccess + 1);
    });
  });

  describe('Multiple evaluations', () => {
    it('should track multiple evaluations in evaluations and return latest evaluation', async () => {
      const experimentId = 'exp-123';
      const serializedScorer = 'mock-serialized-scorer';

      const trace1 = createMockTrace('trace-1');
      const traces = new Map([['trace-1', trace1]]);

      // Setup handlers
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/search', (_req, res, ctx) => {
          return res(ctx.json({ traces: [{ trace_id: 'trace-1' }], next_page_token: undefined }));
        }),
        rest.post('ajax-api/3.0/mlflow/scorer/invoke', (_req, res, ctx) => {
          return res(ctx.json({ jobs: [{ job_id: 'job-multi' }] }));
        }),
        rest.get('ajax-api/3.0/jobs/job-multi', (_req, res, ctx) => {
          return res(
            ctx.json({
              status: 'SUCCEEDED',
              result: { 'trace-1': { assessments: [{ assessment_name: 'test', numeric_value: 1 }] } },
            }),
          );
        }),
      );

      setupTraceFetchHandlers(server, traces);

      const { result } = renderHook(() => useEvaluateTracesAsync({}), { wrapper });

      // Start first evaluation
      await act(async () => {
        const [evaluateFunction] = result.current;
        await evaluateFunction({
          itemIds: ['trace-1'],
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          experimentId,
          judgeInstructions: 'Test 1',
          serializedScorer,
        });
      });

      // Wait for first evaluation to complete
      await waitFor(() => result.current[1].latestEvaluation !== null);

      // Verify first evaluation is tracked
      expect(Object.keys(result.current[1].allEvaluations).length).toBe(1);

      // Start second evaluation
      await act(async () => {
        const [evaluateFunction] = result.current;
        await evaluateFunction({
          itemIds: ['trace-1'],
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          experimentId,
          judgeInstructions: 'Test 2',
          serializedScorer,
        });
      });

      // Wait for second evaluation to also complete
      await waitFor(() => {
        const keys = Object.keys(result.current[1].allEvaluations);
        return keys.length === 2;
      });

      const [, state] = result.current;

      // Verify evaluations contains both evaluation requests
      expect(Object.keys(state.allEvaluations).length).toBe(2);

      // Verify each evaluation is tracked with its own state
      const evaluations = Object.values(state.allEvaluations);
      expect(evaluations).toHaveLength(2);

      // Both evaluations should have unique request keys
      const requestKeys = evaluations.map((ev) => ev.requestKey);
      expect(new Set(requestKeys).size).toBe(2);

      // The latest evaluation's data should be in `data`
      expect(state.latestEvaluation).toBeDefined();
    });
  });

  describe('Partial errors', () => {
    it('should handle partial success when some jobs succeed and others fail', async () => {
      const experimentId = 'exp-123';
      const serializedScorer = 'mock-serialized-scorer';

      const trace1 = createMockTrace('trace-1');
      const trace2 = createMockTrace('trace-2');
      const traces = new Map([
        ['trace-1', trace1],
        ['trace-2', trace2],
      ]);

      // Setup trace search handler
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/search', (_req, res, ctx) => {
          return res(
            ctx.json({
              traces: [{ trace_id: 'trace-1' }, { trace_id: 'trace-2' }],
              next_page_token: undefined,
            }),
          );
        }),
      );

      // Setup scorer invoke handler - returns two jobs
      server.use(
        rest.post('ajax-api/3.0/mlflow/scorer/invoke', (_req, res, ctx) => {
          return res(ctx.json({ jobs: [{ job_id: 'job-success' }, { job_id: 'job-fail' }] }));
        }),
      );

      // Setup job status handlers - one succeeds, one fails
      server.use(
        rest.get('ajax-api/3.0/jobs/job-success', (_req, res, ctx) => {
          return res(
            ctx.json({
              status: 'SUCCEEDED',
              result: {
                'trace-1': { assessments: [{ assessment_name: 'scorer', numeric_value: 0.9 }] },
              },
            }),
          );
        }),
        rest.get('ajax-api/3.0/jobs/job-fail', (_req, res, ctx) => {
          return res(
            ctx.json({
              status: 'FAILED',
              result: 'Failed to evaluate trace-2: Model endpoint unavailable',
            }),
          );
        }),
      );

      setupTraceFetchHandlers(server, traces);

      const onScorerFinished = jest.fn();
      const { result } = renderHook(() => useEvaluateTracesAsync({ onScorerFinished }), { wrapper });

      // Start evaluation
      act(() => {
        const [evaluateFunction] = result.current;
        evaluateFunction({
          itemIds: ['trace-1', 'trace-2'],
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          experimentId,
          judgeInstructions: 'Test',
          serializedScorer,
        });
      });

      // Wait for evaluation to complete with results
      await waitFor(() => {
        expect(result.current[1].isLoading).toBe(false);
        expect(result.current[1].latestEvaluation).not.toBeNull();
      });

      const [, state] = result.current;

      // Overall status should be SUCCEEDED (at least one job succeeded)
      expect(state.error).toBeNull();
      expect(state.latestEvaluation).toHaveLength(2);

      // trace-1 should have successful assessment
      const trace1Result = state.latestEvaluation?.find(
        (r) => !isSessionJudgeEvaluationResult(r) && (r.trace?.info as ModelTraceInfoV3)?.trace_id === 'trace-1',
      );
      expect(trace1Result).toBeDefined();
      expect(trace1Result?.results).toHaveLength(1);
      expect((trace1Result?.results[0] as { numeric_value?: number }).numeric_value).toBe(0.9);
      expect(trace1Result?.error).toBeNull();

      // trace-2 should have error message from failed job
      const trace2Result = state.latestEvaluation?.find(
        (r) => !isSessionJudgeEvaluationResult(r) && (r.trace?.info as ModelTraceInfoV3)?.trace_id === 'trace-2',
      );
      expect(trace2Result).toBeDefined();
      expect(trace2Result?.results).toEqual([]);
      expect(trace2Result?.error).toBe('Failed to evaluate trace-2: Model endpoint unavailable');

      // Callback should report SUCCEEDED (partial success is still success)
      expect(onScorerFinished).toHaveBeenCalledWith(expect.objectContaining({ status: 'SUCCEEDED' }));
    });

    it('should include failed job per-trace data alongside successful job results', async () => {
      const experimentId = 'exp-123';
      const serializedScorer = 'mock-serialized-scorer';

      const trace1 = createMockTrace('trace-1');
      const trace2 = createMockTrace('trace-2');
      const trace3 = createMockTrace('trace-3');
      const traces = new Map([
        ['trace-1', trace1],
        ['trace-2', trace2],
        ['trace-3', trace3],
      ]);

      // Setup trace search handler
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/search', (_req, res, ctx) => {
          return res(
            ctx.json({
              traces: [{ trace_id: 'trace-1' }, { trace_id: 'trace-2' }, { trace_id: 'trace-3' }],
              next_page_token: undefined,
            }),
          );
        }),
      );

      // Setup scorer invoke handler - returns two jobs
      server.use(
        rest.post('ajax-api/3.0/mlflow/scorer/invoke', (_req, res, ctx) => {
          return res(ctx.json({ jobs: [{ job_id: 'job-ok' }, { job_id: 'job-partial-fail' }] }));
        }),
      );

      // Setup job status handlers
      server.use(
        // First job succeeds with trace-1 only
        rest.get('ajax-api/3.0/jobs/job-ok', (_req, res, ctx) => {
          return res(
            ctx.json({
              status: 'SUCCEEDED',
              result: {
                'trace-1': { assessments: [{ assessment_name: 'scorer', numeric_value: 0.95 }] },
              },
            }),
          );
        }),
        // Second job fails but has per-trace results for trace-2 and trace-3
        rest.get('ajax-api/3.0/jobs/job-partial-fail', (_req, res, ctx) => {
          return res(
            ctx.json({
              status: 'FAILED',
              result: {
                'trace-2': {
                  assessments: [{ assessment_name: 'scorer', numeric_value: 0.8 }],
                },
                'trace-3': {
                  assessments: [],
                  failures: [{ error_code: 'SCORER_ERROR', error_message: 'Model timeout on trace-3' }],
                },
              },
            }),
          );
        }),
      );

      setupTraceFetchHandlers(server, traces);

      const { result } = renderHook(() => useEvaluateTracesAsync({}), { wrapper });

      // Start evaluation
      act(() => {
        const [evaluateFunction] = result.current;
        evaluateFunction({
          itemIds: ['trace-1', 'trace-2', 'trace-3'],
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          experimentId,
          judgeInstructions: 'Test',
          serializedScorer,
        });
      });

      // Wait for evaluation to complete - status is SUCCEEDED because at least one job succeeded
      await waitFor(() => {
        expect(result.current[1].isLoading).toBe(false);
        expect(result.current[1].latestEvaluation).not.toBeNull();
      });

      const [, state] = result.current;

      // Should have results from both jobs
      expect(state.latestEvaluation).toHaveLength(3);
      expect(state.error).toBeNull();

      // trace-1 should have assessment from successful job
      const trace1Result = state.latestEvaluation?.find(
        (r) => !isSessionJudgeEvaluationResult(r) && (r.trace?.info as ModelTraceInfoV3)?.trace_id === 'trace-1',
      );
      expect(trace1Result?.results).toHaveLength(1);
      expect((trace1Result?.results[0] as { numeric_value?: number }).numeric_value).toBe(0.95);

      // trace-2 should have assessment from failed job's per-trace data
      const trace2Result = state.latestEvaluation?.find(
        (r) => !isSessionJudgeEvaluationResult(r) && (r.trace?.info as ModelTraceInfoV3)?.trace_id === 'trace-2',
      );
      expect(trace2Result?.results).toHaveLength(1);
      expect((trace2Result?.results[0] as { numeric_value?: number }).numeric_value).toBe(0.8);

      // trace-3 should have error from failed job's failures array
      const trace3Result = state.latestEvaluation?.find(
        (r) => !isSessionJudgeEvaluationResult(r) && (r.trace?.info as ModelTraceInfoV3)?.trace_id === 'trace-3',
      );
      expect(trace3Result?.results).toEqual([]);
      expect(trace3Result?.error).toBe('Model timeout on trace-3');
    });
  });

  describe('Session Level Evaluation', () => {
    it('should successfully evaluate traces grouped by session and return SessionJudgeEvaluationResult', async () => {
      const sessionId = 'session-123';
      const traceId1 = 'trace-1';
      const traceId2 = 'trace-2';
      const experimentId = 'exp-123';
      const jobId = 'job-session';
      const serializedScorer = 'mock-serialized-scorer';

      // Create traces that belong to the same session
      const mockTrace1 = createMockTrace(traceId1, sessionId);
      const mockTrace2 = createMockTrace(traceId2, sessionId);
      const traces = new Map([
        [traceId1, mockTrace1],
        [traceId2, mockTrace2],
      ]);

      // Setup trace search handler - returns traces with session metadata
      server.use(
        rest.post('ajax-api/3.0/mlflow/traces/search', (_req, res, ctx) => {
          return res(
            ctx.json({
              traces: [
                { trace_id: traceId1, trace_metadata: { 'mlflow.trace.session': sessionId } },
                { trace_id: traceId2, trace_metadata: { 'mlflow.trace.session': sessionId } },
              ],
              next_page_token: undefined,
            }),
          );
        }),
      );

      // Setup scorer invoke handler
      server.use(
        rest.post('ajax-api/3.0/mlflow/scorer/invoke', (_req, res, ctx) => {
          return res(ctx.json({ jobs: [{ job_id: jobId }] }));
        }),
      );

      // Setup job status handler - returns assessments for both traces
      server.use(
        rest.get(`ajax-api/3.0/jobs/${jobId}`, (_req, res, ctx) => {
          return res(
            ctx.json({
              status: 'SUCCEEDED',
              result: {
                [traceId1]: {
                  assessments: [
                    {
                      assessment_name: 'session_scorer',
                      numeric_value: 0.8,
                      rationale: 'First turn good',
                    },
                  ],
                },
                [traceId2]: {
                  assessments: [
                    {
                      assessment_name: 'session_scorer',
                      numeric_value: 0.9,
                      rationale: 'Second turn excellent',
                    },
                  ],
                },
              },
            }),
          );
        }),
      );

      // Setup trace fetching handlers
      setupTraceFetchHandlers(server, traces);

      const onScorerFinished = jest.fn();

      const { result } = renderHook(
        () =>
          useEvaluateTracesAsync({
            onScorerFinished,
          }),
        { wrapper },
      );

      // Start evaluation
      act(() => {
        const [evaluateFunction] = result.current;
        evaluateFunction({
          itemIds: [sessionId],
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          experimentId,
          judgeInstructions: 'Evaluate the session',
          evaluationScope: ScorerEvaluationScope.SESSIONS,
          serializedScorer,
        });
      });

      // Wait for job to complete and results to be available
      await waitFor(() => {
        expect(result.current[1].isLoading).toBe(false);
        expect(result.current[1].latestEvaluation).not.toBeNull();
      });

      const [, finalState] = result.current;

      // Verify final state - should have one session result
      expect(finalState.latestEvaluation).toHaveLength(1);

      const sessionResult = finalState.latestEvaluation?.[0] as JudgeEvaluationResult;

      // Verify it's a SessionJudgeEvaluationResult
      expect(isSessionJudgeEvaluationResult(sessionResult)).toBe(true);

      if (isSessionJudgeEvaluationResult(sessionResult)) {
        // Verify session ID
        expect(sessionResult.sessionId).toBe(sessionId);

        // Verify traces array contains both traces from the session
        expect(sessionResult.traces).toHaveLength(2);
        expect(sessionResult.traces?.map((t) => (t.info as ModelTraceInfoV3).trace_id)).toContain(traceId1);
        expect(sessionResult.traces?.map((t) => (t.info as ModelTraceInfoV3).trace_id)).toContain(traceId2);

        // Verify aggregated results from all traces in the session
        expect(sessionResult.results).toHaveLength(2);
        expect(sessionResult.results).toContainEqual(
          expect.objectContaining({
            assessment_name: 'session_scorer',
            numeric_value: 0.8,
            rationale: 'First turn good',
          }),
        );
        expect(sessionResult.results).toContainEqual(
          expect.objectContaining({
            assessment_name: 'session_scorer',
            numeric_value: 0.9,
            rationale: 'Second turn excellent',
          }),
        );

        expect(sessionResult.error).toBeNull();
      }

      expect(finalState.error).toBeNull();
      expect(onScorerFinished).toHaveBeenCalledWith(expect.objectContaining({ status: 'SUCCEEDED' }));
    });
  });
});
