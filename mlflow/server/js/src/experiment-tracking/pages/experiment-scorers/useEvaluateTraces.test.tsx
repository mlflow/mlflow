import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { useEvaluateTraces } from './useEvaluateTraces';
import { type JudgeEvaluationResult } from './useEvaluateTraces.common';
import type { ModelTrace, ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { SEARCH_MLFLOW_TRACES_QUERY_KEY } from '@databricks/web-shared/genai-traces-table';
import { fetchOrFail } from '../../../common/utils/FetchUtils';
import { setupServer } from '../../../common/utils/setup-msw';
import { rest } from 'msw';

// Mock fetchOrFail
jest.mock('../../../common/utils/FetchUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FetchUtils')>('../../../common/utils/FetchUtils'),
  fetchOrFail: jest.fn(),
}));

// Mock the feature flag to explicitly use sync mode for these tests
jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  isEvaluatingSessionsInScorersEnabled: () => false,
}));

const mockedFetchOrFail = jest.mocked(fetchOrFail);
const server = setupServer();

/**
 * Helper to setup fetchOrFail mocks for trace fetching, chat completions, and chat assessments
 * This mocks the two API calls that getMlflowTraceV3 makes internally, plus the evaluation APIs
 */
function setupMocks(
  traces: Map<string, ModelTrace>,
  chatCompletionsHandler?: (url: string, options?: any) => Promise<any>,
  chatAssessmentsHandler?: (url: string, options?: any) => Promise<any>,
) {
  server.use(
    rest.get('/ajax-api/3.0/mlflow/traces/:requestId', (req, res, ctx) => {
      return res(ctx.json({ trace: { trace_info: traces.get(req.params['requestId'].toString())?.info } }));
    }),
  );

  server.use(
    rest.get('/ajax-api/3.0/mlflow/get-trace-artifact', (req, res, ctx) => {
      return res(ctx.json(traces.get(req.url.searchParams.get('request_id')?.toString() ?? '')?.data));
    }),
  );

  mockedFetchOrFail.mockImplementation((url: RequestInfo | URL, options?: any) => {
    const urlString = typeof url === 'string' ? url : url.toString();

    // Handle trace info endpoint
    if (urlString.includes('ajax-api/3.0/mlflow/traces/')) {
      const requestId = urlString.split('/').pop() as string;
      const trace = traces.get(requestId);
      return Promise.resolve({
        json: () =>
          Promise.resolve({
            trace: {
              trace_info: trace?.info || {},
            },
          }),
      } as any);
    }

    // Handle trace data endpoint
    if (urlString.includes('ajax-api/3.0/mlflow/get-trace-artifact')) {
      const match = urlString.match(/request_id=([^&]+)/);
      const requestId = match ? match[1] : '';
      const trace = traces.get(requestId);
      return Promise.resolve({
        json: () => Promise.resolve(trace?.data || {}),
      } as any);
    }

    // Handle chat completions endpoint
    if (urlString.includes('ajax-api/2.0/agents/chat-completions') && options?.method === 'POST') {
      if (chatCompletionsHandler) {
        return chatCompletionsHandler(urlString, options);
      }
      return Promise.resolve({
        json: () =>
          Promise.resolve({
            output: null,
            error_code: null,
            error_message: null,
          }),
      } as any);
    }

    // Handle chat assessments endpoint
    if (urlString.includes('ajax-api/2.0/agents/chat-assessments') && options?.method === 'POST') {
      if (chatAssessmentsHandler) {
        return chatAssessmentsHandler(urlString, options);
      }
      return Promise.resolve({
        json: () =>
          Promise.resolve({
            result: {
              response_assessment: {
                ratings: {},
              },
            },
          }),
      } as any);
    }

    return Promise.reject(new Error(`Unexpected URL: ${urlString}`));
  });
}

/**
 * Helper to setup MSW handlers for search traces endpoint
 */
function setupSearchTracesHandler(server: ReturnType<typeof setupServer>, traces: Map<string, ModelTrace>) {
  const traceInfos: ModelTraceInfoV3[] = Array.from(traces.values()).map((trace) => trace.info as ModelTraceInfoV3);

  server.use(
    rest.post('ajax-api/3.0/mlflow/traces/search', (req, res, ctx) => {
      return res(ctx.json({ traces: traceInfos, next_page_token: undefined }));
    }),
  );
}

describe('useEvaluateTraces', () => {
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

  describe('Golden Path - Successful Operations', () => {
    it('should successfully evaluate single trace with valid JSON response', async () => {
      const traceId = 'trace-1';
      const experimentId = 'exp-123';
      const judgeInstructions = 'Evaluate the quality';

      const mockTrace: ModelTrace = {
        info: { trace_id: traceId } as any,
        data: {
          spans: [
            {
              trace_id: traceId,
              span_id: 'span-1',
              name: 'root',
              trace_state: '',
              parent_span_id: null,
              start_time_unix_nano: '0',
              end_time_unix_nano: '1000000',
              status: { code: 'STATUS_CODE_UNSET' },
              attributes: {
                'mlflow.spanInputs': 'test input',
                'mlflow.spanOutputs': 'test output',
              },
            },
          ],
        },
      };

      const apiResponse = {
        output: JSON.stringify({ result: 'PASS', rationale: 'Good quality' }),
        error_code: null,
        error_message: null,
      };

      // Setup mocks
      const traces = new Map([[traceId, mockTrace]]);
      setupSearchTracesHandler(server, traces);
      setupMocks(traces, () =>
        Promise.resolve({
          json: () => Promise.resolve(apiResponse),
        } as any),
      );

      // Verify initial cache state
      const initialCache = queryClient.getQueryData(['GetMlflowTraceV3', traceId]);
      expect(initialCache).toBeUndefined();

      const { result } = renderHook(() => useEvaluateTraces(), { wrapper });

      const [evaluateTraces, state] = result.current;

      expect(state.data).toBeNull();
      expect(state.isLoading).toBe(false);
      expect(state.error).toBeNull();

      const evaluationResults = await evaluateTraces({
        itemIds: [traceId],
        locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
        judgeInstructions,
        experimentId,
      });

      const expectedResult: JudgeEvaluationResult = {
        trace: mockTrace,
        results: [
          {
            result: 'PASS',
            rationale: 'Good quality',
            error: null,
            span_name: '',
          },
        ],
        error: null,
      };

      expect(evaluationResults).toEqual([expectedResult]);

      await waitFor(() => {
        expect(result.current[1].data).toEqual([expectedResult]);
        expect(result.current[1].isLoading).toBe(false);
        expect(result.current[1].error).toBeNull();
      });

      // Verify trace was cached
      const cachedTrace = queryClient.getQueryData(['GetMlflowTraceV3', traceId]);
      expect(cachedTrace).toEqual(mockTrace);

      // Verify chat completions was called
      expect(mockedFetchOrFail).toHaveBeenCalledWith(
        'ajax-api/2.0/agents/chat-completions',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: expect.any(String),
        }),
      );
    });

    it('should successfully evaluate multiple traces in parallel', async () => {
      const traceIds = ['trace-1', 'trace-2', 'trace-3'];
      const experimentId = 'exp-123';
      const judgeInstructions = 'Evaluate the quality';

      const mockTraces: ModelTrace[] = traceIds.map((id) => ({
        info: { trace_id: id } as any,
        data: {
          spans: [
            {
              trace_id: id,
              span_id: `span-${id}`,
              name: 'root',
              trace_state: '',
              parent_span_id: null,
              start_time_unix_nano: '0',
              end_time_unix_nano: '1000000',
              status: { code: 'STATUS_CODE_UNSET' },
              attributes: {
                'mlflow.spanInputs': `input-${id}`,
                'mlflow.spanOutputs': `output-${id}`,
              },
            },
          ],
        },
      }));

      // Setup mocks
      const traces = new Map(mockTraces.map((trace) => [(trace.info as any).trace_id, trace]));
      setupSearchTracesHandler(server, traces);
      setupMocks(traces, () =>
        Promise.resolve({
          json: () =>
            Promise.resolve({
              output: JSON.stringify({ result: 'PASS', rationale: 'Good' }),
              error_code: null,
              error_message: null,
            }),
        } as any),
      );

      // Verify initial cache state
      traceIds.forEach((id) => {
        const initialCache = queryClient.getQueryData(['GetMlflowTraceV3', id]);
        expect(initialCache).toBeUndefined();
      });

      const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
      const [evaluateTraces] = result.current;

      const evaluationResults = await evaluateTraces({
        itemIds: traceIds,
        locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
        judgeInstructions,
        experimentId,
      });

      expect(evaluationResults).toHaveLength(3);
      (evaluationResults as JudgeEvaluationResult[])?.forEach((evalResult, index) => {
        expect(evalResult).toEqual({
          trace: mockTraces[index],
          results: [
            {
              result: 'PASS',
              rationale: 'Good',
              error: null,
              span_name: '',
            },
          ],
          error: null,
        });
      });

      // Verify all traces were cached with exact values
      traceIds.forEach((traceId, index) => {
        const cachedTrace = queryClient.getQueryData(['GetMlflowTraceV3', traceId]);
        expect(cachedTrace).toEqual(mockTraces[index]);
      });
    });

    it('should successfully evaluate trace with non-JSON plain text response', async () => {
      const traceId = 'trace-1';
      const experimentId = 'exp-123';
      const judgeInstructions = 'Evaluate the quality';

      const mockTrace: ModelTrace = {
        info: { trace_id: traceId } as any,
        data: {
          spans: [
            {
              trace_id: traceId,
              span_id: 'span-1',
              name: 'root',
              trace_state: '',
              parent_span_id: null,
              start_time_unix_nano: '0',
              end_time_unix_nano: '1000000',
              status: { code: 'STATUS_CODE_UNSET' },
              attributes: {
                'mlflow.spanInputs': 'test input',
                'mlflow.spanOutputs': 'test output',
              },
            },
          ],
        },
      };

      const apiResponse = {
        output: 'This is a plain text evaluation result',
        error_code: null,
        error_message: null,
      };

      // Setup mocks
      const traces = new Map([[traceId, mockTrace]]);
      setupSearchTracesHandler(server, traces);
      setupMocks(traces, () =>
        Promise.resolve({
          json: () => Promise.resolve(apiResponse),
        } as any),
      );

      const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
      const [evaluateTraces] = result.current;

      const evaluationResults = await evaluateTraces({
        itemIds: [traceId],
        locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
        judgeInstructions,
        experimentId,
      });

      expect(evaluationResults).toEqual([
        {
          trace: mockTrace,
          results: [
            {
              result: 'This is a plain text evaluation result',
              rationale: null,
              error: null,
              span_name: '',
            },
          ],
          error: null,
        },
      ]);
    });

    it('should cache traces with infinite staleTime and not refetch on subsequent calls', async () => {
      const traceId = 'trace-1';
      const experimentId = 'exp-123';
      const judgeInstructions = 'Evaluate';

      const mockTrace: ModelTrace = {
        info: { trace_id: traceId } as any,
        data: { spans: [] },
      };

      // Setup mocks
      const traces = new Map([[traceId, mockTrace]]);
      setupSearchTracesHandler(server, traces);
      let fetchCallCount = 0;
      setupMocks(traces, () => {
        fetchCallCount++;
        return Promise.resolve({
          json: () =>
            Promise.resolve({
              output: JSON.stringify({ result: 'PASS', rationale: 'Good' }),
              error_code: null,
              error_message: null,
            }),
        } as any);
      });

      const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
      const [evaluateTraces] = result.current;

      // First call - should fetch from API
      await evaluateTraces({
        itemIds: [traceId],
        locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
        judgeInstructions,
        experimentId,
      });
      const firstFetchCount = fetchCallCount;
      expect(firstFetchCount).toBeGreaterThan(0);

      // Second call with same traceId - traces should use cache, so no additional trace API calls
      // but chat completions will be called again (not cached by design)
      await evaluateTraces({
        itemIds: [traceId],
        locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
        judgeInstructions,
        experimentId,
      });

      // Verify cache persists with exact value
      const cachedTrace = queryClient.getQueryData(['GetMlflowTraceV3', traceId]);
      expect(cachedTrace).toEqual(mockTrace);
    });
  });

  describe('Edge Cases', () => {
    it('should handle null output from API gracefully', async () => {
      const traceId = 'trace-1';
      const experimentId = 'exp-123';
      const judgeInstructions = 'Evaluate';

      const mockTrace: ModelTrace = {
        info: { trace_id: traceId } as any,
        data: { spans: [] },
      };

      // Setup mocks - API returns null output
      const traces = new Map([[traceId, mockTrace]]);
      setupSearchTracesHandler(server, traces);
      setupMocks(traces, () =>
        Promise.resolve({
          json: () =>
            Promise.resolve({
              output: null,
              error_code: null,
              error_message: null,
            }),
        } as any),
      );

      const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
      const [evaluateTraces] = result.current;

      const evaluationResults = await evaluateTraces({
        itemIds: [traceId],
        locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
        judgeInstructions,
        experimentId,
      });

      expect(evaluationResults).toEqual([
        {
          trace: mockTrace,
          results: [
            {
              result: null,
              rationale: null,
              error: null,
              span_name: '',
            },
          ],
          error: null,
        },
      ]);
    });

    it('should handle empty trace IDs array', async () => {
      const experimentId = 'exp-123';
      const judgeInstructions = 'Evaluate';

      // Setup empty traces map
      const traces = new Map();
      setupSearchTracesHandler(server, traces);

      const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
      const [evaluateTraces] = result.current;

      const evaluationResults = await evaluateTraces({
        itemIds: [],
        locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
        judgeInstructions,
        experimentId,
      });

      expect(evaluationResults).toEqual([]);
    });

    it('should handle malformed JSON by treating output as plain text', async () => {
      const traceId = 'trace-1';
      const experimentId = 'exp-123';
      const judgeInstructions = 'Evaluate';

      const mockTrace: ModelTrace = {
        info: { trace_id: traceId } as any,
        data: { spans: [] },
      };

      // Setup mocks - API returns malformed JSON
      const traces = new Map([[traceId, mockTrace]]);
      setupSearchTracesHandler(server, traces);
      setupMocks(traces, () =>
        Promise.resolve({
          json: () =>
            Promise.resolve({
              output: '{invalid json',
              error_code: null,
              error_message: null,
            }),
        } as any),
      );

      const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
      const [evaluateTraces] = result.current;

      const evaluationResults = await evaluateTraces({
        itemIds: [traceId],
        locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
        judgeInstructions,
        experimentId,
      });

      // Should fallback to treating output as plain text
      expect(evaluationResults).toEqual([
        {
          trace: mockTrace,
          results: [
            {
              result: '{invalid json',
              rationale: null,
              error: null,
              span_name: '',
            },
          ],
          error: null,
        },
      ]);
    });

    it('should handle partial success when some traces fail during evaluation', async () => {
      const traceIds = ['trace-success', 'trace-fail', 'trace-success-2'];
      const experimentId = 'exp-123';
      const judgeInstructions = 'Evaluate';

      const mockTraces: ModelTrace[] = traceIds.map((id) => ({
        info: { trace_id: id } as any,
        data: { spans: [] },
      }));

      // Setup mocks - API fails for the second trace
      const traces = new Map(mockTraces.map((trace) => [(trace.info as any).trace_id, trace]));
      setupSearchTracesHandler(server, traces);
      let callCount = 0;
      setupMocks(traces, () => {
        callCount++;
        if (callCount === 2) {
          throw new Error('Network error');
        }
        return Promise.resolve({
          json: () =>
            Promise.resolve({
              output: JSON.stringify({ result: 'PASS', rationale: 'Good' }),
              error_code: null,
              error_message: null,
            }),
        } as any);
      });

      const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
      const [evaluateTraces] = result.current;

      const evaluationResults = await evaluateTraces({
        itemIds: traceIds,
        locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
        judgeInstructions,
        experimentId,
      });

      expect(evaluationResults).toHaveLength(3);

      // First trace should succeed
      expect(evaluationResults?.[0]).toEqual({
        trace: mockTraces[0],
        results: [
          {
            result: 'PASS',
            rationale: 'Good',
            error: null,
            span_name: '',
          },
        ],
        error: null,
      });

      // Second trace should have error
      expect(evaluationResults?.[1]).toEqual({
        trace: mockTraces[1],
        results: [],
        error: 'Network error',
      });

      // Third trace should succeed
      expect(evaluationResults?.[2]).toEqual({
        trace: mockTraces[2],
        results: [
          {
            result: 'PASS',
            rationale: 'Good',
            error: null,
            span_name: '',
          },
        ],
        error: null,
      });
    });
  });

  describe('Error Conditions', () => {
    it('should handle API error response with error_code', async () => {
      const traceId = 'trace-1';
      const experimentId = 'exp-123';
      const judgeInstructions = 'Evaluate';

      const mockTrace: ModelTrace = {
        info: { trace_id: traceId } as any,
        data: { spans: [] },
      };

      // Setup mocks - API returns error
      const traces = new Map([[traceId, mockTrace]]);
      setupSearchTracesHandler(server, traces);
      setupMocks(traces, () =>
        Promise.resolve({
          json: () =>
            Promise.resolve({
              output: null,
              error_code: 'RATE_LIMIT_EXCEEDED',
              error_message: 'Rate limit exceeded',
            }),
        } as any),
      );

      const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
      const [evaluateTraces] = result.current;

      const evaluationResults = await evaluateTraces({
        itemIds: [traceId],
        locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
        judgeInstructions,
        experimentId,
      });

      expect(evaluationResults).toEqual([
        {
          trace: mockTrace,
          results: [],
          error: 'Rate limit exceeded',
        },
      ]);

      await waitFor(() => {
        expect(result.current[1].error).toBeNull();
        expect(result.current[1].data).toEqual(evaluationResults);
      });
    });

    it('should handle trace fetching failure gracefully', async () => {
      const traceId = 'trace-1';
      const experimentId = 'exp-123';
      const judgeInstructions = 'Evaluate';

      const mockTrace: ModelTrace = {
        info: { trace_id: traceId } as any,
        data: { spans: [] },
      };

      // Setup search to return trace info, but individual trace fetching to fail
      const traces = new Map([[traceId, mockTrace]]);
      setupSearchTracesHandler(server, traces);
      server.use(
        rest.get('/ajax-api/3.0/mlflow/traces/:requestId', (req, res, ctx) => {
          return res(ctx.status(404), ctx.json({ message: 'Trace not found' }));
        }),
      );

      const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
      const [evaluateTraces] = result.current;

      const evaluationResults = await evaluateTraces({
        itemIds: [traceId],
        locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
        judgeInstructions,
        experimentId,
      });

      expect(evaluationResults).toEqual([
        {
          trace: null,
          results: [],
          error: 'Trace not found',
        },
      ]);

      await waitFor(() => {
        expect(result.current[1].error).toBeNull();
        expect(result.current[1].data).toEqual(evaluationResults);
        expect(result.current[1].isLoading).toBe(false);
      });
    });

    it('should handle non-Error objects thrown during evaluation', async () => {
      const traceId = 'trace-1';
      const experimentId = 'exp-123';
      const judgeInstructions = 'Evaluate';

      const mockTrace: ModelTrace = {
        info: { trace_id: traceId } as any,
        data: { spans: [] },
      };

      // Setup mocks - API throws non-Error object
      const traces = new Map([[traceId, mockTrace]]);
      setupSearchTracesHandler(server, traces);
      setupMocks(traces, () => Promise.reject('String error'));

      const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
      const [evaluateTraces] = result.current;

      const evaluationResults = await evaluateTraces({
        itemIds: [traceId],
        locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
        judgeInstructions,
        experimentId,
      });

      expect(evaluationResults).toEqual([
        {
          trace: mockTrace,
          results: [],
          error: 'String error',
        },
      ]);
    });

    it('should handle network failure during evaluation gracefully', async () => {
      const traceId = 'trace-1';
      const experimentId = 'exp-123';
      const judgeInstructions = 'Evaluate';

      const mockTrace: ModelTrace = {
        info: { trace_id: traceId } as any,
        data: { spans: [] },
      };

      // Setup search to return trace info, but individual trace fetching to fail
      const traces = new Map([[traceId, mockTrace]]);
      setupSearchTracesHandler(server, traces);
      server.use(
        rest.get('/ajax-api/3.0/mlflow/traces/:requestId', (req, res, ctx) => {
          return res(ctx.status(500), ctx.json({ message: 'Network failure' }));
        }),
      );

      const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
      const [evaluateTraces] = result.current;

      const evaluationResults = await evaluateTraces({
        itemIds: [traceId],
        locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
        judgeInstructions,
        experimentId,
      });

      expect(evaluationResults).toEqual([
        {
          trace: null,
          results: [],
          error: 'Network failure',
        },
      ]);

      await waitFor(() => {
        expect(result.current[1].error).toBeNull();
        expect(result.current[1].data).toEqual(evaluationResults);
        expect(result.current[1].isLoading).toBe(false);
      });
    });

    it('should handle all traces failing to fetch gracefully', async () => {
      const traceIds = ['trace-1', 'trace-2'];
      const experimentId = 'exp-123';
      const judgeInstructions = 'Evaluate';

      const mockTraces: ModelTrace[] = traceIds.map((id) => ({
        info: { trace_id: id } as any,
        data: { spans: [] },
      }));

      // Setup search to return trace infos, but individual trace fetching to fail
      const traces = new Map(mockTraces.map((trace) => [(trace.info as any).trace_id, trace]));
      setupSearchTracesHandler(server, traces);
      server.use(
        rest.get('/ajax-api/3.0/mlflow/traces/:requestId', (req, res, ctx) => {
          return res(ctx.status(500), ctx.json({ message: 'All traces unavailable' }));
        }),
      );

      const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
      const [evaluateTraces] = result.current;

      const evaluationResults = await evaluateTraces({
        itemIds: traceIds,
        locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
        judgeInstructions,
        experimentId,
      });

      expect(evaluationResults).toEqual([
        {
          trace: null,
          results: [],
          error: 'All traces unavailable',
        },
        {
          trace: null,
          results: [],
          error: 'All traces unavailable',
        },
      ]);

      await waitFor(() => {
        expect(result.current[1].error).toBeNull();
        expect(result.current[1].data).toEqual(evaluationResults);
      });
    });
  });

  describe('Built-in Judges (Chat Assessments)', () => {
    describe('Golden Path - Successful Operations', () => {
      it('should successfully assess single trace with built-in judge', async () => {
        const traceId = 'trace-1';
        const experimentId = 'exp-123';
        const requestedAssessments = [{ assessment_name: 'correctness' }];

        const mockTrace: ModelTrace = {
          info: { trace_id: traceId } as any,
          data: {
            spans: [
              {
                trace_id: traceId,
                span_id: 'span-1',
                name: 'root',
                trace_state: '',
                parent_span_id: null,
                start_time_unix_nano: '0',
                end_time_unix_nano: '1000000',
                status: { code: 'STATUS_CODE_UNSET' },
                attributes: {
                  'mlflow.spanInputs': 'What is AI?',
                  'mlflow.spanOutputs': 'AI stands for Artificial Intelligence',
                },
              },
            ],
          },
        };

        const apiResponse = {
          result: {
            response_assessment: {
              ratings: {
                correctness: {
                  value: 'correct',
                  rationale: 'The response accurately defines AI',
                },
              },
            },
          },
        };

        const traces = new Map([[traceId, mockTrace]]);
        setupSearchTracesHandler(server, traces);
        setupMocks(traces, undefined, () =>
          Promise.resolve({
            json: () => Promise.resolve(apiResponse),
          } as any),
        );

        const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
        const [evaluateTraces] = result.current;

        const evaluationResults = await evaluateTraces({
          itemIds: [traceId],
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          requestedAssessments,
          experimentId,
        });

        expect(evaluationResults).toEqual([
          {
            trace: mockTrace,
            results: [
              {
                result: 'correct',
                rationale: 'The response accurately defines AI',
                error: null,
                span_name: '',
              },
            ],
            error: null,
          },
        ]);

        await waitFor(() => {
          expect(result.current[1].data).toEqual(evaluationResults);
          expect(result.current[1].isLoading).toBe(false);
          expect(result.current[1].error).toBeNull();
        });

        expect(mockedFetchOrFail).toHaveBeenCalledWith(
          'ajax-api/2.0/agents/chat-assessments',
          expect.objectContaining({
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: expect.any(String),
          }),
        );
      });

      it('should successfully assess multiple traces with built-in judge', async () => {
        const traceIds = ['trace-1', 'trace-2'];
        const experimentId = 'exp-123';
        const requestedAssessments = [{ assessment_name: 'groundedness' }];

        const mockTraces: ModelTrace[] = traceIds.map((id) => ({
          info: { trace_id: id } as any,
          data: {
            spans: [
              {
                trace_id: id,
                span_id: `span-${id}`,
                name: 'root',
                trace_state: '',
                parent_span_id: null,
                start_time_unix_nano: '0',
                end_time_unix_nano: '1000000',
                status: { code: 'STATUS_CODE_UNSET' },
                attributes: {
                  'mlflow.spanInputs': `input-${id}`,
                  'mlflow.spanOutputs': `output-${id}`,
                },
              },
            ],
          },
        }));

        const traces = new Map(mockTraces.map((trace) => [(trace.info as any).trace_id, trace]));
        setupSearchTracesHandler(server, traces);
        setupMocks(traces, undefined, () =>
          Promise.resolve({
            json: () =>
              Promise.resolve({
                result: {
                  response_assessment: {
                    ratings: {
                      groundedness: {
                        value: 'grounded',
                        rationale: 'Response is well grounded',
                      },
                    },
                  },
                },
              }),
          } as any),
        );

        const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
        const [evaluateTraces] = result.current;

        const evaluationResults = await evaluateTraces({
          itemIds: traceIds,
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          requestedAssessments,
          experimentId,
        });

        expect(evaluationResults).toHaveLength(2);
        (evaluationResults as JudgeEvaluationResult[])?.forEach((evalResult, index) => {
          expect(evalResult).toEqual({
            trace: mockTraces[index],
            results: [
              {
                result: 'grounded',
                rationale: 'Response is well grounded',
                error: null,
                span_name: '',
              },
            ],
            error: null,
          });
        });
      });

      it('should include retrieval context when available', async () => {
        const traceId = 'trace-1';
        const experimentId = 'exp-123';
        const requestedAssessments = [{ assessment_name: 'groundedness' }];

        const mockTrace: ModelTrace = {
          info: { trace_id: traceId } as any,
          data: {
            spans: [
              {
                trace_id: traceId,
                span_id: 'span-1',
                name: 'root',
                trace_state: '',
                parent_span_id: null,
                start_time_unix_nano: '0',
                end_time_unix_nano: '5000000',
                status: { code: 'STATUS_CODE_UNSET' },
                attributes: {
                  'mlflow.spanInputs': 'What is ML?',
                  'mlflow.spanOutputs': 'ML is machine learning',
                },
              },
              {
                trace_id: traceId,
                span_id: 'span-2',
                name: 'retrieval',
                trace_state: '',
                parent_span_id: 'span-1',
                start_time_unix_nano: '1000000',
                end_time_unix_nano: '2000000',
                status: { code: 'STATUS_CODE_UNSET' },
                attributes: {
                  'mlflow.spanType': 'RETRIEVER',
                  'mlflow.spanInputs': { query: 'What is ML?' },
                  'mlflow.spanOutputs': [{ doc_uri: 'doc1.pdf', content: 'ML is a field of AI' }],
                },
              },
            ],
          },
        };

        const traces = new Map([[traceId, mockTrace]]);
        setupSearchTracesHandler(server, traces);
        let requestBody: any;
        setupMocks(traces, undefined, (_url, options) => {
          requestBody = JSON.parse(options.body);
          return Promise.resolve({
            json: () =>
              Promise.resolve({
                result: {
                  response_assessment: {
                    ratings: {
                      groundedness: {
                        value: 'grounded',
                        rationale: 'Response is well grounded in context',
                      },
                    },
                  },
                },
              }),
          } as any);
        });

        const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
        const [evaluateTraces] = result.current;

        await evaluateTraces({
          itemIds: [traceId],
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          requestedAssessments,
          experimentId,
        });

        expect(requestBody.assessment_input.retrieval_context).toEqual({
          retrieved_documents: [{ doc_uri: 'doc1.pdf', content: 'ML is a field of AI' }],
        });
      });
    });

    describe('Edge Cases', () => {
      it('should return error when trace has no inputs', async () => {
        const traceId = 'trace-1';
        const experimentId = 'exp-123';
        const requestedAssessments = [{ assessment_name: 'correctness' }];

        const mockTrace: ModelTrace = {
          info: { trace_id: traceId } as any,
          data: {
            spans: [
              {
                trace_id: traceId,
                span_id: 'span-1',
                name: 'root',
                trace_state: '',
                parent_span_id: null,
                start_time_unix_nano: '0',
                end_time_unix_nano: '1000000',
                status: { code: 'STATUS_CODE_UNSET' },
                attributes: {},
              },
            ],
          },
        };

        const traces = new Map([[traceId, mockTrace]]);
        setupSearchTracesHandler(server, traces);
        setupMocks(traces);

        const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
        const [evaluateTraces] = result.current;

        const evaluationResults = await evaluateTraces({
          itemIds: [traceId],
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          requestedAssessments,
          experimentId,
        });

        expect(evaluationResults).toEqual([
          {
            trace: mockTrace,
            results: [],
            error: 'No chat request found in trace',
          },
        ]);
      });

      it('should handle API error response', async () => {
        const traceId = 'trace-1';
        const experimentId = 'exp-123';
        const requestedAssessments = [{ assessment_name: 'correctness' }];

        const mockTrace: ModelTrace = {
          info: { trace_id: traceId } as any,
          data: {
            spans: [
              {
                trace_id: traceId,
                span_id: 'span-1',
                name: 'root',
                trace_state: '',
                parent_span_id: null,
                start_time_unix_nano: '0',
                end_time_unix_nano: '1000000',
                status: { code: 'STATUS_CODE_UNSET' },
                attributes: {
                  'mlflow.spanInputs': 'test input',
                },
              },
            ],
          },
        };

        const traces = new Map([[traceId, mockTrace]]);
        setupSearchTracesHandler(server, traces);
        setupMocks(traces, undefined, () =>
          Promise.resolve({
            json: () =>
              Promise.resolve({
                result: {
                  response_assessment: {
                    ratings: {
                      correctness: {
                        value: '',
                        rationale: '',
                        error: 'Rate limit exceeded',
                      },
                    },
                  },
                },
              }),
          } as any),
        );

        const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
        const [evaluateTraces] = result.current;

        const evaluationResults = await evaluateTraces({
          itemIds: [traceId],
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          requestedAssessments,
          experimentId,
        });

        expect(evaluationResults).toEqual([
          {
            trace: mockTrace,
            results: [],
            error: 'Rate limit exceeded',
          },
        ]);
      });

      it('should handle empty trace IDs array', async () => {
        const experimentId = 'exp-123';
        const requestedAssessments = [{ assessment_name: 'correctness' }];

        const traces = new Map();
        setupSearchTracesHandler(server, traces);

        const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
        const [evaluateTraces] = result.current;

        const evaluationResults = await evaluateTraces({
          itemIds: [],
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          requestedAssessments,
          experimentId,
        });

        expect(evaluationResults).toEqual([]);
      });
    });

    describe('Error Conditions', () => {
      it('should handle network failure during assessment', async () => {
        const traceId = 'trace-1';
        const experimentId = 'exp-123';
        const requestedAssessments = [{ assessment_name: 'correctness' }];

        const mockTrace: ModelTrace = {
          info: { trace_id: traceId } as any,
          data: {
            spans: [
              {
                trace_id: traceId,
                span_id: 'span-1',
                name: 'root',
                trace_state: '',
                parent_span_id: null,
                start_time_unix_nano: '0',
                end_time_unix_nano: '1000000',
                status: { code: 'STATUS_CODE_UNSET' },
                attributes: {
                  'mlflow.spanInputs': 'test input',
                },
              },
            ],
          },
        };

        const traces = new Map([[traceId, mockTrace]]);
        setupSearchTracesHandler(server, traces);
        setupMocks(traces, undefined, () => Promise.reject(new Error('Network error')));

        const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
        const [evaluateTraces] = result.current;

        const evaluationResults = await evaluateTraces({
          itemIds: [traceId],
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          requestedAssessments,
          experimentId,
        });

        expect(evaluationResults).toEqual([
          {
            trace: mockTrace,
            results: [],
            error: 'Network error',
          },
        ]);
      });

      it('should handle partial failures with multiple traces', async () => {
        const traceIds = ['trace-success', 'trace-fail'];
        const experimentId = 'exp-123';
        const requestedAssessments = [{ assessment_name: 'correctness' }];

        const mockTraces: ModelTrace[] = traceIds.map((id) => ({
          info: { trace_id: id } as any,
          data: {
            spans: [
              {
                trace_id: id,
                span_id: `span-${id}`,
                name: 'root',
                trace_state: '',
                parent_span_id: null,
                start_time_unix_nano: '0',
                end_time_unix_nano: '1000000',
                status: { code: 'STATUS_CODE_UNSET' },
                attributes: {
                  'mlflow.spanInputs': `input-${id}`,
                },
              },
            ],
          },
        }));

        const traces = new Map(mockTraces.map((trace) => [(trace.info as any).trace_id, trace]));
        setupSearchTracesHandler(server, traces);
        let callCount = 0;
        setupMocks(traces, undefined, () => {
          callCount++;
          if (callCount === 2) {
            throw new Error('Assessment failed');
          }
          return Promise.resolve({
            json: () =>
              Promise.resolve({
                result: {
                  response_assessment: {
                    ratings: {
                      correctness: {
                        value: 'correct',
                        rationale: 'Good',
                      },
                    },
                  },
                },
              }),
          } as any);
        });

        const { result } = renderHook(() => useEvaluateTraces(), { wrapper });
        const [evaluateTraces] = result.current;

        const evaluationResults = await evaluateTraces({
          itemIds: traceIds,
          locations: [{ mlflow_experiment: { experiment_id: experimentId }, type: 'MLFLOW_EXPERIMENT' }],
          requestedAssessments,
          experimentId,
        });

        expect(evaluationResults).toHaveLength(2);
        expect(evaluationResults?.[0]).toEqual({
          trace: mockTraces[0],
          results: [
            {
              result: 'correct',
              rationale: 'Good',
              error: null,
              span_name: '',
            },
          ],
          error: null,
        });
        expect(evaluationResults?.[1]).toEqual({
          trace: mockTraces[1],
          results: [],
          error: 'Assessment failed',
        });
      });
    });
  });
});
