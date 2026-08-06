import { describe, beforeEach, it, expect } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { rest } from 'msw';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { setupServer } from '../../../common/utils/setup-msw';
import { useGetSessionsForEvaluation } from './useGetSessionsForEvaluation';
import type { EvaluateTracesParams } from './types';

type CapturedSearchBody = {
  filter?: string;
  max_results?: number;
  order_by?: string[];
};

const LOCATIONS = [{ type: 'MLFLOW_EXPERIMENT', mlflow_experiment: { experiment_id: 'exp-123' } }];

const createTraceInfo = (traceId: string, sessionId: string, requestTime: number): ModelTraceInfoV3 =>
  ({
    trace_id: traceId,
    trace_location: {
      type: 'MLFLOW_EXPERIMENT',
      mlflow_experiment: { experiment_id: 'exp-123' },
    },
    request_time: requestTime,
    state: 'OK',
    trace_metadata: { 'mlflow.trace.session': sessionId },
  }) as unknown as ModelTraceInfoV3;

const buildParams = (overrides: Partial<EvaluateTracesParams>): EvaluateTracesParams =>
  ({
    locations: LOCATIONS,
    experimentId: 'exp-123',
    judgeInstructions: 'judge',
    ...overrides,
  }) as EvaluateTracesParams;

describe('useGetSessionsForEvaluation', () => {
  const server = setupServer();
  let queryClient: QueryClient;
  let wrapper: React.ComponentType<{ children: React.ReactNode }>;
  let capturedBodies: CapturedSearchBody[];

  const useSearchHandler = (resolver: (body: CapturedSearchBody) => ModelTraceInfoV3[]) =>
    server.use(
      rest.post('ajax-api/3.0/mlflow/traces/search', async (req, res, ctx) => {
        const body = (await req.json()) as CapturedSearchBody;
        capturedBodies.push(body);
        return res(ctx.json({ traces: resolver(body) }));
      }),
    );

  beforeEach(() => {
    capturedBodies = [];
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  });

  it('resolves a selected session via targeted filter even when it is outside the latest-500 window', async () => {
    // The unfiltered "latest traces" search only contains recent sessions;
    // the selected session is older and would be silently dropped without the fix.
    useSearchHandler((body) => {
      if (body.filter === "metadata.`mlflow.trace.session` = 'old-session'") {
        // Return out of order to also verify traces are sorted by request_time.
        return [createTraceInfo('old-2', 'old-session', 2000), createTraceInfo('old-1', 'old-session', 1000)];
      }
      return Array.from({ length: 600 }, (_, index) =>
        createTraceInfo(`recent-${index}`, 'recent-session', 10_000 + index),
      );
    });

    const { result } = renderHook(() => useGetSessionsForEvaluation(), { wrapper });
    const sessions = await result.current(buildParams({ itemIds: ['old-session'] }));

    expect(sessions).toHaveLength(1);
    expect(sessions[0].sessionId).toBe('old-session');
    expect(sessions[0].traceInfos.map((trace) => trace.trace_id)).toEqual(['old-1', 'old-2']);
    // Every issued request must carry the session filter — no unfiltered window scan.
    expect(capturedBodies).not.toHaveLength(0);
    expect(capturedBodies.every((body) => body.filter === "metadata.`mlflow.trace.session` = 'old-session'")).toBe(
      true,
    );
  });

  it('resolves each selected session with its own cached filtered query', async () => {
    useSearchHandler((body) => {
      if (body.filter === "metadata.`mlflow.trace.session` = 'session-1'") {
        return [createTraceInfo('trace-1', 'session-1', 1000)];
      }
      if (body.filter === "metadata.`mlflow.trace.session` = 'session-2'") {
        return [createTraceInfo('trace-2', 'session-2', 2000)];
      }
      return [];
    });

    const { result } = renderHook(() => useGetSessionsForEvaluation(), { wrapper });
    const sessions = await result.current(buildParams({ itemIds: ['session-1', 'session-2'] }));

    expect(sessions.map((session) => session.sessionId).sort()).toEqual(['session-1', 'session-2']);
    const issuedFilters = capturedBodies.map((body) => body.filter).sort();
    expect(issuedFilters).toEqual([
      "metadata.`mlflow.trace.session` = 'session-1'",
      "metadata.`mlflow.trace.session` = 'session-2'",
    ]);
  });

  it('escapes single quotes and backslashes in session IDs when building the filter', async () => {
    useSearchHandler(() => []);

    const { result } = renderHook(() => useGetSessionsForEvaluation(), { wrapper });
    await result.current(buildParams({ itemIds: ["sess'ion\\x"] }));

    expect(capturedBodies).toHaveLength(1);
    expect(capturedBodies[0].filter).toBe("metadata.`mlflow.trace.session` = 'sess\\'ion\\\\x'");
  });

  it('keeps the unselected path on the latest-500 window scan without a filter', async () => {
    useSearchHandler(() => [
      createTraceInfo('a-1', 'session-a', 1000),
      createTraceInfo('a-2', 'session-a', 2000),
      createTraceInfo('b-1', 'session-b', 3000),
    ]);

    const { result } = renderHook(() => useGetSessionsForEvaluation(), { wrapper });
    const sessions = await result.current(buildParams({ itemCount: 1 }));

    expect(capturedBodies).toHaveLength(1);
    expect(capturedBodies[0].filter).toBeUndefined();
    expect(capturedBodies[0].max_results).toBe(500);
    expect(capturedBodies[0].order_by).toEqual(['timestamp DESC']);
    // Sliced to the requested number of latest sessions.
    expect(sessions).toHaveLength(1);
    expect(sessions[0].sessionId).toBe('session-a');
    expect(sessions[0].traceInfos).toHaveLength(2);
  });
});
