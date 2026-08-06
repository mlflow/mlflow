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

  it('resolves filter-unsafe session IDs via the window scan instead of a broken filter', async () => {
    // A session ID containing a quote or backslash cannot round-trip through a search
    // filter (the parser strips quotes without decoding escapes), so it must not be put
    // in a filter — it is resolved through the window scan and matched by exact ID.
    const unsafe = "sess'ion\\x";
    useSearchHandler((body) => {
      if (body.filter) {
        return [];
      }
      return [createTraceInfo('unsafe-1', unsafe, 1000)];
    });

    const { result } = renderHook(() => useGetSessionsForEvaluation(), { wrapper });
    const sessions = await result.current(buildParams({ itemIds: [unsafe] }));

    expect(sessions).toHaveLength(1);
    expect(sessions[0].sessionId).toBe(unsafe);
    // Only the unfiltered window scan was issued — no filter for the unsafe ID.
    expect(capturedBodies).toHaveLength(1);
    expect(capturedBodies[0].filter).toBeUndefined();
  });

  it('mixes targeted queries for safe IDs with a window scan when any selected ID is filter-unsafe', async () => {
    const unsafe = "o'brien";
    useSearchHandler((body) => {
      if (body.filter === "metadata.`mlflow.trace.session` = 'safe-session'") {
        return [createTraceInfo('safe-1', 'safe-session', 1000)];
      }
      if (body.filter) {
        return [];
      }
      return [createTraceInfo('unsafe-1', unsafe, 2000)];
    });

    const { result } = renderHook(() => useGetSessionsForEvaluation(), { wrapper });
    const sessions = await result.current(buildParams({ itemIds: ['safe-session', unsafe] }));

    expect(sessions.map((session) => session.sessionId).sort()).toEqual(['safe-session', unsafe].sort());
    expect(capturedBodies.map((body) => body.filter)).toEqual(
      expect.arrayContaining([undefined, "metadata.`mlflow.trace.session` = 'safe-session'"]),
    );
    expect(capturedBodies).toHaveLength(2);
  });

  it('bounds concurrent per-session queries in waves', async () => {
    const sessionIds = Array.from({ length: 25 }, (_, index) => `session-${index}`);

    // Track in-flight requests to assert the concurrency bound.
    let inFlight = 0;
    let maxInFlight = 0;
    server.use(
      rest.post('ajax-api/3.0/mlflow/traces/search', async (req, res, ctx) => {
        inFlight += 1;
        maxInFlight = Math.max(maxInFlight, inFlight);
        const body = (await req.json()) as CapturedSearchBody;
        await new Promise((resolve) => setTimeout(resolve, 20));
        const match = body.filter?.match(/^metadata\.`mlflow\.trace\.session` = 'session-(\d+)'$/);
        const traces = match ? [createTraceInfo(`trace-${match[1]}`, `session-${match[1]}`, Number(match[1]))] : [];
        inFlight -= 1;
        return res(ctx.json({ traces }));
      }),
    );

    const { result } = renderHook(() => useGetSessionsForEvaluation(), { wrapper });
    const sessions = await result.current(buildParams({ itemIds: sessionIds }));

    expect(sessions).toHaveLength(25);
    expect(maxInFlight).toBeLessThanOrEqual(20);
    // Queries within a wave still run concurrently, they are not serialized one by one.
    expect(maxInFlight).toBeGreaterThan(1);
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
