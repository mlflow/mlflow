import { describe, expect, test } from '@jest/globals';

import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import { QueryClient, QueryObserver } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

import { pruneTraceCache, subscribeToTraceCachePruning } from './useGetTrace';

const createTrace = (payloads: string[]): ModelTrace =>
  ({
    data: {
      spans: payloads.map((payload, index) => ({
        trace_id: 'trace',
        span_id: String(index),
        trace_state: '',
        parent_span_id: null,
        name: 'span',
        start_time_unix_nano: '0',
        end_time_unix_nano: '1',
        status: { code: 'STATUS_CODE_OK' },
        attributes: { payload },
      })),
    },
    info: {},
  }) as ModelTrace;

const setTrace = (queryClient: QueryClient, id: string, trace: ModelTrace, updatedAt: number) =>
  queryClient.setQueryData(['getTrace', id, undefined], trace, { updatedAt });

describe('pruneTraceCache', () => {
  test('keeps every inactive trace while their payloads fit the cache budget', () => {
    const queryClient = new QueryClient();
    for (let index = 0; index < 6; index++) {
      setTrace(queryClient, `trace-${index}`, createTrace(['small']), index + 1);
    }

    pruneTraceCache(queryClient);

    for (let index = 0; index < 6; index++) {
      expect(queryClient.getQueryData(['getTrace', `trace-${index}`, undefined])).toBeDefined();
    }
  });

  test('evicts oldest traces until their string payloads fit the cache budget', () => {
    const queryClient = new QueryClient();
    const oneMiB = 'x'.repeat(1024 * 1024);
    setTrace(queryClient, 'old', createTrace(Array(33).fill(oneMiB)), 1);
    setTrace(queryClient, 'new', createTrace(Array(33).fill(oneMiB)), 2);

    pruneTraceCache(queryClient);

    expect(queryClient.getQueryData(['getTrace', 'old', undefined])).toBeUndefined();
    expect(queryClient.getQueryData(['getTrace', 'new', undefined])).toBeDefined();
  });

  test('does not retain a single trace larger than the cache budget', () => {
    const queryClient = new QueryClient();
    const oneMiB = 'x'.repeat(1024 * 1024);
    setTrace(queryClient, 'oversized', createTrace(Array(65).fill(oneMiB)), 1);

    pruneTraceCache(queryClient);

    expect(queryClient.getQueryData(['getTrace', 'oversized', undefined])).toBeUndefined();
  });

  test('does not evict an observed trace', () => {
    const queryClient = new QueryClient();
    const queryKey = ['getTrace', 'observed', undefined];
    const oneMiB = 'x'.repeat(1024 * 1024);
    setTrace(queryClient, 'observed', createTrace(Array(65).fill(oneMiB)), 1);
    setTrace(queryClient, 'inactive', createTrace(['small']), 2);
    const observer = new QueryObserver(queryClient, { queryKey, enabled: false });
    const unsubscribe = observer.subscribe(() => {});

    pruneTraceCache(queryClient);

    expect(queryClient.getQueryData(queryKey)).toBeDefined();
    expect(queryClient.getQueryData(['getTrace', 'inactive', undefined])).toBeUndefined();
    unsubscribe();
  });

  test('does not prune shared two-part getTrace queries', () => {
    const queryClient = new QueryClient();
    const sharedQueryKey = ['getTrace', 'shared'];
    queryClient.setQueryData(sharedQueryKey, createTrace(Array(65).fill('x'.repeat(1024 * 1024))));

    pruneTraceCache(queryClient);

    expect(queryClient.getQueryData(sharedQueryKey)).toBeDefined();
  });

  test('keeps the newly observed trace when an observer moves between cached keys', () => {
    const queryClient = new QueryClient();
    const payload = 'x'.repeat(11 * 1024 * 1024);
    for (let index = 0; index < 7; index++) {
      setTrace(queryClient, `trace-${index}`, createTrace([payload]), index + 1);
    }
    const unsubscribeCache = subscribeToTraceCachePruning(queryClient);
    const observer = new QueryObserver(queryClient, {
      queryKey: ['getTrace', 'trace-6', undefined],
      enabled: false,
    });

    const unsubscribeObserver = observer.subscribe(() => {});
    observer.setOptions({ queryKey: ['getTrace', 'trace-0', undefined], enabled: false });
    pruneTraceCache(queryClient);

    expect(queryClient.getQueryData(['getTrace', 'trace-0', undefined])).toBeDefined();
    expect(queryClient.getQueryData(['getTrace', 'trace-1', undefined])).toBeUndefined();
    unsubscribeObserver();
    unsubscribeCache();
  });

  test('prunes a trace that finishes loading after its observer is removed', async () => {
    const queryClient = new QueryClient();
    const queryKey = ['getTrace', 'late', undefined];
    const oversizedTrace = createTrace(Array(65).fill('x'.repeat(1024 * 1024)));
    let resolveTrace!: (trace: ModelTrace) => void;
    const tracePromise = new Promise<ModelTrace>((resolve) => {
      resolveTrace = resolve;
    });
    const unsubscribeCache = subscribeToTraceCachePruning(queryClient);
    const observer = new QueryObserver(queryClient, {
      queryKey,
      queryFn: () => tracePromise,
      enabled: false,
    });
    const unsubscribeObserver = observer.subscribe(() => {});

    const fetch = observer.refetch();
    unsubscribeObserver();
    resolveTrace(oversizedTrace);
    await fetch;

    expect(queryClient.getQueryData(queryKey)).toBeUndefined();
    unsubscribeCache();
  });

  test('prunes the final closed trace when the subscription is removed', () => {
    const queryClient = new QueryClient();
    const queryKey = ['getTrace', 'last', undefined];
    setTrace(queryClient, 'last', createTrace(Array(65).fill('x'.repeat(1024 * 1024))), 1);
    const unsubscribeCache = subscribeToTraceCachePruning(queryClient);
    const observer = new QueryObserver(queryClient, { queryKey, enabled: false });
    const unsubscribeObserver = observer.subscribe(() => {});

    unsubscribeObserver();
    expect(queryClient.getQueryData(queryKey)).toBeDefined();
    unsubscribeCache();

    expect(queryClient.getQueryData(queryKey)).toBeUndefined();
  });
});
