import { describe, expect, test } from '@jest/globals';

import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import { QueryClient, QueryObserver } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

import { pruneTraceCache } from './useGetTrace';

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
  test('keeps only the five most recently loaded inactive traces', () => {
    const queryClient = new QueryClient();
    for (let index = 0; index < 6; index++) {
      setTrace(queryClient, `trace-${index}`, createTrace(['small']), index + 1);
    }

    pruneTraceCache(queryClient);

    expect(queryClient.getQueryData(['getTrace', 'trace-0', undefined])).toBeUndefined();
    for (let index = 1; index < 6; index++) {
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
});
