import { afterEach, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { QueryClient, QueryObserver } from '../../../../../../common/utils/reactQueryHooks';
import { MOCK_V3_TRACE } from '@databricks/web-shared/model-trace-explorer/ModelTraceExplorer.test-utils';
import { installTraceCacheBudget, TRACE_CACHE_TIME_MS } from './traceCacheBudget';

const key = (id: string) => ['getTrace', id, undefined];
const traceWithAttributes = (attributes: any) => ({
  ...MOCK_V3_TRACE,
  data: { spans: [{ ...MOCK_V3_TRACE.data.spans[0], attributes }] },
});

describe('drawer cache budgets', () => {
  let client: QueryClient;
  beforeEach(() => {
    client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    installTraceCacheBudget(client);
  });
  afterEach(() => {
    client.clear();
    jest.restoreAllMocks();
  });

  it('evicts the oldest trace when aggregate map and OTLP attributes exceed the character budget', () => {
    const payload = 'a'.repeat(32 * 1024 * 1024 + 1);
    client.setQueryData(key('first'), traceWithAttributes({ payload }));
    const second = traceWithAttributes([{ key: 'payload', value: { string_value: payload } }]);
    client.setQueryData(key('second'), second);
    expect(client.getQueryData(key('first'))).toBeUndefined();
    expect(client.getQueryData(key('second'))).toBe(second);
  });

  it.each(['native-map', 'native-OTLP'])('accounts for nested %s strings in the retained budget', (shape) => {
    const content = 'a'.repeat(64 * 1024 * 1024 + 1);
    const attributes =
      shape === 'native-map'
        ? { payload: { records: [{ content }] } }
        : [
            {
              key: 'mlflow.spanOutputs',
              value: {
                kvlist_value: {
                  values: [
                    {
                      key: 'records',
                      value: {
                        array_value: {
                          values: [
                            { kvlist_value: { values: [{ key: 'content', value: { string_value: content } }] } },
                          ],
                        },
                      },
                    },
                  ],
                },
              },
            },
          ];
    client.setQueryData(key(shape), traceWithAttributes(attributes));
    expect(client.getQueryData(key(shape))).toBeUndefined();
  });

  it('expires a completed trace at the age limit and leaves other trace query families alone', () => {
    const old = Date.now() - TRACE_CACHE_TIME_MS;
    client.setQueryData(['getTrace', 'shared-table'], MOCK_V3_TRACE, { updatedAt: old });
    client.setQueryData(['GET_TRACE_BY_ID', 'shared-explorer'], MOCK_V3_TRACE, { updatedAt: old });
    client.setQueryData(key('expired'), MOCK_V3_TRACE, { updatedAt: old });
    expect(client.getQueryData(key('expired'))).toBeUndefined();
    expect(client.getQueryData(['getTrace', 'shared-table'])).toBe(MOCK_V3_TRACE);
    expect(client.getQueryData(['GET_TRACE_BY_ID', 'shared-explorer'])).toBe(MOCK_V3_TRACE);
  });

  it('keeps an actively observed oversized trace until its last observer leaves', () => {
    const trace = traceWithAttributes({ payload: 'a'.repeat(64 * 1024 * 1024 + 1) });
    const observer = new QueryObserver(client, { queryKey: key('active'), enabled: false });
    const stop = observer.subscribe(() => {});
    try {
      client.setQueryData(key('active'), trace);
      expect(client.getQueryData(key('active'))).toBe(trace);
      const otherObserver = new QueryObserver(client, { queryKey: key('active'), enabled: false });
      const stopOther = otherObserver.subscribe(() => {});
      stop();
      expect(client.getQueryData(key('active'))).toBe(trace);
      stopOther();
      expect(client.getQueryData(key('active'))).toBeUndefined();
    } finally {
      stop();
    }
  });

  it('keeps an inactive refetch until it settles, without losing its caller result', async () => {
    client.setQueryData(key('refresh'), MOCK_V3_TRACE);
    const oversized = traceWithAttributes({ payload: 'a'.repeat(64 * 1024 * 1024 + 1) });
    let finish!: (value: typeof oversized) => void;
    const result = client.fetchQuery({
      queryKey: key('refresh'),
      queryFn: () =>
        new Promise<typeof oversized>((resolve) => {
          finish = resolve;
        }),
    });
    for (let i = 0; i < 8; i++) client.setQueryData(key(`other-${i}`), MOCK_V3_TRACE);
    expect(client.getQueryData(key('refresh'))).toBe(MOCK_V3_TRACE);
    finish(oversized);
    await expect(result).resolves.toEqual(oversized);
    expect(client.getQueryData(key('refresh'))).toBeUndefined();
  });

  it('does no cache scan for unrelated query events and installs once per client', () => {
    const scan = jest.spyOn(client.getQueryCache(), 'findAll');
    installTraceCacheBudget(client);
    client.setQueryData(['trace-info', 'unrelated'], { status: 'OK' });
    expect(scan).not.toHaveBeenCalled();
    client.setQueryData(key('owned'), MOCK_V3_TRACE);
    expect(scan).toHaveBeenCalledTimes(1);
  });

  it('releases an oversized retained body after its active refetch fails', async () => {
    jest.spyOn(console, 'error').mockImplementation(() => {});
    const trace = traceWithAttributes({ payload: 'a'.repeat(64 * 1024 * 1024 + 1) });
    const observer = new QueryObserver(client, { queryKey: key('failed-refresh'), enabled: false });
    const stop = observer.subscribe(() => {});
    try {
      client.setQueryData(key('failed-refresh'), trace);
      await expect(
        client.fetchQuery({
          queryKey: key('failed-refresh'),
          queryFn: async () => {
            throw new Error('offline');
          },
        }),
      ).rejects.toThrow('offline');
      expect(client.getQueryData(key('failed-refresh'))).toBe(trace);
      stop();
      expect(client.getQueryData(key('failed-refresh'))).toBeUndefined();
    } finally {
      stop();
    }
  });

  it('counts immutable span attributes once across later trace arrivals', () => {
    let reads = 0;
    const span = {
      ...MOCK_V3_TRACE.data.spans[0],
      get attributes() {
        reads++;
        return { payload: 'owned' };
      },
    };
    client.setQueryData(key('first'), { ...MOCK_V3_TRACE, data: { spans: [span] } });
    const firstReads = reads;
    expect(firstReads).toBeGreaterThan(0);
    client.setQueryData(key('second'), MOCK_V3_TRACE);
    client.setQueryData(key('third'), MOCK_V3_TRACE);
    expect(reads).toBe(firstReads);
  });
});
