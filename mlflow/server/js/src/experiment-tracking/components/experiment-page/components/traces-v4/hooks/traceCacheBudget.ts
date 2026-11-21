import type { QueryClient } from '../../../../../../common/utils/reactQueryHooks';
import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import { countTraceSpanAttributeChars } from '@databricks/web-shared/model-trace-explorer/traceResponseParsing';

const MAX_TRACES = 5;
const MAX_ATTRIBUTE_CHARS = 64 * 1024 * 1024;
export const TRACE_CACHE_TIME_MS = 5 * 60 * 1000;
const installed = new WeakSet<QueryClient>();

// The policy lives with each query cache so it also sees the last drawer observer leave.
export const installTraceCacheBudget = (client: QueryClient): void => {
  if (installed.has(client)) return;
  installed.add(client);
  const cache = client.getQueryCache();
  const sizes = new WeakMap<ModelTrace, number>();
  const isDrawerTrace = (key: readonly unknown[]) => key[0] === 'getTrace' && key.length === 3;
  const prune = () => {
    let totalChars = 0;
    const now = Date.now();
    const entries = cache
      .findAll({ queryKey: ['getTrace'], type: 'inactive' })
      .filter(
        (query) =>
          isDrawerTrace(query.queryKey) &&
          query.getObserversCount() === 0 &&
          query.state.data !== undefined &&
          query.state.fetchStatus === 'idle',
      )
      .sort((a, b) => a.state.dataUpdatedAt - b.state.dataUpdatedAt)
      .filter((query) => {
        const trace = query.state.data as ModelTrace | undefined;
        let chars = trace && sizes.get(trace);
        if (chars === undefined) {
          chars = trace?.data?.spans?.reduce((sum, span) => sum + countTraceSpanAttributeChars(span), 0) ?? 0;
          if (trace) sizes.set(trace, chars);
        }
        if (chars > MAX_ATTRIBUTE_CHARS || now - query.state.dataUpdatedAt >= TRACE_CACHE_TIME_MS) {
          cache.remove(query);
          return false;
        }
        totalChars += chars;
        return true;
      });
    let count = entries.length;
    for (const query of entries) {
      if (count <= MAX_TRACES && totalChars <= MAX_ATTRIBUTE_CHARS) break;
      const trace = query.state.data as ModelTrace | undefined;
      totalChars -= (trace && sizes.get(trace)) ?? 0;
      count--;
      cache.remove(query);
    }
  };
  cache.subscribe((event) => {
    if (
      isDrawerTrace(event.query.queryKey) &&
      (event.type === 'observerRemoved' || (event.type === 'updated' && event.action.type === 'success'))
    ) {
      prune();
    }
  });
  prune();
};
