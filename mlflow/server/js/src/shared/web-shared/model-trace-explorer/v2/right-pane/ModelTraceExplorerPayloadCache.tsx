import { createContext, useContext, useMemo, useRef, useState } from 'react';

type PayloadCacheKind = 'fields' | 'section-json' | 'input-chat-json' | 'output-chat-json';
type PayloadCache = <T>(kind: PayloadCacheKind, payload: unknown, compute: () => T) => T;
const PayloadCacheContext = createContext<PayloadCache | undefined>(undefined);

const MAX_ENTRIES_PER_KIND = 8;

const createPayloadCache = (): PayloadCache => {
  const kinds: Record<PayloadCacheKind, { payload: unknown; result: unknown }[]> = {
    fields: [],
    'section-json': [],
    'input-chat-json': [],
    'output-chat-json': [],
  };
  return <T,>(kind: PayloadCacheKind, payload: unknown, compute: () => T): T => {
    const entries = kinds[kind];
    const index = entries.findIndex((entry) => entry.payload === payload);
    if (index >= 0) {
      const [entry] = entries.splice(index, 1);
      entries.unshift(entry);
      return entry.result as T;
    }
    const result = compute();
    entries.unshift({ payload, result });
    // Section JSON cannot evict comparison field pairs from their separate budget.
    if (entries.length > MAX_ENTRIES_PER_KIND) entries.pop();
    return result;
  };
};

export const ModelTraceExplorerPayloadCacheProvider = ({ children }: { children: React.ReactNode }) => {
  const [cache] = useState(createPayloadCache);
  return <PayloadCacheContext.Provider value={cache}>{children}</PayloadCacheContext.Provider>;
};

// Payloads are immutable trace values; each kind computes a result using only that payload.
export const useCachedPayloadResult = <T,>(kind: PayloadCacheKind, payload: unknown, compute: () => T): T => {
  const cache = useContext(PayloadCacheContext);
  const computeRef = useRef(compute);
  computeRef.current = compute;
  return useMemo(
    () => (cache && payload !== undefined ? cache(kind, payload, computeRef.current) : computeRef.current()),
    [cache, kind, payload],
  );
};
