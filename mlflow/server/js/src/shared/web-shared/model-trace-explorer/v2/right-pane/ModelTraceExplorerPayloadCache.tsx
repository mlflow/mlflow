import { createContext, useContext, useMemo, useRef, useState } from 'react';

type PayloadCache = <T>(kind: string, payload: unknown, compute: () => T) => T;
const PayloadCacheContext = createContext<PayloadCache | undefined>(undefined);

const createPayloadCache = (): PayloadCache => {
  const entries: { kind: string; payload: unknown; result: unknown }[] = [];
  return <T,>(kind: string, payload: unknown, compute: () => T): T => {
    const index = entries.findIndex((entry) => entry.kind === kind && entry.payload === payload);
    if (index >= 0) {
      const [entry] = entries.splice(index, 1);
      entries.unshift(entry);
      return entry.result as T;
    }
    const result = compute();
    entries.unshift({ kind, payload, result });
    // Keep only a few selected payloads while switching tabs or inspecting nearby spans.
    if (entries.length > 8) entries.pop();
    return result;
  };
};

export const ModelTraceExplorerPayloadCacheProvider = ({ children }: { children: React.ReactNode }) => {
  const [cache] = useState(createPayloadCache);
  return <PayloadCacheContext.Provider value={cache}>{children}</PayloadCacheContext.Provider>;
};

// Payloads are immutable trace values; each kind computes a result using only that payload.
export const useCachedPayloadResult = <T,>(kind: string, payload: unknown, compute: () => T): T => {
  const cache = useContext(PayloadCacheContext);
  const computeRef = useRef(compute);
  computeRef.current = compute;
  return useMemo(
    () => (cache && payload !== undefined ? cache(kind, payload, computeRef.current) : computeRef.current()),
    [cache, kind, payload],
  );
};
