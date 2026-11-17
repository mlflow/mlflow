export const createTraceResponseWorker = (): Worker | undefined =>
  typeof Worker === 'undefined' ? undefined : new Worker(new URL('./traceResponse.worker.ts', import.meta.url));
