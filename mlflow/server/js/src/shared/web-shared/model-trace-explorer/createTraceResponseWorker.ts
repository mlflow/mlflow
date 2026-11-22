export const createTraceResponseWorker = (): Worker =>
  new Worker(new URL('./traceResponse.worker.ts', import.meta.url));
