// Stub — only needed so service.ts module loads. Tests only use exported helpers.
export const init = () => {};
export const startSpan = () => ({});
export const flushTraces = async () => {};
export const tracingContext = (_: any, fn: any) => fn();
export const SpanStatusCode = { OK: 'OK', ERROR: 'ERROR', UNSET: 'UNSET' };
