import { parseTraceResponseBody, type TraceResponseMessage } from './traceResponseParsing';

const WORKER_MIN_BYTES = 64 * 1024;
const WORKER_TIMEOUT_MS = 10_000;
const abortError = () => new DOMException('Aborted', 'AbortError');

// The transferred body is lost on worker failure; GET callers can fetch it again.
export class TraceResponseWorkerLostError extends Error {}

export const createTraceResponseParser = () => {
  let worker: Worker | undefined;
  let ready = false;
  let disposed = false;
  let stopped = false;
  let nextId = 0;
  let startTimer: ReturnType<typeof setTimeout> | undefined;
  const pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();

  const stop = (error: Error) => {
    // shortcut: disable a failed worker for this page; add bounded restart if transient failures warrant it.
    stopped = true;
    ready = false;
    clearTimeout(startTimer);
    worker?.terminate();
    worker = undefined;
    for (const request of pending.values()) request.reject(error);
    pending.clear();
  };

  if (typeof Worker !== 'undefined') {
    startTimer = setTimeout(() => stop(new TraceResponseWorkerLostError()), WORKER_TIMEOUT_MS);
    void import('./createTraceResponseWorker')
      .then(({ createTraceResponseWorker }) => {
        if (stopped) return;
        worker = createTraceResponseWorker();
        worker.onerror = () => stop(new TraceResponseWorkerLostError());
        worker.onmessageerror = () => stop(new TraceResponseWorkerLostError());
        worker.onmessage = ({ data }: MessageEvent<TraceResponseMessage>) => {
          if (!worker || disposed) return;
          if (data.type === 'ready') {
            ready = true;
            clearTimeout(startTimer);
          } else {
            const request = pending.get(data.id);
            pending.delete(data.id);
            if (data.type === 'parsed') request?.resolve(data.value);
            else {
              const ErrorClass =
                data.name === 'SyntaxError'
                  ? SyntaxError
                  : data.name === 'RangeError'
                    ? RangeError
                    : data.name === 'TypeError'
                      ? TypeError
                      : Error;
              const error = new ErrorClass(data.message);
              error.name = data.name;
              request?.reject(error);
            }
          }
        };
      })
      .catch((cause) => stop(new TraceResponseWorkerLostError('Worker startup failed', { cause })));
  }

  return {
    async parse(response: Response, signal?: AbortSignal): Promise<unknown> {
      if (disposed || signal?.aborted) throw abortError();
      // Loading, unsupported or failed workers never delay a response.
      if (!ready) {
        const value = await response.json();
        if (disposed || signal?.aborted) throw abortError();
        return value;
      }
      const body = await response.arrayBuffer();
      if (disposed || signal?.aborted) throw abortError();
      if (!ready || body.byteLength < WORKER_MIN_BYTES) return parseTraceResponseBody(body);
      return new Promise((resolve, reject) => {
        const id = nextId++;
        const onAbort = () => {
          pending.delete(id);
          finish(() => reject(abortError()));
        };
        const timer = setTimeout(() => {
          pending.delete(id);
          // Keep existing requests independent, but do not reuse a worker that timed out.
          ready = false;
          stopped = true;
          finish(() => reject(new TraceResponseWorkerLostError()));
        }, WORKER_TIMEOUT_MS);
        const finish = (settle: () => void) => {
          clearTimeout(timer);
          signal?.removeEventListener('abort', onAbort);
          if (stopped && !pending.size) {
            worker?.terminate();
            worker = undefined;
          }
          settle();
        };
        signal?.addEventListener('abort', onAbort, { once: true });
        pending.set(id, {
          resolve: (value) => finish(() => resolve(value)),
          reject: (error) => finish(() => reject(error)),
        });
        try {
          worker?.postMessage({ id, body }, [body]);
        } catch (cause) {
          stop(new TraceResponseWorkerLostError('Worker transfer failed', { cause }));
        }
      });
    },
    dispose() {
      disposed = true;
      stop(abortError());
    },
  };
};
