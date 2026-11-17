import { parseTraceResponseBody, type TraceResponseRequest, type TraceResponseMessage } from './traceResponseParsing';

const scope = globalThis as unknown as {
  onmessage: (event: MessageEvent<TraceResponseRequest>) => void;
  postMessage: (message: TraceResponseMessage) => void;
};

scope.onmessage = ({ data: { id, body } }) => {
  try {
    // shortcut: whole-object delivery still clones; chunk span replies for very large traces.
    scope.postMessage({ type: 'parsed', id, value: parseTraceResponseBody(body) });
  } catch (error) {
    scope.postMessage({ type: 'failed', id, message: error instanceof Error ? error.message : String(error) });
  }
};
scope.postMessage({ type: 'ready' });
