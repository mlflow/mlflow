import {
  countTraceSpanAttributeChars,
  getTraceResponseSpanContainers,
  parseTraceResponseBody,
  type TraceResponseRequest,
  type TraceResponseMessage,
} from './traceResponseParsing';
import { compactTraceSpanPayloads } from './compactJsonPayload';

const CHUNK_ATTRIBUTE_CHARS = 2 * 1024 * 1024;
const CHUNK_SPANS = 256;

const scope = globalThis as unknown as {
  onmessage: (event: MessageEvent<TraceResponseRequest>) => void;
  postMessage: (message: TraceResponseMessage) => void;
};

scope.onmessage = ({ data: { id, body } }) => {
  try {
    const value = parseTraceResponseBody(body);
    getTraceResponseSpanContainers(value).forEach((container, index) => {
      const { spans } = container;
      let start = 0;
      let chars = 0;
      for (let end = 0; end < spans.length; end++) {
        spans[end] = compactTraceSpanPayloads(spans[end]);
        const spanChars = countTraceSpanAttributeChars(spans[end]);
        if (end > start && (chars + spanChars > CHUNK_ATTRIBUTE_CHARS || end - start >= CHUNK_SPANS)) {
          scope.postMessage({ type: 'span-chunk', id, index, spans: spans.slice(start, end) });
          start = end;
          chars = 0;
        }
        chars += spanChars;
      }
      if (start > 0) {
        scope.postMessage({ type: 'span-chunk', id, index, spans: spans.slice(start) });
        container.spans = [];
      }
      // shortcut: a single oversized span is indivisible; stream its payload if that remains a bottleneck.
    });
    scope.postMessage({ type: 'parsed', id, value });
  } catch (error) {
    scope.postMessage({
      type: 'failed',
      id,
      name: error instanceof Error ? error.name : 'Error',
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
scope.postMessage({ type: 'ready' });
