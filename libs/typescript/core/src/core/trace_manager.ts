import { LiveSpan, Span } from './entities/span';
import { TraceInfo } from './entities/trace_info';
import { Trace } from './entities/trace';
import { TraceData } from './entities/trace_data';
import {
  REQUEST_RESPONSE_PREVIEW_MAX_LENGTH,
  SpanAttributeKey,
  TraceMetadataKey,
} from './constants';

/**
 * Internal representation to keep the state of a trace.
 * Uses a Map<string, LiveSpan> instead of TraceData to allow access by span_id.
 */
class _Trace {
  info: TraceInfo;
  spanDict: Map<string, LiveSpan>;

  constructor(info: TraceInfo) {
    this.info = info;
    this.spanDict = new Map<string, LiveSpan>();
  }

  /**
   * Convert the internal trace representation to an MLflow Trace object
   */
  toMlflowTrace(): Trace {
    // Convert LiveSpan, mutable objects, into immutable Span objects before persisting
    const traceData = new TraceData([...this.spanDict.values()] as Span[]);

    const root_span = traceData.spans.find((span) => span.parentId == null);
    if (root_span) {
      // Only set previews if they haven't been explicitly set by updateCurrentTrace
      if (!this.info.requestPreview) {
        this.info.requestPreview = getTruncatedPreview(
          root_span._span.attributes[SpanAttributeKey.INPUTS] as string,
          'user',
        );
      }
      if (!this.info.responsePreview) {
        this.info.responsePreview = getTruncatedPreview(
          root_span._span.attributes[SpanAttributeKey.OUTPUTS] as string,
          'assistant',
        );
      }

      // TODO: Remove this once the new trace table UI is available that is based on MLflow V3 trace.
      // Until then, these two metadata are still used to render the "request" and "response" columns.
      this.info.traceMetadata[TraceMetadataKey.INPUTS] = this.info.requestPreview;
      this.info.traceMetadata[TraceMetadataKey.OUTPUTS] = this.info.responsePreview;
    }

    return new Trace(this.info, traceData);
  }
}

/**
 * Generate a truncated preview string from span inputs/outputs.
 * Extracts message text content from known chat formats (OpenAI messages,
 * choices, Responses API) before truncating, so the preview shows readable
 * text instead of cut-off JSON.
 *
 * Mirrors the Python SDK's `_get_truncated_preview` in
 * `mlflow/tracing/utils/truncation.py`.
 */
function getTruncatedPreview(inputsOrOutputs: string, role: string): string {
  if (!inputsOrOutputs) {
    return '';
  }

  const maxLength = REQUEST_RESPONSE_PREVIEW_MAX_LENGTH;
  let content: string | null = null;

  try {
    const obj = JSON.parse(inputsOrOutputs);
    if (obj && typeof obj === 'object') {
      const messages = tryExtractMessages(obj as Record<string, unknown>);
      if (messages && messages.length > 0) {
        const msg = getLastMessageByRole(messages, role);
        content = getTextContentFromMessage(msg);
      }
    }
  } catch {
    // Not valid JSON — use as-is
  }

  const preview = content || inputsOrOutputs;
  if (preview.length <= maxLength) {
    return preview;
  }
  return preview.slice(0, maxLength - 3) + '...';
}

function tryExtractMessages(obj: Record<string, unknown>): Record<string, unknown>[] | null {
  // OpenAI ChatCompletion request format: { messages: [...] }
  if (Array.isArray(obj.messages)) {
    return obj.messages.filter(isMessage);
  }

  // OpenAI ChatCompletion response format: { choices: [{ message: {...} }] }
  if (
    Array.isArray(obj.choices) &&
    obj.choices.length > 0 &&
    typeof obj.choices[0] === 'object' &&
    obj.choices[0] != null
  ) {
    const msg = (obj.choices[0] as Record<string, unknown>).message;
    if (isMessage(msg)) {
      return [msg];
    }
  }

  // OpenAI Responses API request format: { input: [...] }
  if (Array.isArray(obj.input)) {
    return obj.input.filter(isMessage);
  }

  // OpenAI Responses API response format: { output: [...] }
  if (Array.isArray(obj.output)) {
    return obj.output.filter(isMessage);
  }

  // ResponsesAgent input: { request: { ... } }
  if (obj.request && typeof obj.request === 'object') {
    return tryExtractMessages(obj.request as Record<string, unknown>);
  }

  return null;
}

function isMessage(item: unknown): item is Record<string, unknown> {
  return (
    item != null &&
    typeof item === 'object' &&
    'role' in (item as Record<string, unknown>) &&
    'content' in (item as Record<string, unknown>)
  );
}

function getLastMessageByRole(
  messages: Record<string, unknown>[],
  role: string,
): Record<string, unknown> {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === role) {
      return messages[i];
    }
  }
  return messages[messages.length - 1];
}

function getTextContentFromMessage(message: Record<string, unknown>): string | null {
  const content = message.content;
  if (typeof content === 'string') {
    return content;
  }
  if (Array.isArray(content)) {
    for (const part of content) {
      if (typeof part === 'string') {
        return part;
      }
      if (
        part &&
        typeof part === 'object' &&
        'type' in (part as Record<string, unknown>) &&
        ((part as Record<string, unknown>).type === 'text' ||
          (part as Record<string, unknown>).type === 'output_text')
      ) {
        return (part as Record<string, unknown>).text as string;
      }
    }
  }
  return null;
}

/**
 * Manage spans and traces created by the tracing system in memory.
 * This class is implemented as a singleton.
 */
export class InMemoryTraceManager {
  private static _instance: InMemoryTraceManager | undefined;

  // In-memory cache to store trace_id -> _Trace mapping
  // TODO: Add TTL to the trace buffer similarly to Python SDK
  private _traces: Map<string, _Trace>;
  // Store mapping between OpenTelemetry trace ID and MLflow trace ID
  private _otelIdToMlflowTraceId: Map<string, string>;

  // Store the last active trace ID
  lastActiveTraceId: string | undefined;

  /**
   * Singleton pattern implementation
   */
  static getInstance(): InMemoryTraceManager {
    if (InMemoryTraceManager._instance == null) {
      InMemoryTraceManager._instance = new InMemoryTraceManager();
    }
    return InMemoryTraceManager._instance;
  }

  private constructor() {
    this._traces = new Map<string, _Trace>();
    this._otelIdToMlflowTraceId = new Map<string, string>();
  }

  /**
   * Register a new trace info object to the in-memory trace registry.
   * @param otelTraceId The OpenTelemetry trace ID for the new trace
   * @param traceInfo The trace info object to be stored
   */
  registerTrace(otelTraceId: string, traceInfo: TraceInfo): void {
    this._traces.set(traceInfo.traceId, new _Trace(traceInfo));
    this._otelIdToMlflowTraceId.set(otelTraceId, traceInfo.traceId);
  }

  /**
   * Store the given span in the in-memory trace data.
   * @param span The span to be stored
   */
  registerSpan(span: LiveSpan): void {
    const trace = this._traces.get(span.traceId);
    if (trace) {
      trace.spanDict.set(span.spanId, span);
    } else {
      console.debug(`Tried to register span ${span.spanId} for trace ${span.traceId}
                     but trace not found. Please make sure to register the trace first.`);
    }
  }

  /**
   * Get the trace for the given trace ID.
   * Returns the trace object or null if not found.
   * @param traceId The trace ID to look up
   */
  getTrace(traceId: string): _Trace | null {
    return this._traces.get(traceId) || null;
  }

  /**
   * Get the MLflow trace ID for the given OpenTelemetry trace ID.
   * @param otelTraceId The OpenTelemetry trace ID
   */
  getMlflowTraceIdFromOtelId(otelTraceId: string): string | null {
    return this._otelIdToMlflowTraceId.get(otelTraceId) || null;
  }

  /**
   * Get the span for the given trace ID and span ID.
   * @param traceId The trace ID
   * @param spanId The span ID
   */
  getSpan(traceId?: string | null, spanId?: string | null): LiveSpan | null {
    if (traceId == null || spanId == null) {
      return null;
    }
    return this._traces.get(traceId)?.spanDict.get(spanId) || null;
  }

  /**
   * Pop trace data for the given OpenTelemetry trace ID and return it as
   * a ready-to-publish Trace object.
   * @param otelTraceId The OpenTelemetry trace ID
   */
  popTrace(otelTraceId: string): Trace | null {
    const mlflowTraceId = this._otelIdToMlflowTraceId.get(otelTraceId);
    if (!mlflowTraceId) {
      console.debug(`Tried to pop trace ${otelTraceId} but no trace found.`);
      return null;
    }

    this._otelIdToMlflowTraceId.delete(otelTraceId);
    const trace = this._traces.get(mlflowTraceId);
    if (trace) {
      this._traces.delete(mlflowTraceId);
      return trace.toMlflowTrace();
    }
    console.debug(`Tried to pop trace ${otelTraceId} but trace not found.`);
    return null;
  }

  /**
   * Clear all the aggregated trace data. This should only be used for testing.
   */
  static reset(): void {
    if (InMemoryTraceManager._instance) {
      InMemoryTraceManager._instance._traces.clear();
      InMemoryTraceManager._instance._otelIdToMlflowTraceId.clear();
      InMemoryTraceManager._instance = undefined;
    }
  }
}
