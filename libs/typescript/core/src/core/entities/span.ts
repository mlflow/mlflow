import {
  HrTime,
  INVALID_SPANID,
  INVALID_TRACEID,
  SpanStatusCode as OTelSpanStatusCode
} from '@opentelemetry/api';
import type { Span as OTelSpan } from '@opentelemetry/sdk-trace-base';
import { SpanAttributeKey, SpanType, NO_OP_SPAN_TRACE_ID } from '../constants';
import { SpanEvent } from './span_event';
import { SpanStatus, SpanStatusCode } from './span_status';
import {
  convertHrTimeToNanoSeconds,
  convertNanoSecondsToHrTime,
  encodeSpanIdToBase64,
  encodeTraceIdToBase64,
  decodeIdFromBase64
} from '../utils';
import { safeJsonStringify } from '../utils/json';
/**
 * MLflow Span interface
 */

export interface ISpan {
  /**
   * The OpenTelemetry span wrapped by MLflow Span
   */
  readonly _span: OTelSpan;

  /**
   * The trace ID
   */
  readonly traceId: string;

  /**
   * The attributes of the span
   */
  readonly attributes: Record<string, any>;

  get spanId(): string;
  get name(): string;
  get spanType(): SpanType;
  get startTime(): HrTime;
  get endTime(): HrTime | null;
  get parentId(): string | null;
  get status(): SpanStatus;
  get inputs(): any;
  get outputs(): any;

  /**
   * Get an attribute from the span
   * @param key Attribute key
   * @returns Attribute value
   */
  getAttribute(key: string): any;

  /**
   * Get events from the span
   */
  get events(): SpanEvent[];

  /**
   * Convert this span to JSON format
   * @returns JSON object representation of the span
   */
  toJson(): SerializedSpan;
}

/**
 * MLflow Span class that wraps the OpenTelemetry Span.
 */
export class Span implements ISpan {
  readonly _span: OTelSpan;
  readonly _attributesRegistry: SpanAttributesRegistry;

  /**
   * Create a new MLflowSpan
   * @param span OpenTelemetry span
   */
  constructor(span: OTelSpan, is_mutable: boolean = false) {
    this._span = span;

    if (is_mutable) {
      this._attributesRegistry = new SpanAttributesRegistry(span);
    } else {
      this._attributesRegistry = new CachedSpanAttributesRegistry(span);
    }
  }

  get traceId(): string {
    return this.getAttribute(SpanAttributeKey.TRACE_ID) as string;
  }

  get spanId(): string {
    return this._span.spanContext().spanId;
  }

  get spanType(): SpanType {
    return this.getAttribute(SpanAttributeKey.SPAN_TYPE) as SpanType;
  }

  /**
   * Get the parent span ID
   */
  get parentId(): string | null {
    return this._span.parentSpanContext?.spanId ?? null;
  }

  get name(): string {
    return this._span.name;
  }

  get startTime(): HrTime {
    return this._span.startTime;
  }

  get endTime(): HrTime | null {
    return this._span.endTime;
  }

  get status(): SpanStatus {
    return SpanStatus.fromOtelStatus(this._span.status);
  }

  get inputs(): any {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-return
    return this.getAttribute(SpanAttributeKey.INPUTS);
  }

  get outputs(): any {
    // eslint-disable-next-line @typescript-eslint/no-unsafe-return
    return this.getAttribute(SpanAttributeKey.OUTPUTS);
  }

  get attributes(): Record<string, any> {
    return this._attributesRegistry.getAll();
  }

  getAttribute(key: string): any {
    return this._attributesRegistry.get(key);
  }

  get events(): SpanEvent[] {
    return this._span.events.map((event) => {
      const [seconds, nanoseconds] = event.time;
      return new SpanEvent({
        name: event.name,
        attributes: event.attributes as Record<string, any>,
        timestamp: BigInt(seconds) * 1_000_000_000n + BigInt(nanoseconds)
      });
    });
  }

  /**
   * Convert this span to JSON format (OpenTelemetry format)
   * @returns JSON object representation of the span
   */
  toJson(): SerializedSpan {
    return {
      trace_id: encodeTraceIdToBase64(this.traceId),
      span_id: encodeSpanIdToBase64(this.spanId),
      // Use empty string for parent_span_id if it is not set, to be consistent with Python implementation.
      parent_span_id: this.parentId ? encodeSpanIdToBase64(this.parentId) : '',
      name: this.name,
      start_time_unix_nano: convertHrTimeToNanoSeconds(this.startTime),
      end_time_unix_nano: this.endTime ? convertHrTimeToNanoSeconds(this.endTime) : null,
      status: {
        code: this.status?.statusCode || SpanStatusCode.UNSET,
        message: this.status?.description
      },
      attributes: this.attributes || {},
      events: this.events.map((event) => ({
        name: event.name,
        time_unix_nano: event.timestamp,
        attributes: event.attributes || {}
      }))
    };
  }

  /**
   * Create a Span from JSON data (following Python implementation)
   * Converts the JSON data back into OpenTelemetry-compatible span
   */
  static fromJson(json: SerializedSpan): ISpan {
    // Convert the JSON data back to an OpenTelemetry-like span structure
    // This is simplified compared to Python but follows the same pattern

    const otelSpanData = {
      name: json.name,
      startTime: convertNanoSecondsToHrTime(json.start_time_unix_nano),
      endTime: json.end_time_unix_nano ? convertNanoSecondsToHrTime(json.end_time_unix_nano) : null,
      status: {
        code: convertStatusCodeToOTel(json.status.code),
        message: json.status.message
      },
      // For fromJson, attributes are already in their final form (not JSON serialized)
      // so we store them directly
      attributes: json.attributes || {},
      events: (json.events || []).map((event) => ({
        name: event.name,
        time: convertNanoSecondsToHrTime(event.time_unix_nano),
        attributes: event.attributes || {}
      })),
      ended: true,
      // Add spanContext() method that returns proper SpanContext
      spanContext: () => ({
        traceId: decodeIdFromBase64(json.trace_id),
        spanId: decodeIdFromBase64(json.span_id),
        traceFlags: 1, // Sampled
        isRemote: false
      }),
      // Add parentSpanContext for parent span ID
      parentSpanContext: json.parent_span_id
        ? {
            traceId: decodeIdFromBase64(json.trace_id),
            spanId: decodeIdFromBase64(json.parent_span_id),
            traceFlags: 1,
            isRemote: false
          }
        : undefined
    };

    // Create a span that behaves like our Span class but from downloaded data
    return new Span(otelSpanData as OTelSpan, false); // false = immutable
  }
}

/**
 * Convert MLflow status codes to OpenTelemetry status codes
 * @param statusCode Status code from MLflow JSON format
 * @returns OpenTelemetry compatible status code
 */
function convertStatusCodeToOTel(statusCode?: string): OTelSpanStatusCode {
  if (!statusCode) {
    return OTelSpanStatusCode.UNSET;
  }

  // Handle MLflow format -> OTel format conversion
  switch (statusCode) {
    case 'STATUS_CODE_OK':
      return OTelSpanStatusCode.OK;
    case 'STATUS_CODE_ERROR':
      return OTelSpanStatusCode.ERROR;
    case 'STATUS_CODE_UNSET':
      return OTelSpanStatusCode.UNSET;
    // Also handle OTel format directly
    case 'OK':
      return OTelSpanStatusCode.OK;
    case 'ERROR':
      return OTelSpanStatusCode.ERROR;
    case 'UNSET':
      return OTelSpanStatusCode.UNSET;
    default:
      return OTelSpanStatusCode.UNSET;
  }
}

export class LiveSpan extends Span {
  // Internal only flag to allow mutating the ended span
  allowMutatingEndedSpan: boolean = false;

  constructor(span: OTelSpan, traceId: string, span_type: SpanType) {
    super(span, true);
    this.setAttribute(SpanAttributeKey.TRACE_ID, traceId);
    this.setAttribute(SpanAttributeKey.SPAN_TYPE, span_type);
  }

  /**
   * Set the type of the span
   * @param spanType The type of the span
   */
  setSpanType(spanType: SpanType): void {
    this.setAttribute(SpanAttributeKey.SPAN_TYPE, spanType);
  }

  /**
   * Set inputs for the span
   * @param inputs Input data for the span
   */
  setInputs(inputs: any): void {
    this.setAttribute(SpanAttributeKey.INPUTS, inputs);
  }

  /**
   * Set outputs for the span
   * @param outputs Output data for the span
   */
  setOutputs(outputs: any): void {
    this.setAttribute(SpanAttributeKey.OUTPUTS, outputs);
  }

  /**
   * Set an attribute on the span
   * @param key Attribute key
   * @param value Attribute value
   */
  setAttribute(key: string, value: any): void {
    this._attributesRegistry.set(key, value, this.allowMutatingEndedSpan);
  }

  /**
   * Set multiple attributes on the span
   * @param attributes Object containing key-value pairs for attributes
   */
  setAttributes(attributes: Record<string, any>): void {
    if (!attributes || Object.keys(attributes).length === 0) {
      return;
    }

    Object.entries(attributes).forEach(([key, value]) => {
      this.setAttribute(key, value);
    });
  }

  /**
   * Add an event to the span
   * @param event Event object with name and attributes
   */
  addEvent(event: SpanEvent): void {
    // Convert BigInt timestamp to HrTime for OpenTelemetry
    const timeInput = convertNanoSecondsToHrTime(event.timestamp);
    this._span.addEvent(event.name, event.attributes, timeInput);
  }

  /**
   * Record an exception event to the span
   * @param error Error object
   */
  recordException(error: Error): void {
    this._span.recordException(error);
  }

  /**
   * Set the status of the span
   * @param status Status code or SpanStatus object
   * @param description Optional description for the status
   */
  setStatus(status: SpanStatus | SpanStatusCode | string, description?: string): void {
    if (status instanceof SpanStatus) {
      this._span.setStatus(status.toOtelStatus());
    } else if (typeof status === 'string') {
      const spanStatus = new SpanStatus(status as SpanStatusCode, description);
      this._span.setStatus(spanStatus.toOtelStatus());
    }
  }

  /**
   * End the span
   *
   * @param outputs Optional outputs to set before ending
   * @param attributes Optional attributes to set before ending
   * @param status Optional status code
   * @param endTimeNs Optional end time in nanoseconds
   */
  end(options?: {
    outputs?: any;
    attributes?: Record<string, any>;
    status?: SpanStatus | SpanStatusCode;
    endTimeNs?: number;
  }): void {
    try {
      if (options?.outputs != null) {
        this.setOutputs(options.outputs);
      }

      if (options?.attributes != null) {
        this.setAttributes(options.attributes);
      }

      if (options?.status != null) {
        this.setStatus(options.status);
      }

      // NB: In OpenTelemetry, status code remains UNSET if not explicitly set
      // by the user. However, there is no way to set the status when using
      // `trace` function wrapper. Therefore, we just automatically set the status
      // to OK if it is not ERROR.
      if (this.status.statusCode !== SpanStatusCode.ERROR) {
        this.setStatus(SpanStatusCode.OK);
      }

      // OTel SDK default end time to current time if not provided
      const endTime = options?.endTimeNs
        ? convertNanoSecondsToHrTime(options.endTimeNs)
        : undefined;
      this._span.end(endTime);
    } catch (error) {
      console.error(`Failed to end span ${this.spanId}: ${String(error)}.`);
    }
  }
}

/**
 * A no-operation span implementation that doesn't record anything
 */
export class NoOpSpan implements ISpan {
  readonly _span: any; // Use any for NoOp span to avoid type conflicts
  readonly _attributesRegistry: SpanAttributesRegistry;

  allowMutatingEndedSpan: boolean = false;

  constructor(span?: any) {
    // Create a minimal no-op span object
    this._span = span || {
      spanContext: () => ({
        spanId: INVALID_SPANID,
        traceId: INVALID_TRACEID
      }),
      attributes: {},
      events: []
    };
    this._attributesRegistry = new SpanAttributesRegistry(this._span as OTelSpan);
  }

  get traceId(): string {
    return NO_OP_SPAN_TRACE_ID;
  }
  get spanId(): string {
    return '';
  }
  get parentId(): string | null {
    return null;
  }
  get name(): string {
    return '';
  }
  get spanType(): SpanType {
    return SpanType.UNKNOWN;
  }
  get startTime(): HrTime {
    return [0, 0];
  }
  get endTime(): HrTime | null {
    return null;
  }
  get status(): SpanStatus {
    return new SpanStatus(SpanStatusCode.UNSET);
  }
  get inputs(): any {
    return null;
  }
  get outputs(): any {
    return null;
  }
  get attributes(): Record<string, any> {
    return {};
  }

  getAttribute(_key: string): any {
    return null;
  }

  // Implement all methods to do nothing
  setSpanType(_spanType: SpanType): void {}
  setInputs(_inputs: any): void {}
  setOutputs(_outputs: any): void {}
  setAttribute(_key: string, _value: any): void {}
  setAttributes(_attributes: Record<string, any>): void {}
  setStatus(_status: SpanStatus | SpanStatusCode | string, _description?: string): void {}
  addEvent(_event: SpanEvent): void {}
  recordException(_error: Error): void {}
  end(
    _outputs?: any,
    _attributes?: Record<string, any>,
    _status?: SpanStatus | SpanStatusCode,
    _endTimeNs?: number
  ): void {}

  get events(): SpanEvent[] {
    return [];
  }

  toJson(): SerializedSpan {
    return {
      trace_id: NO_OP_SPAN_TRACE_ID,
      span_id: '',
      parent_span_id: '',
      name: '',
      start_time_unix_nano: 0n,
      end_time_unix_nano: null,
      status: { code: 'UNSET', message: '' },
      attributes: {},
      events: []
    };
  }
}

export interface SerializedSpan {
  trace_id: string;
  span_id: string;
  parent_span_id: string;
  name: string;
  // Use bigint for nanosecond timestamps to maintain precision
  start_time_unix_nano: bigint;
  end_time_unix_nano: bigint | null;
  status: {
    code: string;
    message: string;
  };
  attributes: Record<string, any>;
  events: {
    name: string;
    time_unix_nano: bigint;
    attributes: Record<string, any>;
  }[];
}

/**
 * A utility class to manage the span attributes.
 * In MLflow users can add arbitrary key-value pairs to the span attributes, however,
 * OpenTelemetry only allows a limited set of types to be stored in the attribute values.
 * Therefore, we serialize all values into JSON string before storing them in the span.
 * This class provides simple getter and setter methods to interact with the span attributes
 * without worrying about the serde process.
 */
class SpanAttributesRegistry {
  private readonly _span: OTelSpan;

  constructor(otelSpan: OTelSpan) {
    this._span = otelSpan;
  }

  /**
   * Get all attributes as a dictionary
   */
  getAll(): Record<string, any> {
    const result: Record<string, any> = {};
    if (this._span.attributes) {
      Object.keys(this._span.attributes).forEach((key) => {
        result[key] = this.get(key);
      });
    }
    return result;
  }

  /**
   * Get a single attribute value
   */
  get(key: string): any {
    const serializedValue = this._span.attributes?.[key];
    if (serializedValue && typeof serializedValue === 'string') {
      try {
        return JSON.parse(serializedValue);
      } catch (e) {
        // If JSON.parse fails, this might be a raw string value or
        // the span was created from JSON (attributes already parsed)
        // In that case, return the value as-is
        return serializedValue;
      }
    }
    return serializedValue;
  }

  /**
   * Set a single attribute value
   */
  set(key: string, value: any, allowMutatingEndedSpan: boolean = false): void {
    if (typeof key !== 'string') {
      console.warn(`Attribute key must be a string, but got ${typeof key}. Skipping.`);
      return;
    }

    if (allowMutatingEndedSpan && this._span.ended) {
      // Directly set the attribute value to bypass the isSpanEnded check.
      this._span.attributes[key] = value;
      return;
    }

    // NB: OpenTelemetry attribute can store not only string but also a few primitives like
    // int, float, bool, and list of them. However, we serialize all into JSON string here
    // for the simplicity in deserialization process.
    this._span.setAttribute(key, safeJsonStringify(value));
  }
}

/**
 * A cache-enabled version of the SpanAttributesRegistry.
 * The caching helps to avoid the redundant deserialization of the attribute, however, it does
 * not handle the value change well. Therefore, this class should only be used for the persisted
 * spans that are immutable, and thus implemented as a subclass of SpanAttributesRegistry.
 */
class CachedSpanAttributesRegistry extends SpanAttributesRegistry {
  private readonly _cache = new Map<string, any>();

  /**
   * Get a single attribute value with LRU caching (maxsize=128)
   */
  get(key: string): any {
    if (this._cache.has(key)) {
      // Move to end (most recently used)
      const value = this._cache.get(key);
      this._cache.delete(key);
      this._cache.set(key, value);
      return value;
    }

    const value = super.get(key);

    // Implement LRU eviction
    if (this._cache.size >= 128) {
      // Remove least recently used (first entry)
      const firstKey = this._cache.keys().next().value;
      if (firstKey != null) {
        this._cache.delete(firstKey);
      }
    }

    this._cache.set(key, value);
    return value;
  }

  /**
   * Set operation is not allowed for cached registry (immutable spans)
   */
  set(_key: string, _value: any, _allowMutatingEndedSpan: boolean = false): void {
    throw new Error('The attributes of the immutable span must not be updated.');
  }
}

/**
 * Factory function to create a span object.
 */
export function createMlflowSpan(
  otelSpan: any,
  traceId: string,
  spanType?: string
): NoOpSpan | Span | LiveSpan {
  // NonRecordingSpan always has a spanId of '0000000000000000'
  // https://github.com/open-telemetry/opentelemetry-js/blob/f2cfd1327a5b131ea795301b10877291aac4e6f5/api/src/trace/invalid-span-constants.ts#L23C32-L23C48
  /* eslint-disable @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-unsafe-call */
  if (!otelSpan || otelSpan.spanContext().spanId === INVALID_SPANID) {
    return new NoOpSpan(otelSpan);
  }

  // If the span is completed, it should be immutable.
  if (otelSpan.ended) {
    return new Span(otelSpan);
  }

  return new LiveSpan(otelSpan, traceId, (spanType as SpanType) || SpanType.UNKNOWN);
  /* eslint-enable @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-unsafe-call */
}
