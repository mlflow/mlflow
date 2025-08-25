import { ISpan, SerializedSpan, Span } from './span';

/**
 * Represents the spans and associated data for a trace
 */
export class TraceData {
  /**
   * The spans that make up this trace
   */
  spans: ISpan[];

  /**
   * Create a new TraceData instance
   * @param spans List of spans
   */
  constructor(spans: ISpan[] = []) {
    this.spans = spans;
  }

  /**
   * Convert this TraceData instance to JSON format
   * @returns JSON object representation of the TraceData
   */
  toJson(): SerializedTraceData {
    return {
      spans: this.spans.map((span) => span.toJson())
    };
  }

  /**
   * Create a TraceData instance from JSON data (following Python implementation)
   * @param json JSON object containing trace data
   * @returns TraceData instance
   */
  static fromJson(json: SerializedTraceData): TraceData {
    const spans: ISpan[] = json.spans.map((spanData) => Span.fromJson(spanData));
    return new TraceData(spans);
  }
}

export interface SerializedTraceData {
  spans: SerializedSpan[];
}
