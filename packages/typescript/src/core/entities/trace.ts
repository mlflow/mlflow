import { SerializedTraceInfo, TraceInfo } from './trace_info';
import { SerializedTraceData, TraceData } from './trace_data';

/**
 * Represents a complete trace with metadata and span data
 */
export class Trace {
  /**
   * Trace metadata
   */
  info: TraceInfo;

  /**
   * Trace data containing spans
   */
  data: TraceData;

  /**
   * Create a new Trace instance
   * @param info Trace metadata
   * @param data Trace data containing spans
   */
  constructor(info: TraceInfo, data: TraceData) {
    this.info = info;
    this.data = data;
  }

  /**
   * Convert this Trace instance to JSON format
   * @returns JSON object representation of the Trace
   */
  toJson(): SerializedTrace {
    return {
      info: this.info.toJson(),
      data: this.data.toJson()
    };
  }

  /**
   * Create a Trace instance from JSON data
   * @param json JSON object containing trace data
   * @returns Trace instance
   */
  static fromJson(json: SerializedTrace): Trace {
    const info = TraceInfo.fromJson(json.info);
    const data = TraceData.fromJson(json.data);
    return new Trace(info, data);
  }
}

interface SerializedTrace {
  info: SerializedTraceInfo;
  data: SerializedTraceData;
}
