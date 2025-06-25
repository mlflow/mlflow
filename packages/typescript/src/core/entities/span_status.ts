import { SpanStatus as OTelStatus, SpanStatusCode as OTelSpanStatusCode } from '@opentelemetry/api';

/**
 * MLflow Span Status module
 *
 * This module provides the MLflow SpanStatusCode enum and SpanStatus class,
 * matching the Python MLflow implementation.
 */

/**
 * Enum for status code of a span
 * Uses the same set of status codes as OTLP SpanStatusCode
 * https://github.com/open-telemetry/opentelemetry-proto/blob/189b2648d29aa6039aeb2
lemetry/proto/trace/v1/trace.proto#L314-L322
 */
export enum SpanStatusCode {
  /** Status is unset/unspecified */
  UNSET = 'STATUS_CODE_UNSET',
  /** The operation completed successfully */
  OK = 'STATUS_CODE_OK',
  /** The operation encountered an error */
  ERROR = 'STATUS_CODE_ERROR'
}

/**
 * Status of the span or the trace.
 */
export class SpanStatus {
  /**
   * The status code of the span or the trace.
   */
  readonly statusCode: SpanStatusCode;

  /**
   * Description of the status. This should be only set when the status is ERROR,
   * otherwise it will be ignored.
   */
  readonly description: string;

  /**
   * Create a new SpanStatus instance
   * @param statusCode The status code - must be one of SpanStatusCode enum values
   * @param description Optional description, typically used for ERROR status
   */
  constructor(statusCode: SpanStatusCode, description: string = '') {
    // If user provides a string status code, validate it and convert to enum
    this.statusCode = statusCode;
    this.description = description;
  }

  /**
   * Convert SpanStatus object to OpenTelemetry status object.
   */
  toOtelStatus(): OTelStatus {
    let otelStatusCode: OTelSpanStatusCode;

    switch (this.statusCode) {
      case SpanStatusCode.OK:
        otelStatusCode = OTelSpanStatusCode.OK;
        break;
      case SpanStatusCode.ERROR:
        otelStatusCode = OTelSpanStatusCode.ERROR;
        break;
      case SpanStatusCode.UNSET:
      default:
        otelStatusCode = OTelSpanStatusCode.UNSET;
        break;
    }

    return {
      code: otelStatusCode,
      message: this.description
    };
  }

  /**
   * Convert OpenTelemetry status object to SpanStatus object.
   */
  static fromOtelStatus(otelStatus: OTelStatus): SpanStatus {
    let statusCode: SpanStatusCode;

    switch (otelStatus.code) {
      case OTelSpanStatusCode.OK:
        statusCode = SpanStatusCode.OK;
        break;
      case OTelSpanStatusCode.ERROR:
        statusCode = SpanStatusCode.ERROR;
        break;
      case OTelSpanStatusCode.UNSET:
      default:
        statusCode = SpanStatusCode.UNSET;
        break;
    }

    return new SpanStatus(statusCode, otelStatus.message ?? '');
  }

  /**
   * Convert this SpanStatus to JSON format
   * @returns JSON object representation of the span status
   */
  toJson(): Record<string, string | SpanStatusCode> {
    return {
      status_code: this.statusCode,
      description: this.description
    };
  }
}
