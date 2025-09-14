import { SpanStatusCode } from '@opentelemetry/api';

/**
 * Enum representing the state of a trace
 */
export enum TraceState {
  /**
   * Unspecified trace state
   */
  STATE_UNSPECIFIED = 'STATE_UNSPECIFIED',

  /**
   * Trace successfully completed
   */
  OK = 'OK',

  /**
   * Trace encountered an error
   */
  ERROR = 'ERROR',

  /**
   * Trace is currently in progress
   */
  IN_PROGRESS = 'IN_PROGRESS'
}

/**
 * Convert OpenTelemetry status code to MLflow TraceState
 * @param statusCode OpenTelemetry status code
 */
export function fromOtelStatus(statusCode: SpanStatusCode): TraceState {
  switch (statusCode) {
    case SpanStatusCode.OK:
      return TraceState.OK;
    case SpanStatusCode.ERROR:
      return TraceState.ERROR;
    default:
      return TraceState.STATE_UNSPECIFIED;
  }
}
