/**
 * MLflow Span Event Entities
 *
 * This module provides TypeScript implementations of MLflow span event entities,
 * compatible with OpenTelemetry for recording specific occurrences during span execution.
 */

/**
 * Type definition for attribute values that can be stored in span events.
 * Compatible with OpenTelemetry attribute value types.
 */
export type AttributeValue = string | number | boolean | string[] | number[] | boolean[];

/**
 * Parameters for creating a SpanEvent instance.
 */
export interface SpanEventParams {
  /** Name of the event */
  name: string;
  /**
   * The exact time the event occurred, measured in nanoseconds since epoch.
   * If not provided, the current time will be used.
   */
  timestamp?: bigint;
  /**
   * A collection of key-value pairs representing detailed attributes of the event,
   * such as exception stack traces or other contextual information.
   */
  attributes?: Record<string, AttributeValue>;
}

/**
 * Represents an event that records specific occurrences or moments in time
 * during a span, such as an exception being thrown. Compatible with OpenTelemetry.
 *
 * SpanEvents are used to capture important moments during span execution,
 * providing detailed context about what happened and when.
 *
 * @example
 * ```typescript
 * // Create a custom event
 * const event = new SpanEvent({
 *   name: 'user_action',
 *   attributes: {
 *     'action.type': 'click',
 *     'element.id': 'submit-button'
 *   }
 * });
 *
 * // Create an event from an exception
 * const errorEvent = SpanEvent.fromException(new Error('Something went wrong'));
 * ```
 */
export class SpanEvent {
  /** Name of the event */
  readonly name: string;

  /**
   * The exact time the event occurred, measured in nanosecond since epoch.
   * Defaults to current time if not provided during construction.
   */
  readonly timestamp: bigint;

  /**
   * A collection of key-value pairs representing detailed attributes of the event.
   * Attributes provide contextual information about the event.
   */
  readonly attributes: Record<string, AttributeValue>;

  /**
   * Creates a new SpanEvent instance.
   *
   * @param params - Event parameters including name, optional timestamp, and attributes
   *
   * @example
   * ```typescript
   * const event = new SpanEvent({
   *   name: 'database_query',
   *   attributes: {
   *     'db.statement': 'SELECT * FROM users',
   *     'db.duration_ms': 150
   *   }
   * });
   * ```
   */
  constructor(params: SpanEventParams) {
    this.name = params.name;
    this.timestamp = params.timestamp ?? this.getCurrentTimeNano();
    this.attributes = params.attributes ?? {};
  }

  /**
   * Creates a span event from an exception.
   *
   * This is a convenience method for creating events that represent exceptions
   * or errors that occurred during span execution. The event will include
   * standard exception attributes like message, type, and stack trace.
   *
   * @param exception - The exception to create an event from
   * @returns New SpanEvent instance representing the exception
   *
   * @example
   * ```typescript
   * try {
   *   // Some operation that might fail
   *   throw new Error('Database connection failed');
   * } catch (error) {
   *   const errorEvent = SpanEvent.fromException(error);
   *   span.addEvent(errorEvent);
   * }
   * ```
   */
  static fromException(exception: Error): SpanEvent {
    const stackTrace = this.getStackTrace(exception);

    return new SpanEvent({
      name: 'exception',
      attributes: {
        'exception.message': exception.message,
        'exception.type': exception.name,
        'exception.stacktrace': stackTrace
      }
    });
  }

  /**
   * Gets the stack trace from an error object.
   *
   * @param error - The error to extract stack trace from
   * @returns Stack trace as a string, or error representation if stack trace unavailable
   */
  private static getStackTrace(error: Error): string {
    try {
      return error.stack ?? String(error);
    } catch {
      // Fallback if stack trace extraction fails
      return String(error);
    }
  }

  /**
   * Convert this SpanEvent to JSON format
   * @returns JSON object representation of the span event
   */
  toJson(): Record<string, string | bigint | Record<string, AttributeValue>> {
    return {
      name: this.name,
      timestamp: this.timestamp,
      attributes: this.attributes
    };
  }

  /**
   * Gets the current time in nanoseconds since epoch.
   *
   * @returns Current timestamp in nanoseconds
   */
  private getCurrentTimeNano(): bigint {
    return BigInt(Date.now()) * BigInt(1e6);
  }
}
