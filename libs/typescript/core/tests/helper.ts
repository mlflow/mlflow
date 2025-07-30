import { LiveSpan } from '../src/core/entities/span';
import { SpanType } from '../src/core/constants';

/**
 * Port and tracking URI for the local MLflow server used for testing.
 * If the server is not running, jest.global-setup.ts will start it.
 */
export const TEST_PORT = 5000;
export const TEST_TRACKING_URI = `http://localhost:${TEST_PORT}`;

/**
 * Mock OpenTelemetry span class for testing
 */
export class MockOtelSpan {
  name: string;
  attributes: Record<string, any>;
  spanId: string;
  traceId: string;

  constructor(
    name: string = 'test-span',
    spanId: string = 'test-span-id',
    traceId: string = 'test-trace-id'
  ) {
    this.name = name;
    this.spanId = spanId;
    this.traceId = traceId;
    this.attributes = {};
  }

  getAttribute(key: string): any {
    return this.attributes[key];
  }

  setAttribute(key: string, value: any): void {
    this.attributes[key] = value;
  }

  spanContext() {
    return {
      spanId: this.spanId,
      traceId: this.traceId
    };
  }
}

/**
 * Create a mock OpenTelemetry span with the given parameters
 */
export function createMockOtelSpan(
  name: string = 'test-span',
  spanId: string = 'test-span-id',
  traceId: string = 'test-trace-id'
): MockOtelSpan {
  return new MockOtelSpan(name, spanId, traceId);
}

/**
 * Create a test LiveSpan with mock OpenTelemetry span
 */
export function createTestSpan(
  name: string = 'test-span',
  traceId: string = 'test-trace-id',
  spanId: string = 'test-span-id',
  spanType: SpanType = SpanType.UNKNOWN
): LiveSpan {
  const mockOtelSpan = createMockOtelSpan(name, spanId, traceId);
  // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
  return new LiveSpan(mockOtelSpan as any, traceId, spanType);
}
