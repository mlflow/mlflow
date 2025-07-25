import { SpanStatusCode as OTelSpanStatusCode } from '@opentelemetry/api';
import { SpanStatus, SpanStatusCode } from '../../../src/core/entities/span_status';

describe('SpanStatus', () => {
  describe('initialization', () => {
    // Test both enum and string initialization (parameterized test equivalent)
    const testCases = [
      { input: 'STATUS_CODE_OK', expected: SpanStatusCode.OK },
      { input: 'STATUS_CODE_ERROR', expected: SpanStatusCode.ERROR },
      { input: 'STATUS_CODE_UNSET', expected: SpanStatusCode.UNSET }
    ];

    testCases.forEach(({ input, expected }) => {
      it(`should initialize with status code ${input}`, () => {
        const spanStatus = new SpanStatus(input as SpanStatusCode, 'test');
        expect(spanStatus.statusCode).toBe(expected);
        expect(spanStatus.description).toBe('test');
      });
    });
  });

  describe('OpenTelemetry status conversion', () => {
    const conversionTestCases = [
      {
        mlflowStatus: SpanStatusCode.OK,
        otelStatus: OTelSpanStatusCode.OK
      },
      {
        mlflowStatus: SpanStatusCode.ERROR,
        otelStatus: OTelSpanStatusCode.ERROR
      },
      {
        mlflowStatus: SpanStatusCode.UNSET,
        otelStatus: OTelSpanStatusCode.UNSET
      }
    ];

    conversionTestCases.forEach(({ mlflowStatus, otelStatus }) => {
      it(`should convert ${mlflowStatus} to OpenTelemetry status correctly`, () => {
        const spanStatus = new SpanStatus(mlflowStatus);
        const otelStatusResult = spanStatus.toOtelStatus();

        expect(otelStatusResult.code).toBe(otelStatus);
      });
    });
  });

  describe('toJson round-trip serialization', () => {
    it('should serialize and recreate status with all properties', () => {
      const originalStatus = new SpanStatus(SpanStatusCode.ERROR, 'Something went wrong');

      const json = originalStatus.toJson();

      // Verify JSON structure
      expect(json).toEqual({
        status_code: SpanStatusCode.ERROR,
        description: 'Something went wrong'
      });

      // Create new status from JSON data
      const recreatedStatus = new SpanStatus(json.status_code as SpanStatusCode, json.description);

      // Verify round-trip preservation
      expect(recreatedStatus.statusCode).toBe(originalStatus.statusCode);
      expect(recreatedStatus.description).toBe(originalStatus.description);
      expect(recreatedStatus.toJson()).toEqual(originalStatus.toJson());
    });

    it('should handle status with different status codes', () => {
      const testCases = [
        { code: SpanStatusCode.OK, description: 'All good' },
        { code: SpanStatusCode.ERROR, description: 'Failed operation' },
        { code: SpanStatusCode.UNSET, description: '' }
      ];

      testCases.forEach(({ code, description }) => {
        const originalStatus = new SpanStatus(code, description);
        const json = originalStatus.toJson();
        const recreatedStatus = new SpanStatus(
          json.status_code as SpanStatusCode,
          json.description
        );

        expect(recreatedStatus.statusCode).toBe(originalStatus.statusCode);
        expect(recreatedStatus.description).toBe(originalStatus.description);
        expect(recreatedStatus.toJson()).toEqual(originalStatus.toJson());
      });
    });

    it('should handle status with minimal properties', () => {
      const originalStatus = new SpanStatus(SpanStatusCode.OK);

      const json = originalStatus.toJson();

      expect(json).toEqual({
        status_code: SpanStatusCode.OK,
        description: ''
      });

      const recreatedStatus = new SpanStatus(json.status_code as SpanStatusCode, json.description);

      expect(recreatedStatus.statusCode).toBe(originalStatus.statusCode);
      expect(recreatedStatus.description).toBe(originalStatus.description);
      expect(recreatedStatus.toJson()).toEqual(originalStatus.toJson());
    });
  });
});
