import type { HrTime } from '@opentelemetry/api';
import {
  convertNanoSecondsToHrTime,
  convertHrTimeToNanoSeconds,
  encodeSpanIdToBase64,
  encodeTraceIdToBase64,
  decodeIdFromBase64,
  deduplicateSpanNamesInPlace
} from '../../../src/core/utils';
import { createTestSpan } from '../../helper';
import { LiveSpan } from '../../../src/core/entities/span';

describe('utils', () => {
  describe('convertNanoSecondsToHrTime', () => {
    it('should convert small nanosecond values correctly', () => {
      const result = convertNanoSecondsToHrTime(123456789);
      expect(result).toEqual([0, 123456789]);
    });

    it('should convert values exactly at 1 second', () => {
      const result = convertNanoSecondsToHrTime(1_000_000_000);
      expect(result).toEqual([1, 0]);
    });

    it('should convert large nanosecond values correctly', () => {
      // Note: JavaScript loses precision with very large numbers
      // The computation seconds * 1e9 + nanosInSecond loses precision
      const seconds = 1234567890;
      const nanosInSecond = 123456789;
      const nanos = seconds * 1e9 + nanosInSecond;
      const result = convertNanoSecondsToHrTime(nanos);
      expect(result[0]).toBe(seconds);
      // Due to precision loss in the calculation above
      expect(result[1]).toBe(123456768);
    });

    it('should handle zero', () => {
      const result = convertNanoSecondsToHrTime(0);
      expect(result).toEqual([0, 0]);
    });

    it('should handle maximum safe integer', () => {
      const maxSafeInt = Number.MAX_SAFE_INTEGER;
      const result = convertNanoSecondsToHrTime(maxSafeInt);
      expect(result[0]).toBe(Math.floor(maxSafeInt / 1e9));
      expect(result[1]).toBe(maxSafeInt % 1e9);
    });
  });

  describe('convertHrTimeToNanoSeconds', () => {
    it('should convert HrTime with zero seconds correctly', () => {
      const hrTime: HrTime = [0, 123456789];
      const result = convertHrTimeToNanoSeconds(hrTime);
      expect(result).toBe(123456789n);
    });

    it('should convert HrTime with seconds correctly', () => {
      const hrTime: HrTime = [5, 500000000];
      const result = convertHrTimeToNanoSeconds(hrTime);
      expect(result).toBe(5_500_000_000n);
    });

    it('should handle zero HrTime', () => {
      const hrTime: HrTime = [0, 0];
      const result = convertHrTimeToNanoSeconds(hrTime);
      expect(result).toBe(0n);
    });

    it('should handle large HrTime values', () => {
      const hrTime: HrTime = [1234567890, 123456789];
      const result = convertHrTimeToNanoSeconds(hrTime);
      const expected = 1234567890n * 1_000_000_000n + 123456789n;
      expect(result).toBe(expected);
    });

    it('should be reversible with convertNanoSecondsToHrTime', () => {
      const testValues = [0, 123456789, 1_000_000_000, 5_500_000_000];

      testValues.forEach(nanos => {
        const hrTime = convertNanoSecondsToHrTime(nanos);
        const result = convertHrTimeToNanoSeconds(hrTime);
        expect(result).toBe(BigInt(nanos));
      });
    });
  });

  describe('encodeSpanIdToBase64', () => {
    it('should encode a standard 16-character hex span ID', () => {
      const spanId = '0123456789abcdef';
      const result = encodeSpanIdToBase64(spanId);
      expect(result).toBe('ASNFZ4mrze8=');
    });

    it('should pad short span IDs with zeros', () => {
      const spanId = 'abc';
      const result = encodeSpanIdToBase64(spanId);
      // Should be padded to '0000000000000abc'
      expect(result).toBe('AAAAAAAACrw=');
    });

    it('should handle all zeros', () => {
      const spanId = '0000000000000000';
      const result = encodeSpanIdToBase64(spanId);
      expect(result).toBe('AAAAAAAAAAA=');
    });

    it('should handle all F characters', () => {
      const spanId = 'ffffffffffffffff';
      const result = encodeSpanIdToBase64(spanId);
      expect(result).toBe('//////////8=');
    });

    it('should handle mixed case hex strings', () => {
      const spanId = 'AbCdEf1234567890';
      const result = encodeSpanIdToBase64(spanId);
      // Should work the same as lowercase
      const lowercase = encodeSpanIdToBase64('abcdef1234567890');
      expect(result).toBe(lowercase);
    });
  });

  describe('encodeTraceIdToBase64', () => {
    it('should encode a standard 32-character hex trace ID', () => {
      const traceId = '0123456789abcdef0123456789abcdef';
      const result = encodeTraceIdToBase64(traceId);
      expect(result).toBe('ASNFZ4mrze8BI0VniavN7w==');
    });

    it('should pad short trace IDs with zeros', () => {
      const traceId = 'abc';
      const result = encodeTraceIdToBase64(traceId);
      // Should be padded to '00000000000000000000000000000abc'
      expect(result).toBe('AAAAAAAAAAAAAAAAAAAKvA==');
    });

    it('should handle all zeros', () => {
      const traceId = '00000000000000000000000000000000';
      const result = encodeTraceIdToBase64(traceId);
      expect(result).toBe('AAAAAAAAAAAAAAAAAAAAAA==');
    });

    it('should handle all F characters', () => {
      const traceId = 'ffffffffffffffffffffffffffffffff';
      const result = encodeTraceIdToBase64(traceId);
      expect(result).toBe('/////////////////////w==');
    });

    it('should handle mixed case hex strings', () => {
      const traceId = 'AbCdEf1234567890AbCdEf1234567890';
      const result = encodeTraceIdToBase64(traceId);
      // Should work the same as lowercase
      const lowercase = encodeTraceIdToBase64('abcdef1234567890abcdef1234567890');
      expect(result).toBe(lowercase);
    });
  });

  describe('decodeIdFromBase64', () => {
    it('should decode a base64 encoded span ID back to hex', () => {
      const base64Id = 'ASNFZ4mrze8=';
      const result = decodeIdFromBase64(base64Id);
      expect(result).toBe('0123456789abcdef');
    });

    it('should decode a base64 encoded trace ID back to hex', () => {
      const base64Id = 'ASNFZ4mrze8BI0VniavN7w==';
      const result = decodeIdFromBase64(base64Id);
      expect(result).toBe('0123456789abcdef0123456789abcdef');
    });

    it('should handle all zeros', () => {
      const spanBase64 = 'AAAAAAAAAAA=';
      const traceBase64 = 'AAAAAAAAAAAAAAAAAAAAAA==';

      expect(decodeIdFromBase64(spanBase64)).toBe('0000000000000000');
      expect(decodeIdFromBase64(traceBase64)).toBe('00000000000000000000000000000000');
    });

    it('should handle all F values', () => {
      const spanBase64 = '//////////8=';
      const traceBase64 = '/////////////////////w==';

      expect(decodeIdFromBase64(spanBase64)).toBe('ffffffffffffffff');
      expect(decodeIdFromBase64(traceBase64)).toBe('ffffffffffffffffffffffffffffffff');
    });

    it('should be reversible with encodeSpanIdToBase64', () => {
      const testSpanIds = [
        '0123456789abcdef',
        '0000000000000000',
        'ffffffffffffffff',
        'deadbeef12345678'
      ];

      testSpanIds.forEach(spanId => {
        const encoded = encodeSpanIdToBase64(spanId);
        const decoded = decodeIdFromBase64(encoded);
        expect(decoded).toBe(spanId);
      });
    });

    it('should be reversible with encodeTraceIdToBase64', () => {
      const testTraceIds = [
        '0123456789abcdef0123456789abcdef',
        '00000000000000000000000000000000',
        'ffffffffffffffffffffffffffffffff',
        'deadbeef12345678deadbeef12345678'
      ];

      testTraceIds.forEach(traceId => {
        const encoded = encodeTraceIdToBase64(traceId);
        const decoded = decodeIdFromBase64(encoded);
        expect(decoded).toBe(traceId);
      });
    });

    it('should handle empty base64 string', () => {
      const result = decodeIdFromBase64('');
      expect(result).toBe('');
    });

    it('should handle single byte base64', () => {
      // Base64 for single byte 0xFF
      const result = decodeIdFromBase64('/w==');
      expect(result).toBe('ff');
    });
  });

  describe('Edge cases and integration', () => {
    it('should handle conversion chain for span IDs', () => {
      // Test that we can go from hex -> base64 -> hex without loss
      const originalHex = 'a1b2c3d4e5f67890';
      const base64 = encodeSpanIdToBase64(originalHex);
      const decodedHex = decodeIdFromBase64(base64);

      expect(decodedHex).toBe(originalHex);
    });

    it('should handle conversion chain for trace IDs', () => {
      // Test that we can go from hex -> base64 -> hex without loss
      const originalHex = 'a1b2c3d4e5f67890a1b2c3d4e5f67890';
      const base64 = encodeTraceIdToBase64(originalHex);
      const decodedHex = decodeIdFromBase64(base64);

      expect(decodedHex).toBe(originalHex);
    });

    it('should handle odd-length hex strings for span ID', () => {
      const oddHex = '12345'; // 5 characters
      const base64 = encodeSpanIdToBase64(oddHex);
      const decoded = decodeIdFromBase64(base64);

      // Should be padded to '0000000000012345'
      expect(decoded).toBe('0000000000012345');
    });

    it('should handle odd-length hex strings for trace ID', () => {
      const oddHex = '12345'; // 5 characters
      const base64 = encodeTraceIdToBase64(oddHex);
      const decoded = decodeIdFromBase64(base64);

      // Should be padded to '00000000000000000000000000012345'
      expect(decoded).toBe('00000000000000000000000000012345');
    });

    it('should handle very long hex strings by truncation', () => {
      // Span ID should only use first 16 chars
      const longHex = '0123456789abcdef0123456789abcdef0123456789abcdef';
      const spanBase64 = encodeSpanIdToBase64(longHex);
      const decodedSpan = decodeIdFromBase64(spanBase64);

      // Should only encode first 16 chars
      expect(decodedSpan).toBe('0123456789abcdef');
    });

    it('should produce consistent results across multiple calls', () => {
      const spanId = 'deadbeef12345678';
      const traceId = 'deadbeef12345678deadbeef12345678';

      // Multiple calls should produce identical results
      const spanBase64_1 = encodeSpanIdToBase64(spanId);
      const spanBase64_2 = encodeSpanIdToBase64(spanId);
      expect(spanBase64_1).toBe(spanBase64_2);

      const traceBase64_1 = encodeTraceIdToBase64(traceId);
      const traceBase64_2 = encodeTraceIdToBase64(traceId);
      expect(traceBase64_1).toBe(traceBase64_2);
    });
  });
});

describe('deduplicateSpanNamesInPlace', () => {
  it('should deduplicate spans with duplicate names', () => {
    const spans = [
      createTestSpan('red'),
      createTestSpan('red')
    ];

    deduplicateSpanNamesInPlace(spans);

    expect(spans[0].name).toBe('red_1');
    expect(spans[1].name).toBe('red_2');
  });

  it('should deduplicate only duplicate names, leaving unique names unchanged', () => {
    const spans = [
      createTestSpan('red'),
      createTestSpan('red'),
      createTestSpan('blue')
    ];

    deduplicateSpanNamesInPlace(spans);

    expect(spans[0].name).toBe('red_1');
    expect(spans[1].name).toBe('red_2');
    expect(spans[2].name).toBe('blue');
  });

  it('should handle multiple sets of duplicates', () => {
    const spans = [
      createTestSpan('red'),
      createTestSpan('blue'),
      createTestSpan('red'),
      createTestSpan('green'),
      createTestSpan('blue'),
      createTestSpan('red')
    ];
    deduplicateSpanNamesInPlace(spans);

    expect(spans[0].name).toBe('red_1');
    expect(spans[1].name).toBe('blue_1');
    expect(spans[2].name).toBe('red_2');
    expect(spans[3].name).toBe('green');
    expect(spans[4].name).toBe('blue_2');
    expect(spans[5].name).toBe('red_3');
  });

  it('should handle spans with no duplicates', () => {
    const spans = [
      createTestSpan('red'),
      createTestSpan('blue'),
      createTestSpan('green')
    ];

    deduplicateSpanNamesInPlace(spans);

    expect(spans[0].name).toBe('red');
    expect(spans[1].name).toBe('blue');
    expect(spans[2].name).toBe('green');
  });

  it('should handle empty array', () => {
    const spans: LiveSpan[] = [];

    expect(() => deduplicateSpanNamesInPlace(spans)).not.toThrow();
    expect(spans.length).toBe(0);
  });
});