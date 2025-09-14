import type { HrTime } from '@opentelemetry/api';
import {
  convertNanoSecondsToHrTime,
  convertHrTimeToNanoSeconds,
  encodeSpanIdToBase64,
  encodeTraceIdToBase64,
  decodeIdFromBase64,
  deduplicateSpanNamesInPlace,
  mapArgsToObject
} from '../../../src/core/utils';
import { createTestSpan } from '../../helper';
import { LiveSpan } from '../../../src/core/entities/span';

describe('utils', () => {
  describe('convertNanoSecondsToHrTime', () => {
    // Using table-driven tests with test.each for time conversion
    test.each([
      {
        description: 'small nanosecond values',
        input: 123456789,
        expected: [0, 123456789]
      },
      {
        description: 'values exactly at 1 second',
        input: 1_000_000_000,
        expected: [1, 0]
      },
      {
        description: 'zero',
        input: 0,
        expected: [0, 0]
      }
    ])('should convert $description correctly', ({ input, expected }) => {
      const result = convertNanoSecondsToHrTime(input);
      expect(result).toEqual(expected);
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

      testValues.forEach((nanos) => {
        const hrTime = convertNanoSecondsToHrTime(nanos);
        const result = convertHrTimeToNanoSeconds(hrTime);
        expect(result).toBe(BigInt(nanos));
      });
    });
  });

  describe('encodeSpanIdToBase64', () => {
    // Using array syntax for test.each (alternative to object syntax)
    test.each([
      ['standard 16-character hex span ID', '0123456789abcdef', 'ASNFZ4mrze8='],
      ['short span IDs with zero padding', 'abc', 'AAAAAAAACrw='],
      ['all zeros', '0000000000000000', 'AAAAAAAAAAA='],
      ['all F characters', 'ffffffffffffffff', '//////////8=']
    ])('should encode %s', (description, spanId, expected) => {
      const result = encodeSpanIdToBase64(spanId);
      expect(result).toBe(expected);
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

      testSpanIds.forEach((spanId) => {
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

      testTraceIds.forEach((traceId) => {
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
    const spans = [createTestSpan('red'), createTestSpan('red')];

    deduplicateSpanNamesInPlace(spans);

    expect(spans[0].name).toBe('red_1');
    expect(spans[1].name).toBe('red_2');
  });

  it('should deduplicate only duplicate names, leaving unique names unchanged', () => {
    const spans = [createTestSpan('red'), createTestSpan('red'), createTestSpan('blue')];

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
    const spans = [createTestSpan('red'), createTestSpan('blue'), createTestSpan('green')];

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

describe('mapArgsToObject', () => {
  // Basic argument mapping test cases
  test.each([
    {
      description: 'regular functions',
      func: function add(a: number, b: number) {
        return a + b;
      },
      args: [5, 10],
      expected: { a: 5, b: 10 }
    },
    {
      description: 'arrow functions',
      func: (x: number, y: number) => x * y,
      args: [3, 4],
      expected: { x: 3, y: 4 }
    },
    {
      description: 'single parameter functions',
      func: (value: number) => value * 2,
      args: [7],
      expected: { value: 7 }
    },
    {
      description: 'functions with no parameters',
      func: () => 42,
      args: [],
      expected: {}
    },
    {
      description: 'functions with default parameters',
      func: function withDefaults(name: string, greeting: string = 'Hello') {
        return greeting + ' ' + name;
      },
      args: ['World'],
      expected: { name: 'World' }
    },
    {
      description: 'functions with type annotations',
      func: function typed(id: number, name: string, active: boolean) {
        return { id, name, active };
      },
      args: [123, 'John', true],
      expected: { id: 123, name: 'John', active: true }
    },
    {
      description: 'anonymous functions',
      func: function (first: string, second: number) {
        return first + second;
      },
      args: ['hello', 42],
      expected: { first: 'hello', second: 42 }
    },
    {
      description: 'fewer arguments than parameters',
      func: function threeParams(a: number, b: number, c: number) {
        return a + b + c;
      },
      args: [1, 2],
      expected: { a: 1, b: 2 }
    },
    {
      description: 'more arguments than parameters',
      func: function twoParams(a: number, b: number) {
        return a + b;
      },
      args: [1, 2, 3, 4],
      expected: { a: 1, b: 2 }
    },
    {
      description: 'complex argument types (objects, arrays)',
      func: function complex(obj: object, arr: any[], str: string) {
        return { obj, arr, str };
      },
      args: [{ key: 'value' }, [1, 2, 3], 'test'],
      expected: { obj: { key: 'value' }, arr: [1, 2, 3], str: 'test' }
    },
    {
      description: 'null and undefined arguments',
      func: function nullable(a: any, b: any, c: any) {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-return
        return a + b + c;
      },
      args: [null, undefined, 'value'],
      expected: { a: null, b: undefined, c: 'value' }
    },
    {
      description: 'functions with destructured parameters gracefully',
      func: function withDestructuring({ prop }: any, normal: string) {
        return prop + normal;
      },
      args: [{ prop: 'value' }, 'normal'],
      expected: { normal: { prop: 'value' } }
    },
    {
      description: 'fallback to args array when parameter extraction fails',
      // eslint-disable-next-line @typescript-eslint/no-implied-eval
      func: new Function('return arguments[0] + arguments[1];'),
      args: [1, 2],
      expected: { args: [1, 2] }
    },
    {
      description: 'empty object for no parameters and no arguments',
      func: () => {},
      args: [],
      expected: {}
    },
    {
      description: 'edge case with only whitespace parameters',
      func: () => {},
      args: [],
      expected: {}
    },
    {
      description: 'functions with object destructuring parameters (JS/TS kwarg-only pattern)',
      func: ({ a, b }: { a: number; b: number }) => a + b,
      args: [{ a: 5, b: 10 }],
      expected: { args: [{ a: 5, b: 10 }] }
    }
  ])('should handle $description', ({ func, args, expected }) => {
    const result = mapArgsToObject(func, args);
    expect(result).toEqual(expected);
  });
});
