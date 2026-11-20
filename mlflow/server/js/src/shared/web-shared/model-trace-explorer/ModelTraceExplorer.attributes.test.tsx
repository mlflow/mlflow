import { describe, expect, it } from '@jest/globals';

import { convertOtelAttributesToMap as convertLegacy } from './ModelTraceExplorer.utils';
import { convertOtelAttributesToMap as convertV2 } from './v2/ModelTraceExplorer.utils';

describe.each([
  ['legacy', convertLegacy],
  ['v2', convertV2],
])('%s OTLP attribute maps', (_version, convert) => {
  it('retains ordering, duplicate values, reserved keys and nested values for spans, events and links', () => {
    const attributes = [
      { key: 'a', value: { string_value: 'first' } },
      { key: 'b', value: { bool_value: false } },
      { key: '__proto__', value: { string_value: 'ordinary attribute' } },
      { key: 'constructor', value: { int_value: 0 } },
      { key: 'a', value: { array_value: { values: [{ string_value: 'café 日本語 🙂' }] } } },
      { key: '', value: { string_value: 'skip empty key' } },
      { key: 'missing' },
    ];
    const span = {
      span_id: '0000000000000001',
      attributes,
      events: [{ name: 'event', attributes }],
      links: [{ trace_id: '0'.repeat(32), span_id: '0000000000000002', attributes }],
    } as any;
    const original = JSON.stringify(span);
    const result = convert(span);
    for (const map of [result.attributes, result.events?.[0].attributes, result.links?.[0].attributes]) {
      expect(Object.getPrototypeOf(map)).toBe(Object.prototype);
      expect(Object.entries(map ?? {})).toEqual([
        ['a', ['café 日本語 🙂']],
        ['b', false],
        ['__proto__', 'ordinary attribute'],
        ['constructor', 0],
      ]);
    }
    expect(JSON.stringify(span)).toBe(original);
  });

  it('keeps every entry of a wide attribute list', () => {
    const attributes = Array.from({ length: 4000 }, (_, index) => ({
      key: `attribute-${index}`,
      value: { int_value: index },
    }));
    const result = convert({ span_id: '0000000000000001', attributes } as any);
    expect(Object.keys(result.attributes ?? {})).toHaveLength(4000);
    expect(
      Object.entries(result.attributes ?? {}).every(
        ([key, value], index) => key === `attribute-${index}` && value === index,
      ),
    ).toBe(true);
  });
});
