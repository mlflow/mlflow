import { describe, expect, it, jest } from '@jest/globals';
import { dump } from 'js-yaml';

import { formatJsonStringAsYaml, getYamlCollectionPreview, yamlJsonMayContain } from './ModelTraceExplorerYaml.utils';

jest.mock('js-yaml', () => {
  const actual = jest.requireActual<typeof import('js-yaml')>('js-yaml');
  return { ...actual, dump: jest.fn(actual.dump) };
});

const prefix = (text: string) => text.split('\n').slice(0, 10).join('\n').slice(0, 300);

describe('YAML collection previews', () => {
  it.each([
    Array.from({ length: 500 }, (_, index) => ({ index, message: 'café 日本語 🙂' })),
    Object.fromEntries(Array.from({ length: 500 }, (_, index) => [`key_${index}`, { value: index }])),
    Array.from({ length: 200 }, (_, index) => [[{ index, tags: ['a', 'b'] }]]),
    Array.from({ length: 500 }, () => ({ text: 'a\n  b\n\nc', punctuation: 'true: [quoted]' })),
    Array.from({ length: 500 }, () => ({ mlflow_yaml_preview_cut: 'mlflow_yaml_preview_cut_' })),
    JSON.parse(
      '{"__proto__":{"value":1},"constructor":"original","records":' + JSON.stringify(Array(500).fill(1)) + '}',
    ),
  ])('keeps the exact full-dump prefix for truncated collection %#', (value) => {
    const data = JSON.stringify(value);
    expect(data.length).toBeGreaterThan(300);
    const preview = getYamlCollectionPreview(data, 10, 300);
    expect(preview?.isComplete).toBe(false);
    expect(prefix(preview!.text)).toBe(prefix(formatJsonStringAsYaml(data)));
    expect(formatJsonStringAsYaml(data).startsWith(preview!.text)).toBe(true);
  });

  it.each([null, true, 1, 'café 日本語 🙂', { key: [1, 'a\nb', {}] }])('preserves complete small values', (value) => {
    const data = JSON.stringify(value);
    expect(getYamlCollectionPreview(data, 10, 300)).toEqual({
      text: formatJsonStringAsYaml(data),
      isComplete: true,
    });
  });

  it.each(['word '.repeat(5000), 'line\n'.repeat(5000) + 'tail\n', 'line\n'.repeat(5000) + 'tail'])(
    'bounds large scalar strings while preserving the preview',
    (value) => {
      const data = JSON.stringify(value);
      const preview = getYamlCollectionPreview(data, 10, 300);
      const dumpedValue = jest.mocked(dump).mock.calls.at(-1)?.[0];

      expect(preview?.isComplete).toBe(false);
      expect(typeof dumpedValue === 'string' && dumpedValue.length).toBeLessThanOrEqual(422);
      expect(prefix(preview!.text)).toBe(prefix(formatJsonStringAsYaml(data)));
    },
  );

  it.each([
    { nested: { value: 'line\n'.repeat(5000) + 'tail\n' } },
    Array.from({ length: 20 }, (_, index) => `value_${index}_${'x'.repeat(50)}`),
    { [`key_${'x'.repeat(5000)}`]: true },
  ])('bounds nested strings and keys while preserving the preview %#', (value) => {
    const data = JSON.stringify(value);
    const preview = getYamlCollectionPreview(data, 10, 300);
    const dumpedValue = jest.mocked(dump).mock.calls.at(-1)?.[0];
    const dumpedStrings = JSON.stringify(dumpedValue);

    expect(preview?.isComplete).toBe(false);
    expect(dumpedStrings.length).toBeLessThan(1200);
    expect(prefix(preview!.text)).toBe(prefix(formatJsonStringAsYaml(data)));
  });

  it('keeps the full formatter for scalar styles that cannot be truncated safely', () => {
    const value = 'line\n'.repeat(5000) + '\x07';
    const data = JSON.stringify(value);

    expect(prefix(getYamlCollectionPreview(data, 10, 300)!.text)).toBe(prefix(formatJsonStringAsYaml(data)));
    expect(jest.mocked(dump).mock.calls.at(-2)?.[0]).toBe(value);
  });

  it('keeps the existing invalid-JSON fallback', () => {
    expect(getYamlCollectionPreview('{invalid', 10, 300)).toBeUndefined();
    expect(formatJsonStringAsYaml('{invalid')).toBe('{invalid');
  });

  it('bounds collision checks for repeated marker-like scalar content', () => {
    const data = JSON.stringify(['mlflow_yaml_preview_cut' + '_'.repeat(200), ...Array(500).fill('value')]);
    const includes = jest.spyOn(String.prototype, 'includes');
    try {
      const preview = getYamlCollectionPreview(data, 10, 300);
      const markerChecks = includes.mock.calls.filter(([search]) =>
        search.startsWith('mlflow_yaml_preview_cut'),
      ).length;
      expect(markerChecks).toBeLessThan(100);
      expect(formatJsonStringAsYaml(data).startsWith(preview!.text)).toBe(true);
    } finally {
      includes.mockRestore();
    }
  });

  it('preserves the formatter fallback for native limits but exposes unexpected errors', () => {
    const data = JSON.stringify(Array(500).fill('value'));
    jest.mocked(dump).mockImplementationOnce(() => {
      throw new RangeError('native limit');
    });
    expect(getYamlCollectionPreview(data, 10, 300)).toEqual({ text: formatJsonStringAsYaml(data), isComplete: true });
    jest.mocked(dump).mockImplementationOnce(() => {
      throw new Error('unexpected');
    });
    expect(() => getYamlCollectionPreview(data, 10, 300)).toThrow('unexpected');
  });
});

describe('YAML search precheck', () => {
  it.each([
    {
      messages: ['café 日本語 🙂', 'multi\nline', '  leading\nspace', 'bell\x07\t\x1b\x85\xa0'],
      tiny: 1e-7,
      huge: 1e21,
      zero: -0,
    },
    {
      'key\nname': 'value',
      separators: 'one\u2028two\u2029\ufeff',
      surrogate: '\ud800',
      punctuation: ["it's", 'a"b', 'back\\slash', {}, []],
    },
    { null: null, bool: false, list: [1, true, { nested: 'tail' }] },
  ])('never rejects any substring of the real dump of %#', (value) => {
    const data = JSON.stringify(value);
    const yaml = formatJsonStringAsYaml(data).toLowerCase();
    for (let length = 1; length <= 8; length++) {
      for (let start = 0; start + length <= yaml.length; start++) {
        const filter = yaml.slice(start, start + length);
        expect({ filter, possible: yamlJsonMayContain(data, filter) }).toEqual({ filter, possible: true });
      }
    }
    expect(yamlJsonMayContain(data, 'missingpayload')).toBe(false);
  });

  it('keeps uncertain syntax, malformed JSON and nonfinite numeric values searchable', () => {
    expect(yamlJsonMayContain('{invalid', 'missing')).toBe(true);
    expect(yamlJsonMayContain('1e400', 'inf')).toBe(true);
    expect(yamlJsonMayContain('{}', ': ')).toBe(true);
    expect(yamlJsonMayContain('{}', '')).toBe(false);
  });
});
