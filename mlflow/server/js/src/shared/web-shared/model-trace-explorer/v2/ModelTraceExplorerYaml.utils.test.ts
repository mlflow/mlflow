import { describe, expect, it, jest } from '@jest/globals';

import { formatJsonStringAsYaml, getYamlCollectionPreview } from './ModelTraceExplorerYaml.utils';

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
});
