import { describe, test, expect } from '@jest/globals';
import { getLanguage, TEXT_EXTENSIONS } from './FileUtils';

describe('FileUtils', () => {
  test('getLanguage', () => {
    expect(getLanguage('foo.js')).toBe('js');
    expect(getLanguage('foo.bar.js')).toBe('js');
    expect(getLanguage('foo/bar.js')).toBe('js');
    expect(getLanguage('foo')).toBe('foo');
    expect(getLanguage('MLmodel')).toBe('yaml');
    expect(getLanguage('MLproject')).toBe('yaml');
    expect(getLanguage('events.jsonl')).toBe('json');
  });

  test('supports jsonl text previews', () => {
    expect(TEXT_EXTENSIONS.has('jsonl')).toBe(true);
  });
});
