import { getLanguage } from './FileUtils';

describe('FileUtils', () => {
  test('getLanguage', () => {
    expect(getLanguage('foo.js')).toBe('js');
    expect(getLanguage('foo.bar.js')).toBe('js');
    expect(getLanguage('foo/bar.js')).toBe('js');
    expect(getLanguage('foo')).toBe('foo');
    expect(getLanguage('MLmodel')).toBe('yaml');
    expect(getLanguage('MLproject')).toBe('yaml');
  });
});
