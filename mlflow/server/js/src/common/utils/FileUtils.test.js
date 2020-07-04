import { getLanguage } from './FileUtils';

describe('FileUtils', () => {
  test('getLanguage', () => {
    expect(getLanguage('foo.py')).toBe('py');
    expect(getLanguage('foo/bar.py')).toBe('py');
    expect(getLanguage('foo')).toBe('foo');
    expect(getLanguage('MLmodel')).toBe('yaml');
    expect(getLanguage('MLproject')).toBe('yaml');
  });
});
