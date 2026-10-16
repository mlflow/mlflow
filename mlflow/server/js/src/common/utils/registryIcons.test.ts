import { describe, it, expect } from '@jest/globals';
import { resolveIcon, sanitizeHref } from './registryIcons';

describe('resolveIcon', () => {
  const light = { src: 'https://example.com/light.svg', theme: 'light' };
  const dark = { src: 'https://example.com/dark.svg', theme: 'dark' };
  const any = { src: 'https://example.com/any.svg' };

  it('returns undefined for missing or empty icons', () => {
    expect(resolveIcon(undefined, false)).toBeUndefined();
    expect(resolveIcon(null, false)).toBeUndefined();
    expect(resolveIcon([], false)).toBeUndefined();
  });

  it('prefers the matching theme and otherwise an unthemed icon', () => {
    expect(resolveIcon([light, dark], false)?.src).toBe(light.src);
    expect(resolveIcon([light, dark], true)?.src).toBe(dark.src);
    expect(resolveIcon([any], true)?.src).toBe(any.src);
    expect(resolveIcon([light], true)).toBeUndefined();
  });
});

describe('sanitizeHref', () => {
  it('allows http and https URLs', () => {
    expect(sanitizeHref('https://example.com')).toBe('https://example.com');
    expect(sanitizeHref('http://localhost:5000')).toBe('http://localhost:5000');
  });

  it('rejects non-http URLs and malformed values', () => {
    expect(sanitizeHref(`${'javascript'}:alert(1)`)).toBeUndefined();
    expect(sanitizeHref('data:text/html,<script>alert(1)</script>')).toBeUndefined();
    expect(sanitizeHref('ftp://example.com')).toBeUndefined();
    expect(sanitizeHref(undefined)).toBeUndefined();
    expect(sanitizeHref('not a url')).toBeUndefined();
  });
});
