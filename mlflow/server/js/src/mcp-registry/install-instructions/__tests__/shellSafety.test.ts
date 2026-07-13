import { describe, it, expect } from '@jest/globals';
import { hasShellMetacharacters, shellQuote } from '../../installInstructions';

describe('hasShellMetacharacters', () => {
  it('returns false for safe values', () => {
    expect(hasShellMetacharacters('hello')).toBe(false);
    expect(hasShellMetacharacters('my-package@1.0.0')).toBe(false);
    expect(hasShellMetacharacters('/usr/local/bin/node')).toBe(false);
    expect(hasShellMetacharacters('key=value')).toBe(false);
  });

  it('detects semicolons', () => {
    expect(hasShellMetacharacters('cmd; rm -rf /')).toBe(true);
  });

  it('detects pipes', () => {
    expect(hasShellMetacharacters('cmd | cat')).toBe(true);
  });

  it('detects ampersands', () => {
    expect(hasShellMetacharacters('cmd && evil')).toBe(true);
  });

  it('detects backticks', () => {
    expect(hasShellMetacharacters('`whoami`')).toBe(true);
  });

  it('detects dollar signs', () => {
    expect(hasShellMetacharacters('$(whoami)')).toBe(true);
    expect(hasShellMetacharacters('$HOME')).toBe(true);
  });

  it('detects parentheses', () => {
    expect(hasShellMetacharacters('(subshell)')).toBe(true);
  });

  it('detects newlines', () => {
    expect(hasShellMetacharacters('line1\nline2')).toBe(true);
  });
});

describe('shellQuote', () => {
  it('returns empty quotes for empty string', () => {
    expect(shellQuote('')).toBe("''");
  });

  it('passes through safe values unquoted', () => {
    expect(shellQuote('hello')).toBe('hello');
    expect(shellQuote('my-pkg@1.0.0')).toBe('my-pkg@1.0.0');
    expect(shellQuote('/usr/bin/node')).toBe('/usr/bin/node');
  });

  it('quotes values with spaces', () => {
    expect(shellQuote('hello world')).toBe("'hello world'");
  });

  it('escapes single quotes inside', () => {
    expect(shellQuote("it's")).toBe("'it'\\''s'");
  });

  it('quotes values with shell metacharacters', () => {
    expect(shellQuote('a;b')).toBe("'a;b'");
    expect(shellQuote('a|b')).toBe("'a|b'");
  });
});
