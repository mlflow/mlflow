import { describe, it, expect } from '@jest/globals';
import { resolveInputValue, resolveInputValueForJson, substituteTokens } from '../../installInstructions';

describe('substituteTokens', () => {
  it('replaces known tokens', () => {
    expect(substituteTokens('http://localhost:{port}/mcp', { port: { value: '3000' } })).toBe(
      'http://localhost:3000/mcp',
    );
  });

  it('leaves unknown tokens as-is', () => {
    expect(substituteTokens('http://localhost:{port}/mcp', {})).toBe('http://localhost:{port}/mcp');
  });

  it('handles recursive token substitution', () => {
    const vars = {
      url: { value: '{host}:{port}', variables: { host: { value: 'localhost' }, port: { value: '3000' } } },
    };
    expect(substituteTokens('{url}', vars)).toBe('localhost:3000');
  });

  it('respects max depth to prevent infinite recursion', () => {
    const vars = { a: { value: '{a}' } };
    const result = substituteTokens('{a}', vars, 3);
    expect(result).toBe('{a}');
  });
});

describe('resolveInputValue', () => {
  it('returns fixed value when set', () => {
    expect(resolveInputValue({ value: 'hello' })).toBe('hello');
  });

  it('substitutes tokens in fixed value', () => {
    expect(
      resolveInputValue({
        value: '{host}:{port}',
        variables: { host: { value: 'localhost' }, port: { value: '8080' } },
      }),
    ).toBe('localhost:8080');
  });

  it('returns default when no value', () => {
    expect(resolveInputValue({ default: '3000' })).toBe('3000');
  });

  it('returns choices when no value or default', () => {
    expect(resolveInputValue({ choices: ['a', 'b', 'c'] })).toBe('<a|b|c>');
  });

  it('returns placeholder when nothing else is set', () => {
    expect(resolveInputValue({ placeholder: 'enter-value' })).toBe('enter-value');
  });

  it('derives placeholder from valueHint', () => {
    expect(resolveInputValue({ valueHint: 'API_KEY' })).toBe('<api_key>');
  });

  it('returns <value> when nothing is set', () => {
    expect(resolveInputValue({})).toBe('<value>');
  });

  it('masks secrets regardless of other fields', () => {
    expect(resolveInputValue({ isSecret: true, value: 'actual-key', valueHint: 'API_KEY' })).toBe('<api_key>');
  });

  it('uses filepath format for placeholder', () => {
    expect(resolveInputValue({ format: 'filepath', valueHint: 'config' })).toBe('/path/to/config');
  });

  it('uses boolean format for placeholder', () => {
    expect(resolveInputValue({ format: 'boolean' })).toBe('true');
  });

  it('uses number format for placeholder', () => {
    expect(resolveInputValue({ format: 'number' })).toBe('0');
  });
});

/* eslint-disable no-template-curly-in-string */
describe('resolveInputValueForJson', () => {
  it('returns ${NAME} for secrets', () => {
    expect(resolveInputValueForJson({ name: 'API_KEY', isSecret: true })).toBe('${API_KEY}');
  });

  it('normalizes secret name to uppercase with underscores', () => {
    expect(resolveInputValueForJson({ name: 'my-api-key', isSecret: true })).toBe('${MY_API_KEY}');
  });

  it('uses valueHint for secret name when no name', () => {
    expect(resolveInputValueForJson({ valueHint: 'Token', isSecret: true })).toBe('${TOKEN}');
  });

  it('returns fixed value for non-secrets', () => {
    expect(resolveInputValueForJson({ name: 'HOST', value: 'localhost' })).toBe('localhost');
  });

  it('returns default for non-secrets', () => {
    expect(resolveInputValueForJson({ name: 'PORT', default: '3000' })).toBe('3000');
  });
});
