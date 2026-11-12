import { describe, it, expect } from '@jest/globals';
import { containsNonAsciiCharacters, hasNonAsciiApiKey, isApiKeySecretField } from './secretUtils';

describe('containsNonAsciiCharacters', () => {
  it.each([
    ['sk-abc123', false],
    ['', false],
    ['key with spaces and !@#$%^&*()', false],
    ['가나다', true],
    ['나-abc123', true],
    ['sk-ａｂｃ', true],
    ['sk-abcé', true],
    ['sk-abc😀', true],
  ])('returns the expected result for %j', (value, expected) => {
    expect(containsNonAsciiCharacters(value)).toBe(expected);
  });
});

describe('isApiKeySecretField', () => {
  it('excludes JSON credential fields', () => {
    expect(isApiKeySecretField('api_key')).toBe(true);
    expect(isApiKeySecretField('aws_secret_access_key')).toBe(true);
    expect(isApiKeySecretField('vertex_credentials')).toBe(false);
  });
});

describe('hasNonAsciiApiKey', () => {
  it('detects non-ASCII characters in API key fields', () => {
    expect(hasNonAsciiApiKey({ api_key: '가나다' })).toBe(true);
    expect(hasNonAsciiApiKey({ aws_access_key_id: 'AKIA', aws_secret_access_key: '나-secret' })).toBe(true);
  });

  it('accepts ASCII API keys and empty input', () => {
    expect(hasNonAsciiApiKey({ api_key: 'sk-abc123' })).toBe(false);
    expect(hasNonAsciiApiKey({})).toBe(false);
    expect(hasNonAsciiApiKey(undefined)).toBe(false);
  });

  it('ignores JSON credential fields', () => {
    expect(hasNonAsciiApiKey({ vertex_credentials: '{"client_email": "서비스@example.com"}' })).toBe(false);
  });
});
