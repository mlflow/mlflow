import { describe, it, expect } from '@jest/globals';
import {
  resolveDisplayName,
  STATUS_TAG_COLOR,
  STATUS_TRANSITIONS,
  validateServerJson,
  validateToolsJson,
} from './utils';

describe('resolveDisplayName', () => {
  it('returns display_name when set', () => {
    expect(resolveDisplayName({ display_name: 'My Server', name: 'io.test/server' })).toBe('My Server');
  });

  it('falls back to name when display_name is undefined', () => {
    expect(resolveDisplayName({ name: 'io.test/server' })).toBe('io.test/server');
  });

  it('falls back to name when display_name is empty string', () => {
    expect(resolveDisplayName({ display_name: '', name: 'io.test/server' })).toBe('io.test/server');
  });
});

describe('STATUS_TAG_COLOR', () => {
  it('maps all statuses', () => {
    expect(STATUS_TAG_COLOR.draft).toBe('charcoal');
    expect(STATUS_TAG_COLOR.active).toBe('lime');
    expect(STATUS_TAG_COLOR.deprecated).toBe('lemon');
    expect(STATUS_TAG_COLOR.deleted).toBe('coral');
  });
});

describe('STATUS_TRANSITIONS', () => {
  it('draft can transition to active', () => {
    expect(STATUS_TRANSITIONS.draft).toEqual(['active']);
  });

  it('active can transition to draft and deprecated', () => {
    expect(STATUS_TRANSITIONS.active).toEqual(['draft', 'deprecated']);
  });

  it('deprecated can transition to active', () => {
    expect(STATUS_TRANSITIONS.deprecated).toEqual(['active']);
  });

  it('deleted has no transitions', () => {
    expect(STATUS_TRANSITIONS.deleted).toEqual([]);
  });
});

describe('validateServerJson', () => {
  it('returns error for empty input', () => {
    const result = validateServerJson('');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Server definition is required');
  });

  it('returns error for whitespace-only input', () => {
    const result = validateServerJson('   ');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Server definition is required');
  });

  it('returns error for invalid JSON', () => {
    const result = validateServerJson('{invalid json');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Invalid JSON format in server configuration');
  });

  it('returns error for JSON array', () => {
    const result = validateServerJson('[1, 2, 3]');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Server configuration must be a JSON object');
  });

  it('returns error for JSON string', () => {
    const result = validateServerJson('"hello"');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Server configuration must be a JSON object');
  });

  it('returns error for missing name field', () => {
    const result = validateServerJson('{"version": "1.0.0"}');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Server configuration must include a "name" field');
  });

  it('returns error for missing version field', () => {
    const result = validateServerJson('{"name": "test-server"}');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Server configuration must include a "version" field');
  });

  it('returns error for non-string name', () => {
    const result = validateServerJson('{"name": 123, "version": "1.0.0"}');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Server configuration must include a "name" field');
  });

  it('returns error for empty string name', () => {
    const result = validateServerJson('{"name": "", "version": "1.0.0"}');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Server configuration must include a "name" field');
  });

  it('returns error for non-string version', () => {
    const result = validateServerJson('{"name": "test", "version": 123}');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Server configuration must include a "version" field');
  });

  it('returns error for empty string version', () => {
    const result = validateServerJson('{"name": "test", "version": ""}');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Server configuration must include a "version" field');
  });

  it('returns error for null name', () => {
    const result = validateServerJson('{"name": null, "version": "1.0.0"}');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Server configuration must include a "name" field');
  });

  it('returns error for null version', () => {
    const result = validateServerJson('{"name": "test", "version": null}');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Server configuration must include a "version" field');
  });

  it('returns error for JSON number as top-level value', () => {
    const result = validateServerJson('42');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Server configuration must be a JSON object');
  });

  it('passes for valid server.json with name and version', () => {
    const result = validateServerJson('{"name": "io.github.test/server", "version": "1.0.0"}');
    expect(result.valid).toBe(true);
    expect(result.parsed).toEqual({ name: 'io.github.test/server', version: '1.0.0' });
  });

  it('preserves extra fields in parsed output', () => {
    const input = '{"name": "test", "version": "1.0.0", "title": "My Server", "packages": []}';
    const result = validateServerJson(input);
    expect(result.valid).toBe(true);
    expect(result.parsed?.title).toBe('My Server');
    expect(result.parsed?.packages).toEqual([]);
  });
});

describe('validateToolsJson', () => {
  it('passes for empty input', () => {
    const result = validateToolsJson('');
    expect(result.valid).toBe(true);
  });

  it('returns error for invalid JSON', () => {
    const result = validateToolsJson('{not json}');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Invalid JSON format in tools configuration');
  });

  it('returns error for non-array JSON', () => {
    const result = validateToolsJson('{"name": "search"}');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Tools must be a JSON array');
  });

  it('returns error for tool missing name', () => {
    const result = validateToolsJson('[{"description": "no name"}]');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Tool at index 0 must have a "name" field');
  });

  it('returns error for tool that is an array', () => {
    const result = validateToolsJson('[[1, 2]]');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Tool at index 0 must be a JSON object');
  });

  it('returns error for tool that is a primitive', () => {
    const result = validateToolsJson('["not-an-object"]');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Tool at index 0 must be a JSON object');
  });

  it('passes for valid tools array', () => {
    const result = validateToolsJson('[{"name": "search", "description": "Search the web"}]');
    expect(result.valid).toBe(true);
    expect(result.parsed).toEqual([{ name: 'search', description: 'Search the web' }]);
  });
});
