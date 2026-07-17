import { describe, it, expect } from '@jest/globals';
import { MCPServerAction } from './types';
import {
  sanitizeHref,
  resolveDisplayName,
  resolveBindingDisplayName,
  formatTransportType,
  isValidEndpointUrl,
  formatBindingTarget,
  tagsRecordToArray,
  STATUS_TAG_COLOR,
  STATUS_TRANSITIONS,
  validateServerJson,
  validateToolsJson,
  buildPackageConnectOptionKey,
  buildRemoteConnectOptionKey,
  isServerDimmed,
  getServerPermissions,
} from './utils';
import { MCPStatus, TransportType } from './types';
import { createMockMCPServer, createMockAccessBinding } from './test-utils';

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

describe('buildPackageConnectOptionKey', () => {
  it('joins registry type and identifier', () => {
    expect(buildPackageConnectOptionKey({ registryType: 'npm', identifier: '@acme/pkg' })).toBe('npm:@acme/pkg');
  });
});

describe('buildRemoteConnectOptionKey', () => {
  it('prefers url when present', () => {
    expect(buildRemoteConnectOptionKey({ type: 'sse', url: 'https://example.com' })).toBe('remote:https://example.com');
  });

  it('falls back to type when url is missing', () => {
    expect(buildRemoteConnectOptionKey({ type: 'sse' })).toBe('remote:sse');
  });
});

describe('sanitizeHref', () => {
  it('allows https URLs', () => {
    expect(sanitizeHref('https://example.com')).toBe('https://example.com');
  });

  it('allows http URLs', () => {
    expect(sanitizeHref('http://localhost:5000')).toBe('http://localhost:5000');
  });

  it('rejects javascript: URLs', () => {
    expect(sanitizeHref(`${'javascript'}:alert(1)`)).toBeUndefined();
  });

  it('rejects data: URLs', () => {
    expect(sanitizeHref('data:text/html,<script>alert(1)</script>')).toBeUndefined();
  });

  it('rejects ftp: URLs', () => {
    expect(sanitizeHref('ftp://example.com')).toBeUndefined();
  });

  it('returns undefined for undefined input', () => {
    expect(sanitizeHref(undefined)).toBeUndefined();
  });

  it('returns undefined for malformed URLs', () => {
    expect(sanitizeHref('not a url')).toBeUndefined();
  });
});

describe('getServerPermissions', () => {
  it('grants all actions when allowed_actions is undefined (no auth or admin)', () => {
    const perms = getServerPermissions(createMockMCPServer());
    expect(perms.canUpdate).toBe(true);
    expect(perms.canDelete).toBe(true);
    expect(perms.canManage).toBe(true);
  });

  it('denies all actions when allowed_actions is empty (READ permission)', () => {
    const perms = getServerPermissions(createMockMCPServer({ allowed_actions: [] }));
    expect(perms.canUpdate).toBe(false);
    expect(perms.canDelete).toBe(false);
    expect(perms.canManage).toBe(false);
  });

  it('grants only matching actions', () => {
    const perms = getServerPermissions(
      createMockMCPServer({ allowed_actions: [MCPServerAction.USE, MCPServerAction.UPDATE] }),
    );
    expect(perms.canUpdate).toBe(true);
    expect(perms.canDelete).toBe(false);
    expect(perms.canManage).toBe(false);
  });

  it('grants all actions for MANAGE permission', () => {
    const perms = getServerPermissions(
      createMockMCPServer({
        allowed_actions: [MCPServerAction.USE, MCPServerAction.UPDATE, MCPServerAction.DELETE, MCPServerAction.MANAGE],
      }),
    );
    expect(perms.canUpdate).toBe(true);
    expect(perms.canDelete).toBe(true);
    expect(perms.canManage).toBe(true);
  });
});

describe('isServerDimmed', () => {
  const binding = createMockAccessBinding();

  it('returns false for active server with bindings', () => {
    expect(isServerDimmed(createMockMCPServer({ status: MCPStatus.ACTIVE, access_bindings: [binding] }))).toBe(false);
  });

  it('returns true for active server without bindings', () => {
    expect(isServerDimmed(createMockMCPServer({ status: MCPStatus.ACTIVE, access_bindings: [] }))).toBe(true);
  });

  it('returns true for draft server with bindings', () => {
    expect(isServerDimmed(createMockMCPServer({ status: MCPStatus.DRAFT, access_bindings: [binding] }))).toBe(true);
  });

  it('returns true for draft server without bindings', () => {
    expect(isServerDimmed(createMockMCPServer({ status: MCPStatus.DRAFT, access_bindings: [] }))).toBe(true);
  });

  it('returns true for deprecated server with bindings', () => {
    expect(isServerDimmed(createMockMCPServer({ status: MCPStatus.DEPRECATED, access_bindings: [binding] }))).toBe(
      true,
    );
  });

  it('returns true when status is undefined (no version resolved)', () => {
    expect(isServerDimmed(createMockMCPServer({ access_bindings: [binding] }))).toBe(true);
  });
});

describe('resolveBindingDisplayName', () => {
  it('returns display_name from resolved_version when set', () => {
    expect(
      resolveBindingDisplayName({
        server_name: 'io.test/server',
        resolved_version: { display_name: 'My Display Name', server_json: { title: 'Title' } },
      }),
    ).toBe('My Display Name');
  });

  it('falls back to server_json title when display_name is missing', () => {
    expect(
      resolveBindingDisplayName({
        server_name: 'io.test/server',
        resolved_version: { server_json: { title: 'Server Title' } },
      }),
    ).toBe('Server Title');
  });

  it('falls back to server_name when resolved_version has no display info', () => {
    expect(
      resolveBindingDisplayName({
        server_name: 'io.test/server',
        resolved_version: { server_json: {} },
      }),
    ).toBe('io.test/server');
  });

  it('falls back to server_name when resolved_version is null', () => {
    expect(resolveBindingDisplayName({ server_name: 'io.test/server', resolved_version: null })).toBe('io.test/server');
  });

  it('falls back to server_name when resolved_version is undefined', () => {
    expect(resolveBindingDisplayName({ server_name: 'io.test/server' })).toBe('io.test/server');
  });
});

describe('formatTransportType', () => {
  it('returns label for streamable-http', () => {
    expect(formatTransportType(TransportType.STREAMABLE_HTTP)).toBe('streamable-http');
  });

  it('returns label for sse', () => {
    expect(formatTransportType(TransportType.SSE)).toBe('sse');
  });

  it('falls back to raw value for unknown transport', () => {
    // @ts-expect-error testing fallback for unknown transport type
    expect(formatTransportType('unknown-type')).toBe('unknown-type');
  });
});

describe('isValidEndpointUrl', () => {
  it('accepts valid https URL', () => {
    expect(isValidEndpointUrl('https://example.com/api')).toBe(true);
  });

  it('accepts valid http URL', () => {
    expect(isValidEndpointUrl('http://localhost:8080')).toBe(true);
  });

  it('accepts URL with leading/trailing whitespace', () => {
    expect(isValidEndpointUrl('  https://example.com  ')).toBe(true);
  });

  it('rejects empty string', () => {
    expect(isValidEndpointUrl('')).toBe(false);
  });

  it('rejects whitespace-only string', () => {
    expect(isValidEndpointUrl('   ')).toBe(false);
  });

  it('rejects non-http protocol', () => {
    expect(isValidEndpointUrl('ftp://example.com')).toBe(false);
  });

  it('rejects javascript: protocol', () => {
    // eslint-disable-next-line no-script-url -- testing URL validation rejects this
    expect(isValidEndpointUrl('javascript:alert(1)')).toBe(false);
  });

  it('rejects plain text', () => {
    expect(isValidEndpointUrl('not a url')).toBe(false);
  });

  it('rejects URL without hostname', () => {
    expect(isValidEndpointUrl('https://')).toBe(false);
  });
});

describe('formatBindingTarget', () => {
  it('returns alias prefixed with @ when server_alias is set', () => {
    expect(formatBindingTarget({ server_alias: 'stable', server_version: '1.0.0' })).toBe('@stable');
  });

  it('returns version when server_alias is not set', () => {
    expect(formatBindingTarget({ server_version: '1.0.0' })).toBe('1.0.0');
  });

  it('returns dash when neither alias nor version is set', () => {
    expect(formatBindingTarget({})).toBe('—');
  });

  it('returns dash when version is empty string', () => {
    expect(formatBindingTarget({ server_version: '' })).toBe('—');
  });

  it('prefers alias over version', () => {
    expect(formatBindingTarget({ server_alias: 'latest', server_version: '2.0.0' })).toBe('@latest');
  });
});

describe('tagsRecordToArray', () => {
  it('converts record to key-value array', () => {
    expect(tagsRecordToArray({ env: 'prod', team: 'ml' })).toEqual([
      { key: 'env', value: 'prod' },
      { key: 'team', value: 'ml' },
    ]);
  });

  it('returns empty array for empty record', () => {
    expect(tagsRecordToArray({})).toEqual([]);
  });

  it('returns empty array when called with no arguments', () => {
    expect(tagsRecordToArray()).toEqual([]);
  });
});
