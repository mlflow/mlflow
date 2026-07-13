import { describe, it, expect } from '@jest/globals';
import { generateInstallInstructions } from '../../installInstructions';
import { TransportType } from '../../types';
import type { ServerJSONPayload } from '../../types';

function makePayload(overrides: Partial<ServerJSONPayload> = {}): ServerJSONPayload {
  return {
    name: 'com.acme/test-server',
    version: '1.0.0',
    ...overrides,
  };
}

describe('generateInstallInstructions', () => {
  it('derives server name from registry name', () => {
    const result = generateInstallInstructions(makePayload(), 'com.acme/test-server');
    expect(result.serverName).toBe('acme-test-server');
  });

  it('returns empty blocks for payload with no packages or remotes', () => {
    const result = generateInstallInstructions(makePayload(), 'com.acme/test-server');
    expect(result.blocks).toEqual([]);
  });

  it('orders streamable-http remotes first', () => {
    const result = generateInstallInstructions(
      makePayload({
        remotes: [
          { type: TransportType.SSE, url: 'https://example.com/sse' },
          { type: TransportType.STREAMABLE_HTTP, url: 'https://example.com/mcp' },
        ],
        packages: [
          { registryType: 'npm', identifier: '@acme/server', transport: { type: TransportType.STDIO } },
        ],
      }),
      'com.acme/server',
    );

    expect(result.blocks.length).toBe(3);
    expect(result.blocks[0].label).toContain('streamable-http');
    expect(result.blocks[1].kind).toBe('package');
    expect(result.blocks[2].label).toContain('sse');
  });

  it('orders stdio packages before network packages', () => {
    const result = generateInstallInstructions(
      makePayload({
        packages: [
          { registryType: 'npm', identifier: 'net-pkg', transport: { type: TransportType.STREAMABLE_HTTP, url: 'http://localhost:3000' } },
          { registryType: 'npm', identifier: 'stdio-pkg', transport: { type: TransportType.STDIO } },
        ],
      }),
      'com.acme/server',
    );

    expect(result.blocks[0].label).toContain('stdio-pkg');
    expect(result.blocks[1].label).toContain('net-pkg');
  });

  it('generates fallback for unknown remote transport types', () => {
    const result = generateInstallInstructions(
      makePayload({
        remotes: [{ type: 'websocket' as any, url: 'ws://example.com' }],
      }),
      'com.acme/server',
    );

    expect(result.blocks.length).toBe(1);
    expect(result.blocks[0].kind).toBe('fallback');
    expect(result.blocks[0].fallbackReason).toContain('websocket');
  });

  it('handles a full payload with all entry types', () => {
    const result = generateInstallInstructions(
      makePayload({
        remotes: [
          { type: TransportType.STREAMABLE_HTTP, url: 'https://api.example.com/mcp' },
          { type: TransportType.SSE, url: 'https://api.example.com/sse' },
        ],
        packages: [
          { registryType: 'npm', identifier: '@acme/mcp-server', version: '1.2.3', transport: { type: TransportType.STDIO } },
          { registryType: 'pypi', identifier: 'acme-mcp', version: '2.0.0', transport: { type: TransportType.STDIO } },
        ],
      }),
      'com.acme/mcp-server',
    );

    expect(result.blocks.length).toBe(4);
    expect(result.blocks[0].label).toContain('streamable-http');
    expect(result.blocks[1].label).toContain('@acme/mcp-server');
    expect(result.blocks[2].label).toContain('acme-mcp');
    expect(result.blocks[3].label).toContain('sse');
  });
});
