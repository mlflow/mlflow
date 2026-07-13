import { describe, it, expect } from '@jest/globals';
import { buildRemoteInstruction } from '../../installInstructions';
import { TransportType } from '../../types';

describe('buildRemoteInstruction', () => {
  it('builds a simple streamable-http remote', () => {
    const block = buildRemoteInstruction(
      { type: TransportType.STREAMABLE_HTTP, url: 'https://api.example.com/mcp' },
      'my-server',
    );
    expect(block.kind).toBe('remote');
    expect(block.claudeCodeCommand).toContain('claude mcp add');
    expect(block.claudeCodeCommand).toContain('--transport http');
    expect(block.claudeCodeCommand).toContain('my-server');
    expect(block.claudeCodeCommand).toContain('https://api.example.com/mcp');
    expect(block.mcpJsonConfig).toEqual({ url: 'https://api.example.com/mcp' });
  });

  it('maps streamable-http to http in CLI transport', () => {
    const block = buildRemoteInstruction(
      { type: TransportType.STREAMABLE_HTTP, url: 'https://example.com/mcp' },
      'srv',
    );
    expect(block.claudeCodeCommand).toContain('--transport http');
    expect(block.claudeCodeCommand).not.toContain('streamable-http');
  });

  it('keeps sse transport as-is and adds type to mcpJson', () => {
    const block = buildRemoteInstruction(
      { type: TransportType.SSE, url: 'https://example.com/sse' },
      'srv',
    );
    expect(block.claudeCodeCommand).toContain('--transport sse');
    expect(block.mcpJsonConfig).toEqual({ url: 'https://example.com/sse', type: 'sse' });
  });

  it('includes required headers in CLI and JSON', () => {
    const block = buildRemoteInstruction(
      {
        type: TransportType.STREAMABLE_HTTP,
        url: 'https://example.com/mcp',
        headers: [
          { name: 'Authorization', isRequired: true, isSecret: true },
          { name: 'X-Custom', isRequired: true, value: 'fixed-value' },
        ],
      },
      'srv',
    );
    expect(block.claudeCodeCommand).toContain('--header');
    expect(block.claudeCodeCommand).toContain('Authorization');
    expect(block.mcpJsonConfig!['headers']).toBeDefined();
    const headers = block.mcpJsonConfig!['headers'] as Record<string, string>;
    expect(headers['Authorization']).toContain('${');
    expect(headers['X-Custom']).toBe('fixed-value');
  });

  it('lists optional headers in notes and optionalSettings', () => {
    const block = buildRemoteInstruction(
      {
        type: TransportType.STREAMABLE_HTTP,
        url: 'https://example.com/mcp',
        headers: [
          { name: 'X-Optional', isRequired: false, description: 'Optional header' },
        ],
      },
      'srv',
    );
    expect(block.notes.some((n) => n.includes('Optional headers'))).toBe(true);
    expect(block.optionalSettings).toHaveLength(1);
    expect(block.optionalSettings![0].flag).toContain('X-Optional');
  });

  it('resolves URL template variables', () => {
    const block = buildRemoteInstruction(
      {
        type: TransportType.STREAMABLE_HTTP,
        url: 'https://{host}/mcp',
        variables: { host: { value: 'api.example.com' } },
      },
      'srv',
    );
    expect(block.claudeCodeCommand).toContain('https://api.example.com/mcp');
    expect(block.mcpJsonConfig!['url']).toBe('https://api.example.com/mcp');
  });

  it('returns fallback when no URL', () => {
    const block = buildRemoteInstruction({ type: TransportType.STREAMABLE_HTTP }, 'srv');
    expect(block.claudeCodeCommand).toBeNull();
    expect(block.mcpJsonConfig).toBeNull();
    expect(block.fallbackReason).toBeDefined();
  });

  it('does not add speculative notes about OAuth or SSE legacy', () => {
    const block = buildRemoteInstruction(
      { type: TransportType.SSE, url: 'https://example.com/sse' },
      'srv',
    );
    expect(block.notes.some((n) => n.includes('OAuth'))).toBe(false);
    expect(block.notes.some((n) => n.includes('Legacy'))).toBe(false);
  });
});
