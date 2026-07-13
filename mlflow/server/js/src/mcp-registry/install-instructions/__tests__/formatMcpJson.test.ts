import { describe, it, expect } from '@jest/globals';
import { formatMcpJsonSnippet, getMcpJsonFooterNote } from '../../installInstructions';

describe('formatMcpJsonSnippet', () => {
  it('wraps config in mcpServers with server name', () => {
    const result = formatMcpJsonSnippet('my-server', { url: 'https://example.com' });
    const parsed = JSON.parse(result);
    expect(parsed.mcpServers['my-server']).toEqual({ url: 'https://example.com' });
  });

  it('formats with 2-space indentation', () => {
    const result = formatMcpJsonSnippet('s', { url: 'http://x' });
    expect(result).toContain('  ');
    expect(result).not.toContain('\t');
  });

  /* eslint-disable no-template-curly-in-string */
  it('handles stdio config with command and args', () => {
    const result = formatMcpJsonSnippet('my-server', {
      command: 'npx',
      args: ['-y', '@acme/server@1.0.0'],
      env: { API_KEY: '${API_KEY}' },
    });
    const parsed = JSON.parse(result);
    expect(parsed.mcpServers['my-server'].command).toBe('npx');
    expect(parsed.mcpServers['my-server'].args).toEqual(['-y', '@acme/server@1.0.0']);
    expect(parsed.mcpServers['my-server'].env.API_KEY).toBe('${API_KEY}');
  });
  /* eslint-enable no-template-curly-in-string */
});

describe('getMcpJsonFooterNote', () => {
  it('mentions .mcp.json', () => {
    expect(getMcpJsonFooterNote()).toContain('.mcp.json');
  });

  it('mentions claude mcp list', () => {
    expect(getMcpJsonFooterNote()).toContain('claude mcp list');
  });
});
