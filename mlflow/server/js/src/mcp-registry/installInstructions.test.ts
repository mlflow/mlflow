import { describe, it, expect } from '@jest/globals';
import {
  hasShellMetacharacters,
  shellQuote,
  formatMcpJsonSnippet,
  deriveClientName,
  substituteTokens,
  resolveInputValue,
  resolveInputValueForJson,
  resolveRunner,
  buildPackageSpec,
  getRegistryFlag,
  buildRemoteInstruction,
  buildPackageInstruction,
  generateInstallInstructions,
} from './installInstructions';
import { TransportType } from './types';
import type { ServerJSONPackage, ServerJSONPayload } from './types';

// ---------------------------------------------------------------------------
// Shell safety
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Format MCP JSON
// ---------------------------------------------------------------------------

describe('formatMcpJsonSnippet', () => {
  it('wraps config in mcpServers with server name', () => {
    const result = formatMcpJsonSnippet('my-server', { url: 'https://example.com' });
    const parsed = JSON.parse(result);
    expect(parsed.mcpServers['my-server']).toEqual({ url: 'https://example.com' });
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

// ---------------------------------------------------------------------------
// Resolve server name
// ---------------------------------------------------------------------------

describe('deriveClientName', () => {
  it('uses full registry name for uniqueness', () => {
    expect(deriveClientName('com.acme/full-mcp-server')).toBe('com-acme-full-mcp-server');
  });

  it('distinguishes different namespaces with same slug', () => {
    expect(deriveClientName('io.github.foo/server')).not.toBe(deriveClientName('com.foo/server'));
    expect(deriveClientName('io.github.foo/server')).toBe('io-github-foo-server');
    expect(deriveClientName('com.foo/server')).toBe('com-foo-server');
  });

  it('handles name without namespace', () => {
    expect(deriveClientName('my-server')).toBe('my-server');
  });

  it('replaces dots and special chars with dashes', () => {
    expect(deriveClientName('com.example/my.special@server')).toBe('com-example-my-special-server');
  });

  it('collapses consecutive dashes', () => {
    expect(deriveClientName('com.test/a--b')).toBe('com-test-a-b');
  });

  it('trims leading and trailing dashes', () => {
    expect(deriveClientName('com.test/-server-')).toBe('com-test-server');
  });

  it('lowercases everything', () => {
    expect(deriveClientName('com.Acme/MyServer')).toBe('com-acme-myserver');
  });

  it('handles single-segment namespace', () => {
    expect(deriveClientName('acme/server')).toBe('acme-server');
  });

  it('handles empty slug after slash', () => {
    expect(deriveClientName('com.acme/')).toBe('com-acme');
  });

  it('falls back when input collapses to empty', () => {
    expect(deriveClientName('/')).toBe('mcp-server');
  });
});

// ---------------------------------------------------------------------------
// Resolve input
// ---------------------------------------------------------------------------

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
/* eslint-enable no-template-curly-in-string */

// ---------------------------------------------------------------------------
// Runners
// ---------------------------------------------------------------------------

describe('resolveRunner', () => {
  it('uses runtimeHint when it is a known runner', () => {
    expect(resolveRunner('uvx', 'npm')).toEqual({ runner: 'uvx' });
  });

  it('defaults npm to npx', () => {
    expect(resolveRunner(undefined, 'npm')).toEqual({ runner: 'npx' });
  });

  it('defaults pypi to uvx', () => {
    expect(resolveRunner(undefined, 'pypi')).toEqual({ runner: 'uvx' });
  });

  it('defaults pip to uvx', () => {
    expect(resolveRunner(undefined, 'pip')).toEqual({ runner: 'uvx' });
  });

  it('defaults oci to docker', () => {
    expect(resolveRunner(undefined, 'oci')).toEqual({ runner: 'docker' });
  });

  it('defaults nuget to dnx', () => {
    expect(resolveRunner(undefined, 'nuget')).toEqual({ runner: 'dnx' });
  });

  it('returns null runner for unknown registry with no hint', () => {
    expect(resolveRunner(undefined, 'unknown')).toEqual({ runner: null });
  });

  it('surfaces unrecognized runtimeHint as prerequisite note', () => {
    const result = resolveRunner('bun', 'npm');
    expect(result.runner).toBe('npx');
    expect(result.prerequisiteNote).toContain('bun');
  });

  it('returns null with note for unrecognized hint and unknown registry', () => {
    const result = resolveRunner('custom-runner', 'unknown');
    expect(result.runner).toBeNull();
    expect(result.prerequisiteNote).toContain('custom-runner');
  });
});

describe('buildPackageSpec', () => {
  it('pins npm packages with @', () => {
    expect(buildPackageSpec('@acme/server', '1.2.3', 'npm')).toBe('@acme/server@1.2.3');
  });

  it('pins pypi packages with ==', () => {
    expect(buildPackageSpec('my-server', '2.0.0', 'pypi')).toBe('my-server==2.0.0');
  });

  it('uses identifier verbatim for OCI with tag', () => {
    expect(buildPackageSpec('ghcr.io/org/server:v1.0', '1.0', 'oci')).toBe('ghcr.io/org/server:v1.0');
  });

  it('appends version tag for OCI without tag', () => {
    expect(buildPackageSpec('ghcr.io/org/server', '1.0', 'oci')).toBe('ghcr.io/org/server:1.0');
  });

  it('returns identifier as-is when no version', () => {
    expect(buildPackageSpec('@acme/server', undefined, 'npm')).toBe('@acme/server');
  });

  it('handles nuget with @ version', () => {
    expect(buildPackageSpec('MyPackage', '0.4.0', 'nuget')).toBe('MyPackage@0.4.0');
  });
});

describe('getRegistryFlag', () => {
  it('returns empty for no registryBaseUrl', () => {
    expect(getRegistryFlag('npm', undefined, 'npx')).toEqual([]);
  });

  it('returns --registry for npx with custom URL', () => {
    expect(getRegistryFlag('npm', 'https://custom.registry.com', 'npx')).toEqual([
      '--registry=https://custom.registry.com',
    ]);
  });

  it('returns --index-url for uvx with custom URL', () => {
    expect(getRegistryFlag('pypi', 'https://custom.pypi.com', 'uvx')).toEqual(['--index-url=https://custom.pypi.com']);
  });

  it('skips canonical npm registry', () => {
    expect(getRegistryFlag('npm', 'https://registry.npmjs.org', 'npx')).toEqual([]);
  });

  it('returns empty for OCI regardless of URL', () => {
    expect(getRegistryFlag('oci', 'https://custom.registry.com', 'docker')).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Build remote instruction
// ---------------------------------------------------------------------------

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
    const block = buildRemoteInstruction({ type: TransportType.SSE, url: 'https://example.com/sse' }, 'srv');
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
        headers: [{ name: 'X-Optional', isRequired: false, description: 'Optional header' }],
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
    const block = buildRemoteInstruction({ type: TransportType.SSE, url: 'https://example.com/sse' }, 'srv');
    expect(block.notes.some((n) => n.includes('OAuth'))).toBe(false);
    expect(block.notes.some((n) => n.includes('Legacy'))).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Build package instruction
// ---------------------------------------------------------------------------

function makePkg(overrides: Partial<ServerJSONPackage> = {}): ServerJSONPackage {
  return {
    registryType: 'npm',
    identifier: '@acme/mcp-server',
    transport: { type: TransportType.STDIO },
    ...overrides,
  };
}

describe('buildPackageInstruction', () => {
  describe('stdio packages', () => {
    it('builds a basic npm package command', () => {
      const block = buildPackageInstruction(makePkg(), 'acme-server');
      expect(block.kind).toBe('package');
      expect(block.claudeCodeCommand).toContain('claude mcp add');
      expect(block.claudeCodeCommand).toContain('--transport stdio');
      expect(block.claudeCodeCommand).toContain('npx');
      expect(block.claudeCodeCommand).toContain('-y');
      expect(block.claudeCodeCommand).toContain('@acme/mcp-server');
      expect(block.claudeCodeCommand).toContain('acme-server');
    });

    it('pins npm version with @', () => {
      const block = buildPackageInstruction(makePkg({ version: '2.1.0' }), 'srv');
      expect(block.claudeCodeCommand).toContain('@acme/mcp-server@2.1.0');
    });

    it('uses uvx for pypi packages', () => {
      const block = buildPackageInstruction(makePkg({ registryType: 'pypi', identifier: 'my-server' }), 'srv');
      expect(block.claudeCodeCommand).toContain('uvx');
      expect(block.claudeCodeCommand).not.toContain('npx');
    });

    it('pins pypi version with ==', () => {
      const block = buildPackageInstruction(
        makePkg({ registryType: 'pypi', identifier: 'my-server', version: '1.0.0' }),
        'srv',
      );
      expect(block.claudeCodeCommand).toContain('my-server==1.0.0');
    });

    it('uses docker for OCI packages', () => {
      const block = buildPackageInstruction(
        makePkg({ registryType: 'oci', identifier: 'ghcr.io/org/server:v1' }),
        'srv',
      );
      expect(block.claudeCodeCommand).toContain('docker');
      expect(block.claudeCodeCommand).toContain('run');
      expect(block.claudeCodeCommand).toContain('-i');
      expect(block.claudeCodeCommand).toContain('--rm');
    });

    it('respects runtimeHint override', () => {
      const block = buildPackageInstruction(makePkg({ runtimeHint: 'bunx' }), 'srv');
      expect(block.claudeCodeCommand).toContain('bunx');
      expect(block.claudeCodeCommand).not.toContain('npx');
    });

    it('includes env vars as --env flags', () => {
      const block = buildPackageInstruction(
        makePkg({
          environmentVariables: [{ name: 'API_KEY', isRequired: true, isSecret: true }],
        }),
        'srv',
      );
      expect(block.claudeCodeCommand).toContain('--env');
      expect(block.claudeCodeCommand).toContain('API_KEY');
    });

    it('includes env vars in mcpJson env map', () => {
      const block = buildPackageInstruction(
        makePkg({
          environmentVariables: [{ name: 'API_KEY', isRequired: true, isSecret: true }],
        }),
        'srv',
      );
      const env = block.mcpJsonConfig!['env'] as Record<string, string>;
      expect(env['API_KEY']).toContain('${');
    });

    it('handles non-default npm registry', () => {
      const block = buildPackageInstruction(makePkg({ registryBaseUrl: 'https://custom.npm.com' }), 'srv');
      expect(block.mcpJsonConfig!['args']).toContain('--registry=https://custom.npm.com');
    });

    it('adds -- separator for dnx runner', () => {
      const block = buildPackageInstruction(
        makePkg({ registryType: 'nuget', identifier: 'MyPkg', runtimeHint: 'dnx', version: '0.4.0' }),
        'srv',
      );
      expect(block.claudeCodeCommand).toContain('--');
    });

    it('lists optional env vars in optionalSettings', () => {
      const block = buildPackageInstruction(
        makePkg({
          environmentVariables: [{ name: 'OPTIONAL_VAR', isRequired: false, description: 'Optional' }],
        }),
        'srv',
      );
      expect(block.optionalSettings).toBeDefined();
      expect(block.optionalSettings!.some((s) => s.flag.includes('OPTIONAL_VAR'))).toBe(true);
    });
  });

  describe('network transport packages', () => {
    it('generates two-step instructions for streamable-http', () => {
      const block = buildPackageInstruction(
        makePkg({
          transport: { type: TransportType.STREAMABLE_HTTP, url: 'http://localhost:3000/mcp' },
        }),
        'srv',
      );
      expect(block.claudeCodeCommand).toContain('Step 1');
      expect(block.claudeCodeCommand).toContain('Step 2');
      expect(block.claudeCodeCommand).toContain('claude mcp add');
      expect(block.claudeCodeCommand).toContain('--transport http');
      expect(block.notes.some((n) => n.includes('step 1'))).toBe(true);
    });

    it('generates two-step instructions for sse', () => {
      const block = buildPackageInstruction(
        makePkg({
          transport: { type: TransportType.SSE, url: 'http://localhost:3000/sse' },
        }),
        'srv',
      );
      expect(block.claudeCodeCommand).toContain('Step 1');
      expect(block.claudeCodeCommand).toContain('--transport sse');
      expect(block.mcpJsonConfig!['type']).toBe('sse');
    });
  });

  describe('mcpb fallback', () => {
    it('returns fallback for mcpb packages', () => {
      const block = buildPackageInstruction(makePkg({ registryType: 'mcpb', fileSha256: 'abc123' }), 'srv');
      expect(block.kind).toBe('fallback');
      expect(block.claudeCodeCommand).toBeNull();
      expect(block.mcpJsonConfig).toBeNull();
      expect(block.notes.some((n) => n.includes('shasum'))).toBe(true);
    });

    it('warns when mcpb has no fileSha256', () => {
      const block = buildPackageInstruction(makePkg({ registryType: 'mcpb' }), 'srv');
      expect(block.notes.some((n) => n.includes('No fileSha256'))).toBe(true);
    });
  });

  describe('unknown registry fallback', () => {
    it('returns fallback for unknown registry type', () => {
      const block = buildPackageInstruction(makePkg({ registryType: 'unknown-registry' }), 'srv');
      expect(block.kind).toBe('fallback');
      expect(block.claudeCodeCommand).toBeNull();
      expect(block.fallbackReason).toContain('custom registry (unknown-registry)');
    });

    it('resolves fallback URL from registryBaseUrl', () => {
      const block = buildPackageInstruction(
        makePkg({ registryType: 'mcpx', registryBaseUrl: 'https://registry.acme.example.com' }),
        'srv',
      );
      expect(block.fallbackUrl).toBe('https://registry.acme.example.com');
      expect(block.fallbackReason).toContain('documentation');
    });

    it('resolves fallback URL from fallbackContext when package has no URLs', () => {
      const block = buildPackageInstruction(makePkg({ registryType: 'mcpx' }), 'srv', {
        websiteUrl: 'https://acme.example.com',
        repositoryUrl: 'https://github.com/acme/server',
      });
      expect(block.fallbackUrl).toBe('https://acme.example.com');
    });

    it('falls back to repositoryUrl when no other URLs available', () => {
      const block = buildPackageInstruction(makePkg({ registryType: 'mcpx' }), 'srv', {
        repositoryUrl: 'https://github.com/acme/server',
      });
      expect(block.fallbackUrl).toBe('https://github.com/acme/server');
    });

    it('shows contact message when no fallback URL available', () => {
      const block = buildPackageInstruction(makePkg({ registryType: 'mcpx' }), 'srv');
      expect(block.fallbackUrl).toBeUndefined();
      expect(block.fallbackReason).toContain('Contact the publisher');
    });
  });

  describe('arguments', () => {
    it('includes required named runtimeArguments in CLI command', () => {
      const block = buildPackageInstruction(
        makePkg({
          runtimeArguments: [{ name: '--port', value: '3000', isRequired: true }],
        }),
        'srv',
      );
      expect(block.claudeCodeCommand).toContain('--port');
      expect(block.claudeCodeCommand).toContain('3000');
    });

    it('includes required packageArguments after package spec', () => {
      const block = buildPackageInstruction(
        makePkg({
          packageArguments: [{ name: '--config', value: 'prod.json', isRequired: true }],
        }),
        'srv',
      );
      expect(block.claudeCodeCommand).toContain('--config');
      expect(block.claudeCodeCommand).toContain('prod.json');
    });

    it('omits CLI when named argument value has shell metacharacters', () => {
      const block = buildPackageInstruction(
        makePkg({
          runtimeArguments: [{ name: '--cmd', value: 'test; rm -rf /', isRequired: true }],
        }),
        'srv',
      );
      expect(block.claudeCodeCommand).toBeNull();
      expect(block.mcpJsonConfig).not.toBeNull();
    });

    it('omits CLI when package identifier has shell metacharacters', () => {
      const block = buildPackageInstruction(makePkg({ identifier: '@acme/pkg;rm -rf /' }), 'srv');
      expect(block.claudeCodeCommand).toBeNull();
      expect(block.mcpJsonConfig).not.toBeNull();
    });

    it('omits CLI when package version has shell metacharacters', () => {
      const block = buildPackageInstruction(makePkg({ version: '1.0.0;evil' }), 'srv');
      expect(block.claudeCodeCommand).toBeNull();
      expect(block.mcpJsonConfig).not.toBeNull();
    });

    it('omits CLI when registryBaseUrl has shell metacharacters', () => {
      const block = buildPackageInstruction(makePkg({ registryBaseUrl: 'https://evil.example.com;rm -rf /' }), 'srv');
      expect(block.claudeCodeCommand).toBeNull();
      expect(block.mcpJsonConfig).not.toBeNull();
    });

    it('includes positional arguments in JSON config', () => {
      const block = buildPackageInstruction(
        makePkg({
          runtimeArguments: [{ value: '8080', isRequired: true }],
        }),
        'srv',
      );
      expect(block.mcpJsonConfig).not.toBeNull();
      expect((block.mcpJsonConfig!['args'] as string[]).includes('8080')).toBe(true);
    });
  });

  describe('shell safety', () => {
    it('omits CLI command when fixed env value has metacharacters', () => {
      const block = buildPackageInstruction(
        makePkg({
          environmentVariables: [{ name: 'CMD', isRequired: true, value: 'test; rm -rf /' }],
        }),
        'srv',
      );
      expect(block.claudeCodeCommand).toBeNull();
      expect(block.mcpJsonConfig).not.toBeNull();
      expect(block.notes.some((n) => n.includes('unsafe'))).toBe(true);
    });

    it('omits CLI when streamable-http package identifier is unsafe', () => {
      const block = buildPackageInstruction(
        makePkg({
          identifier: 'pkg;evil',
          transport: { type: TransportType.STREAMABLE_HTTP, url: 'http://localhost:3000' },
        }),
        'srv',
      );
      expect(block.claudeCodeCommand).toBeNull();
      expect(block.mcpJsonConfig).not.toBeNull();
    });

    it('omits CLI when streamable-http fixed env value is unsafe', () => {
      const block = buildPackageInstruction(
        makePkg({
          transport: { type: TransportType.STREAMABLE_HTTP, url: 'http://localhost:3000' },
          environmentVariables: [{ name: 'TOKEN', value: 'abc;rm -rf /', isRequired: true }],
        }),
        'srv',
      );
      expect(block.claudeCodeCommand).toBeNull();
      expect(block.mcpJsonConfig).not.toBeNull();
    });
  });
});

// ---------------------------------------------------------------------------
// Generate install instructions (orchestrator)
// ---------------------------------------------------------------------------

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
    expect(result.serverName).toBe('com-acme-test-server');
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
        packages: [{ registryType: 'npm', identifier: '@acme/server', transport: { type: TransportType.STDIO } }],
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
          {
            registryType: 'npm',
            identifier: 'net-pkg',
            transport: { type: TransportType.STREAMABLE_HTTP, url: 'http://localhost:3000' },
          },
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
    expect(result.blocks[0].fallbackUrl).toBe('ws://example.com');
  });

  it('resolves fallback URL from websiteUrl for unknown remote transport', () => {
    const result = generateInstallInstructions(
      makePayload({
        remotes: [{ type: 'websocket' as any }],
        websiteUrl: 'https://acme.example.com',
      }),
      'com.acme/server',
    );

    expect(result.blocks[0].fallbackUrl).toBe('https://acme.example.com');
    expect(result.blocks[0].fallbackReason).toContain('documentation');
  });

  it('threads fallbackContext to package instructions', () => {
    const result = generateInstallInstructions(
      makePayload({
        packages: [
          { registryType: 'mcpx', identifier: 'example/kitchen-sink', transport: { type: TransportType.STDIO } },
        ],
        websiteUrl: 'https://acme.example.com',
        repository: { url: 'https://github.com/acme/server' },
      }),
      'com.acme/server',
    );

    expect(result.blocks[0].kind).toBe('fallback');
    expect(result.blocks[0].fallbackUrl).toBe('https://acme.example.com');
    expect(result.blocks[0].fallbackReason).toContain('custom registry (mcpx)');
    expect(result.blocks[0].fallbackReason).not.toContain('No known runner');
  });

  it('handles a full payload with all entry types', () => {
    const result = generateInstallInstructions(
      makePayload({
        remotes: [
          { type: TransportType.STREAMABLE_HTTP, url: 'https://api.example.com/mcp' },
          { type: TransportType.SSE, url: 'https://api.example.com/sse' },
        ],
        packages: [
          {
            registryType: 'npm',
            identifier: '@acme/mcp-server',
            version: '1.2.3',
            transport: { type: TransportType.STDIO },
          },
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
