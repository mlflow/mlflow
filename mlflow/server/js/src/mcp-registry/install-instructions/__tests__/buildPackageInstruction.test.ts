import { describe, it, expect } from '@jest/globals';
import { buildPackageInstruction } from '../../installInstructions';
import { TransportType } from '../../types';
import type { ServerJSONPackage } from '../../types';

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
      const block = buildPackageInstruction(
        makePkg({ registryType: 'pypi', identifier: 'my-server' }),
        'srv',
      );
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
      const block = buildPackageInstruction(
        makePkg({ runtimeHint: 'bunx' }),
        'srv',
      );
      expect(block.claudeCodeCommand).toContain('bunx');
      expect(block.claudeCodeCommand).not.toContain('npx');
    });

    it('includes env vars as --env flags', () => {
      const block = buildPackageInstruction(
        makePkg({
          environmentVariables: [
            { name: 'API_KEY', isRequired: true, isSecret: true },
          ],
        }),
        'srv',
      );
      expect(block.claudeCodeCommand).toContain('--env');
      expect(block.claudeCodeCommand).toContain('API_KEY');
    });

    it('includes env vars in mcpJson env map', () => {
      const block = buildPackageInstruction(
        makePkg({
          environmentVariables: [
            { name: 'API_KEY', isRequired: true, isSecret: true },
          ],
        }),
        'srv',
      );
      const env = block.mcpJsonConfig!['env'] as Record<string, string>;
      expect(env['API_KEY']).toContain('${');
    });

    it('handles non-default npm registry', () => {
      const block = buildPackageInstruction(
        makePkg({ registryBaseUrl: 'https://custom.npm.com' }),
        'srv',
      );
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
          environmentVariables: [
            { name: 'OPTIONAL_VAR', isRequired: false, description: 'Optional' },
          ],
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
      const block = buildPackageInstruction(
        makePkg({ registryType: 'mcpb', fileSha256: 'abc123' }),
        'srv',
      );
      expect(block.kind).toBe('fallback');
      expect(block.claudeCodeCommand).toBeNull();
      expect(block.mcpJsonConfig).toBeNull();
      expect(block.notes.some((n) => n.includes('shasum'))).toBe(true);
    });

    it('warns when mcpb has no fileSha256', () => {
      const block = buildPackageInstruction(
        makePkg({ registryType: 'mcpb' }),
        'srv',
      );
      expect(block.notes.some((n) => n.includes('No fileSha256'))).toBe(true);
    });
  });

  describe('unknown registry fallback', () => {
    it('returns fallback for unknown registry type', () => {
      const block = buildPackageInstruction(
        makePkg({ registryType: 'unknown-registry' }),
        'srv',
      );
      expect(block.claudeCodeCommand).toBeNull();
      expect(block.fallbackReason).toContain('No known runner');
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
          environmentVariables: [
            { name: 'CMD', isRequired: true, value: 'test; rm -rf /' },
          ],
        }),
        'srv',
      );
      expect(block.claudeCodeCommand).toBeNull();
      expect(block.mcpJsonConfig).not.toBeNull();
      expect(block.notes.some((n) => n.includes('unsafe'))).toBe(true);
    });
  });
});
