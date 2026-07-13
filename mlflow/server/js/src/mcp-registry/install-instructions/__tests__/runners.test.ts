import { describe, it, expect } from '@jest/globals';
import { resolveRunner, buildPackageSpec, getRegistryFlag } from '../../installInstructions';

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
    expect(getRegistryFlag('pypi', 'https://custom.pypi.com', 'uvx')).toEqual([
      '--index-url=https://custom.pypi.com',
    ]);
  });

  it('skips canonical npm registry', () => {
    expect(getRegistryFlag('npm', 'https://registry.npmjs.org', 'npx')).toEqual([]);
  });

  it('returns empty for OCI regardless of URL', () => {
    expect(getRegistryFlag('oci', 'https://custom.registry.com', 'docker')).toEqual([]);
  });
});
