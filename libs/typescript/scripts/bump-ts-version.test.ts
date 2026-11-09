import { spawnSync } from 'child_process';
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'fs';
import { tmpdir } from 'os';
import { join, relative } from 'path';

const TS_ROOT = join(__dirname, '..');
const SCRIPT = join(__dirname, 'bump-ts-version.ts');
const MANIFEST_RELPATH = join('.claude-plugin', 'plugin.json');

// Mirrors the real manifest's shape: prettier keeps the short `keywords` array
// on one line, which a JSON.stringify round-trip would break.
const MANIFEST = `{
  "name": "mlflow-tracing",
  "version": "0.2.0",
  "description": "Observability plugin for Claude Code with MLflow tracing",
  "keywords": ["mlflow", "tracing", "observability", "llm"],
  "skills": "./skills/"
}
`;

/** Copy every package.json the script expects into a fresh scratch tree. */
function createScratchTree(): string {
  const root = mkdtempSync(join(tmpdir(), 'bump-ts-version-'));
  mkdirSync(join(root, 'core'));
  writeFileSync(
    join(root, 'core', 'package.json'),
    readFileSync(join(TS_ROOT, 'core', 'package.json')),
  );
  for (const name of readdirSync(join(TS_ROOT, 'integrations'))) {
    const packagePath = join(TS_ROOT, 'integrations', name, 'package.json');
    if (!existsSync(packagePath)) {
      continue;
    }
    mkdirSync(join(root, 'integrations', name, '.claude-plugin'), { recursive: true });
    writeFileSync(join(root, 'integrations', name, 'package.json'), readFileSync(packagePath));
  }
  writeFileSync(join(root, 'integrations', 'claude-code', MANIFEST_RELPATH), MANIFEST);
  return root;
}

function runBump(cwd: string, version: string) {
  return spawnSync(process.execPath, [require.resolve('tsx/cli'), SCRIPT, '--version', version], {
    cwd,
    encoding: 'utf-8',
  });
}

function snapshot(root: string): Record<string, string> {
  const files: Record<string, string> = {};
  for (const entry of readdirSync(root, { recursive: true, withFileTypes: true })) {
    if (entry.isFile()) {
      const path = join(entry.parentPath, entry.name);
      files[relative(root, path)] = readFileSync(path, 'utf-8');
    }
  }
  return files;
}

function readJson(path: string): Record<string, any> {
  return JSON.parse(readFileSync(path, 'utf-8')) as Record<string, any>;
}

describe('bump-ts-version', () => {
  let root: string;

  beforeEach(() => {
    root = createScratchTree();
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it('bumps the plugin manifest alongside package.json and preserves its formatting', () => {
    const result = runBump(root, '0.5.0');

    expect(result.status).toBe(0);
    expect(result.stdout).toContain('Updating integrations/claude-code/.claude-plugin/plugin.json');
    expect(result.stdout).toContain('✓ Updated version: 0.2.0 → 0.5.0');

    const manifestText = readFileSync(
      join(root, 'integrations', 'claude-code', MANIFEST_RELPATH),
      'utf-8',
    );
    expect(manifestText).toBe(MANIFEST.replace('"version": "0.2.0"', '"version": "0.5.0"'));

    const packageJson = readJson(join(root, 'integrations', 'claude-code', 'package.json'));
    expect(packageJson.version).toBe('0.5.0');
    expect(packageJson.dependencies['@mlflow/core']).toBe('^0.5.0');
    expect(readJson(join(root, 'core', 'package.json')).version).toBe('0.5.0');
  });

  it('bumps a plugin manifest in any integration that ships one', () => {
    const codexManifest = join(root, 'integrations', 'codex', MANIFEST_RELPATH);
    writeFileSync(codexManifest, MANIFEST.replace('mlflow-tracing', 'codex-tracing'));

    const result = runBump(root, '1.0.0-rc.1');

    expect(result.status).toBe(0);
    expect(readJson(codexManifest).version).toBe('1.0.0-rc.1');
  });

  it('fails without writing anything when a required manifest is missing', () => {
    rmSync(join(root, 'integrations', 'claude-code', MANIFEST_RELPATH));
    const before = snapshot(root);

    const result = runBump(root, '0.5.0');

    expect(result.status).toBe(1);
    expect(result.stderr).toContain('plugin.json does not exist');
    expect(snapshot(root)).toEqual(before);
  });

  it('fails without writing anything when the manifest has no string version', () => {
    writeFileSync(
      join(root, 'integrations', 'claude-code', MANIFEST_RELPATH),
      '{\n  "name": "mlflow-tracing",\n  "version": 2\n}\n',
    );
    const before = snapshot(root);

    const result = runBump(root, '0.5.0');

    expect(result.status).toBe(1);
    expect(result.stderr).toContain("has no string 'version' field");
    expect(snapshot(root)).toEqual(before);
  });

  it('fails without writing anything when the version field cannot be rewritten in place', () => {
    // Valid JSON, but not in the `"version": "..."` form the in-place rewrite expects.
    writeFileSync(
      join(root, 'integrations', 'claude-code', MANIFEST_RELPATH),
      '{\n  "name": "mlflow-tracing",\n  "version" : "0.2.0"\n}\n',
    );
    const before = snapshot(root);

    const result = runBump(root, '0.5.0');

    expect(result.status).toBe(1);
    expect(result.stderr).toContain("Failed to rewrite 'version'");
    expect(snapshot(root)).toEqual(before);
  });
});
