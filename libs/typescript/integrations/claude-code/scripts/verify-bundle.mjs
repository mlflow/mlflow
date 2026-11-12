#!/usr/bin/env node
/**
 * Guard the shipped plugin bundle against silently losing the WAL daemon.
 *
 * `esbuild.config.mjs` copies `bundle/daemon.cjs` next to the hook bundle
 * so marketplace/plugin-cache installs (which have no node_modules tree)
 * can spawn it. The unit tests stub `existsSync`, so they cannot catch a
 * build refactor that drops the copy. This post-build check runs as part
 * of `npm run build` and asserts the artifact is actually present.
 */
import { statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const packageDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const daemonPath = join(packageDir, 'bundle', 'daemon.cjs');

let stats;
try {
  stats = statSync(daemonPath);
} catch {
  throw new Error(`missing ${daemonPath}; the plugin bundle must ship the WAL daemon`);
}

if (!stats.isFile()) {
  throw new Error(`${daemonPath} is not a file`);
}

if (process.platform !== 'win32' && (stats.mode & 0o111) === 0) {
  throw new Error(`${daemonPath} is not executable`);
}
