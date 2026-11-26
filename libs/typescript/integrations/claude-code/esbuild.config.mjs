import { build } from 'esbuild';
import { chmodSync, copyFileSync, existsSync, mkdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';

const banner = { js: '#!/usr/bin/env node' };

// The WAL supervisor's `resolveDaemonEntry()` looks for a `daemon.cjs`
// sibling of the executing bundle before falling back to resolving
// `@mlflow/core` through node_modules. Marketplace/plugin-cache installs
// ship only `bundle/*.cjs` with no node_modules, so the daemon must ride
// along: copy the byte-identical bundle that @mlflow/core ships.
//
// Resolve the source through the package entry point (rather than a
// cwd-relative workspace path) so the copied daemon always matches the
// exact @mlflow/core the hook was built against. The workspace build
// order still guarantees core is built first: see `build:subpackages`
// in the libs/typescript root package.json.
const corePkg = createRequire(import.meta.url).resolve('@mlflow/core/package.json');
const daemonSource = join(dirname(corePkg), 'bundle', 'daemon.cjs');
if (!existsSync(daemonSource)) {
  throw new Error(`missing ${daemonSource}; build @mlflow/core first (npm run build:core)`);
}
mkdirSync('bundle', { recursive: true });
copyFileSync(daemonSource, 'bundle/daemon.cjs');
chmodSync('bundle/daemon.cjs', 0o755);

// @mlflow/core's WAL supervisor resolves the daemon bundle at runtime
// via `require.resolve('@mlflow/core/package.json')` (string-concatenated
// to defeat static analysis) and then `path.join(..., 'bundle',
// 'daemon.cjs')`. Marking both paths as `external` makes the intent
// explicit and guarantees esbuild leaves any future references to them
// alone — the daemon ships as its own binary inside @mlflow/core (and,
// as a sibling copy above, inside this plugin) and must be loaded from
// the installed files on disk, not inlined into the hook bundle.
const external = ['node:*', '@mlflow/core/package.json', '@mlflow/core/bundle/daemon.cjs'];

for (const [entryPoint, outfile] of [
  ['dist/hooks/stop.js', 'bundle/stop.cjs'],
  ['dist/cli.js', 'bundle/cli.cjs'],
]) {
  await build({
    entryPoints: [entryPoint],
    bundle: true,
    platform: 'node',
    format: 'cjs',
    outfile,
    external,
    banner,
  });

  chmodSync(outfile, 0o755);
}
