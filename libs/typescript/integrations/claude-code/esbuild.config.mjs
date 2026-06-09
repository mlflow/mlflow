import { build } from 'esbuild';
import { chmodSync } from 'node:fs';

const banner = { js: '#!/usr/bin/env node' };

// @mlflow/core's WAL supervisor resolves the daemon bundle at runtime
// via `require.resolve('@mlflow/core/package.json')` (string-concatenated
// to defeat static analysis) and then `path.join(..., 'bundle',
// 'daemon.cjs')`. Marking both paths as `external` makes the intent
// explicit and guarantees esbuild leaves any future references to them
// alone — the daemon ships as its own binary inside @mlflow/core and
// must be loaded from the installed package on disk, not inlined into
// the hook bundle.
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
