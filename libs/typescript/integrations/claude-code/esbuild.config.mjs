import { build } from 'esbuild';
import { chmodSync } from 'node:fs';

const banner = { js: '#!/usr/bin/env node' };

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
    external: ['node:*'],
    banner,
  });

  chmodSync(outfile, 0o755);
}
