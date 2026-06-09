import { build } from 'esbuild';
import { chmodSync } from 'node:fs';

await build({
  entryPoints: ['dist/cli.js'],
  bundle: true,
  platform: 'node',
  format: 'esm',
  outfile: 'bundle/cli.js',
  external: ['node:*'],
  banner: {
    js: [
      '#!/usr/bin/env node',
      'import { createRequire as __createRequire } from "node:module";',
      'const require = __createRequire(import.meta.url);',
    ].join('\n'),
  },
});

chmodSync('bundle/cli.js', 0o755);
