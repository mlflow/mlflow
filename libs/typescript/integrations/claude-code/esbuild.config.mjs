import { build } from 'esbuild';
import { chmodSync } from 'node:fs';

await build({
  entryPoints: ['dist/hooks/stop.js'],
  bundle: true,
  platform: 'node',
  format: 'esm',
  outfile: 'bundle/stop.js',
  external: ['node:*'],
  banner: {
    // Create a require function for CJS dependencies that use bare node specifiers
    js: [
      '#!/usr/bin/env node',
      'import { createRequire as __createRequire } from "node:module";',
      'const require = __createRequire(import.meta.url);',
    ].join('\n'),
  },
});

chmodSync('bundle/stop.js', 0o755);
