/**
 * Bundles the WAL uploader daemon into a single, executable CJS file.
 *
 * Why a bundle:
 *   - The daemon is spawned as a child process by the supervisor (see
 *     `src/exporters/wal/supervisor.ts`), which resolves it at runtime
 *     via `@mlflow/core/package.json` + `bundle/daemon.cjs`. Shipping a
 *     single self-contained file means we don't have to worry about
 *     transitive `dist/` files or `node_modules` layout in the target
 *     environment.
 *   - The `#!/usr/bin/env node` banner + `chmod 0o755` makes the bundle
 *     directly executable, which is how the npm `bin` field invokes it.
 */

import { build } from 'esbuild';
import { chmodSync } from 'node:fs';

const outfile = 'bundle/daemon.cjs';

await build({
  entryPoints: ['dist/exporters/wal/daemon.js'],
  bundle: true,
  platform: 'node',
  format: 'cjs',
  outfile,
  external: ['node:*'],
  banner: { js: '#!/usr/bin/env node' },
});

chmodSync(outfile, 0o755);
