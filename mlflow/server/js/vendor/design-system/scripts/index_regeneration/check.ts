import assert from 'assert';
import { readFileSync } from 'fs';
import path from 'path';

import { getRegeneratedIndex } from './utils';

const folders = ['../../src/design-system', '../../src/development'];

folders.forEach((folder: string) => {
  assert(
    readFileSync(path.resolve(__dirname, folder, 'index.ts'), 'utf-8') === getRegeneratedIndex(folder),
    `${folder}/index.ts is not regenerated. Please run \`yarn indexes-regenerate\`.`,
  );
});
