import { writeFileSync } from 'fs';
import path from 'path';

import { getRegeneratedIndex } from './utils';

const folders = ['../../src/design-system', '../../src/development'];

folders.forEach((folder: string) => {
  const directory = path.resolve(__dirname, folder);
  writeFileSync(`${directory}/index.ts`, getRegeneratedIndex(folder));
});
