import assert from 'assert';
import { readFileSync } from 'fs';
import path from 'path';

import { generateSafexKeysFileContent } from './utils';

const designSystemDirectory = path.resolve(__dirname, '../../storybook');

assert(
  readFileSync(path.resolve(designSystemDirectory, 'safexKeys.ts'), 'utf-8') === generateSafexKeysFileContent(),
  'storybook/safexKeys.ts is not regenerated. Please run `yarn storybook:generate-safex-keys`.',
);
