import { writeFileSync } from 'fs';
import path from 'path';

import { generateSafexKeysFileContent } from './utils';

const designSystemDirectory = path.resolve(__dirname, '../../storybook');

writeFileSync(`${designSystemDirectory}/safexKeys.ts`, generateSafexKeysFileContent());
