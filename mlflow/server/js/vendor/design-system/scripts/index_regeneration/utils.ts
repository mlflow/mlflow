/* eslint-disable import/no-extraneous-dependencies */
import { readdirSync, statSync } from 'fs';
import path from 'path';

import { sync } from 'glob';

export function getRegeneratedIndex(basePath: string): string {
  const designSystemDirectory = path.resolve(__dirname, basePath);

  const directory = readdirSync(designSystemDirectory);

  let output = '// Generated file. To regenerate: `yarn indexes-regenerate`\n\n';

  directory.forEach((directoryChildPath: string) => {
    const completePath = path.join(designSystemDirectory, directoryChildPath);
    const pathStats = statSync(completePath);
    const indexFile = sync(`${completePath}/index.*`);

    // Only add the index file if it exists and does not start with _.
    if (pathStats?.isDirectory() && indexFile?.length > 0 && directoryChildPath.indexOf('_') !== 0) {
      output += `export * from './${directoryChildPath}';\n`;
    }
  });

  return output;
}
