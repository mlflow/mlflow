import { execSync } from 'child_process';
import crypto from 'crypto';
import fs from 'fs';
import os from 'os';
import path from 'path';

const GENERATED_FOLDER = 'src/theme/_generated';

function runCommand(command: string): string {
  return execSync(command, { encoding: 'utf-8', stdio: 'pipe' }).trim();
}

function calculateHash(filePath: string): string {
  const fileContent = fs.readFileSync(filePath);
  return crypto.createHash('md5').update(fileContent).digest('hex');
}

function getDirectoryHash(dirPath: string): { [filename: string]: string } {
  const files = fs.readdirSync(dirPath);
  const hashes: { [filename: string]: string } = {};

  for (const file of files) {
    const filePath = path.join(dirPath, file);
    if (fs.statSync(filePath).isFile()) {
      hashes[file] = calculateHash(filePath);
    }
  }

  return hashes;
}

function compareDirectories(dir1: string, dir2: string): boolean {
  const hashes1 = getDirectoryHash(dir1);
  const hashes2 = getDirectoryHash(dir2);

  return JSON.stringify(hashes1) === JSON.stringify(hashes2);
}

function checkThemeGeneration(): void {
  const rootDir = path.resolve(__dirname, '../../');
  const originalGeneratedDir = path.join(rootDir, GENERATED_FOLDER);
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'theme-generation-'));

  // eslint-disable-next-line no-console -- TODO(FEINF-3587)
  console.log('Checking theme generation...');

  // Run the regeneration script with the temp directory
  // eslint-disable-next-line no-console -- TODO(FEINF-3587)
  console.log('Running theme regeneration script...');
  process.env.THEME_GENERATED_DIR = tempDir;
  runCommand('yarn theme-regenerate');

  // Run Prettier on the temp directory
  // eslint-disable-next-line no-console -- TODO(FEINF-3587)
  console.log('Running Prettier on generated files...');
  runCommand(`yarn databricks-prettier ${tempDir} --config ${rootDir}/prettier.config.js --write`);

  // eslint-disable-next-line no-console -- TODO(FEINF-3587)
  console.log('Temp directory:', tempDir);
  // eslint-disable-next-line no-console -- TODO(FEINF-3587)
  console.log('Original generated directory:', originalGeneratedDir);

  // Compare the original directory with the temp directory
  const isIdentical = compareDirectories(originalGeneratedDir, tempDir);

  if (!isIdentical) {
    // eslint-disable-next-line no-console -- TODO(FEINF-3587)
    console.error('Error: Generated theme files are not up-to-date.');
    // eslint-disable-next-line no-console -- TODO(FEINF-3587)
    console.error('Please run `yarn theme-regenerate` and commit the changes.');
    // Clean up the temp directory
    // fs.rmSync(tempDir, { recursive: true, force: true });
    process.exit(1);
  } else {
    // eslint-disable-next-line no-console -- TODO(FEINF-3587)
    console.log('Success: All generated theme files are up-to-date.');
    // Clean up the temp directory
    // fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

checkThemeGeneration();
