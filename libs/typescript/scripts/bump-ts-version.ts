/**
 * CLI script to bump the version of MLflow TypeScript libraries.
 *
 * This script updates the version in:
 * - libs/typescript/core/package.json
 * - libs/typescript/integrations/openai/package.json (version and peerDependencies)
 */

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join } from 'path';

// list of packages that contain `mlflow-tracing` in peerDependencies
const INTEGRATION_PACKAGES = ['openai', 'vercel'];

interface PackageJson {
  name: string;
  version: string;
  peerDependencies?: Record<string, string>;
  [key: string]: any;
}

function bumpVersion(version: string): void {
  // Define paths to package.json files
  const tsRoot = process.cwd();
  const corePackagePath = join(tsRoot, 'core', 'package.json');

  // Validate that files exist
  if (!existsSync(corePackagePath)) {
    console.error(`Error: ${corePackagePath} does not exist`);
    process.exit(1);
  }

  for (const packageName of INTEGRATION_PACKAGES) {
    const packagePath = join(tsRoot, 'integrations', packageName, 'package.json');
    if (!existsSync(packagePath)) {
      console.error(`Error: ${packagePath} does not exist`);
      process.exit(1);
    }
  }

  // Validate version format (semver with optional prerelease)
  // Matches: X.Y.Z or X.Y.Z-rc.0 or X.Y.Z-beta.1 etc.
  const semverPattern = /^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?$/;
  if (!semverPattern.test(version)) {
    console.error(
      `Error: Invalid version format '${version}'. Expected format: X.Y.Z or X.Y.Z-rc.0`
    );
    process.exit(1);
  }

  // Update core package.json
  console.log(`Updating core/package.json...`);
  const corePackageText = readFileSync(corePackagePath, 'utf-8');
  const corePackage: PackageJson = JSON.parse(corePackageText);

  const oldCoreVersion = corePackage.version;
  corePackage.version = version;

  writeFileSync(corePackagePath, JSON.stringify(corePackage, null, 2) + '\n', 'utf-8');
  console.log(`  ✓ Updated version: ${oldCoreVersion} → ${version}`);

  for (const packageName of INTEGRATION_PACKAGES) {
    const packagePath = join(tsRoot, 'integrations', packageName, 'package.json');
    console.log(`Updating integrations/${packageName}/package.json...`);
    const packageText = readFileSync(packagePath, 'utf-8');
    const packageJson: PackageJson = JSON.parse(packageText);

    const oldVersion = packageJson.version;
    packageJson.version = version;

    writeFileSync(packagePath, JSON.stringify(packageJson, null, 2) + '\n', 'utf-8');
    console.log(`  ✓ Updated version: ${oldVersion} → ${version}`);

    if (packageJson.peerDependencies && 'mlflow-tracing' in packageJson.peerDependencies) {
      const oldPeerDep = packageJson.peerDependencies['mlflow-tracing'];
      packageJson.peerDependencies['mlflow-tracing'] = `^${version}`;
      writeFileSync(packagePath, JSON.stringify(packageJson, null, 2) + '\n', 'utf-8');
      console.log(`  ✓ Updated peerDependency mlflow-tracing: ${oldPeerDep} → ^${version}`);
    }
  }

  console.log(`\n✅ Successfully bumped TypeScript library versions to ${version}`);
}

function main(): void {
  const args = process.argv.slice(2);

  // Parse arguments
  let version: string | null = null;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--version') {
      if (i + 1 < args.length) {
        version = args[i + 1];
        i++;
      } else {
        console.error('Error: --version requires a value');
        process.exit(1);
      }
    } else if (args[i] === '--help' || args[i] === '-h') {
      console.log('Usage: tsx scripts/bump-ts-version.ts --version <version>');
      console.log('\nBump the version of MLflow TypeScript libraries');
      console.log('\nOptions:');
      console.log('  --version <version>  Version to bump to (e.g., 0.1.2, 0.2.0, or 1.0.0-rc.0)');
      console.log('  --help, -h           Show this help message');
      process.exit(0);
    }
  }

  if (!version) {
    console.error('Error: --version is required');
    console.log('\nUsage: tsx scripts/bump-ts-version.ts --version <version>');
    process.exit(1);
  }

  bumpVersion(version);
}

main();
