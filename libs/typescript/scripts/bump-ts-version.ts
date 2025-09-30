#!/usr/bin/env tsx
/**
 * CLI script to bump the version of MLflow TypeScript libraries.
 *
 * This script updates the version in:
 * - libs/typescript/core/package.json
 * - libs/typescript/integrations/openai/package.json (version and peerDependencies)
 */

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';

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
  const openaiPackagePath = join(tsRoot, 'integrations', 'openai', 'package.json');

  // Validate that files exist
  if (!existsSync(corePackagePath)) {
    console.error(`Error: ${corePackagePath} does not exist`);
    process.exit(1);
  }

  if (!existsSync(openaiPackagePath)) {
    console.error(`Error: ${openaiPackagePath} does not exist`);
    process.exit(1);
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

  // Update openai package.json
  console.log(`Updating integrations/openai/package.json...`);
  const openaiPackageText = readFileSync(openaiPackagePath, 'utf-8');
  const openaiPackage: PackageJson = JSON.parse(openaiPackageText);

  const oldOpenaiVersion = openaiPackage.version;
  openaiPackage.version = version;

  // Update peerDependencies for mlflow-tracing
  if (openaiPackage.peerDependencies && 'mlflow-tracing' in openaiPackage.peerDependencies) {
    const oldPeerDep = openaiPackage.peerDependencies['mlflow-tracing'];
    // Update peerDependency to match new version (keep the ^ prefix)
    openaiPackage.peerDependencies['mlflow-tracing'] = `^${version}`;
    console.log(`  ✓ Updated peerDependency mlflow-tracing: ${oldPeerDep} → ^${version}`);
  }

  writeFileSync(openaiPackagePath, JSON.stringify(openaiPackage, null, 2) + '\n', 'utf-8');
  console.log(`  ✓ Updated version: ${oldOpenaiVersion} → ${version}`);

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
