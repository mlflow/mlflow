/**
 * Resolves the TypeScript SDK publish matrix by checking which packages have
 * new commits since their last release tag.
 *
 * Usage:
 *   npx tsx libs/typescript/scripts/resolve-ts-publish-matrix.ts --packages <comma-separated|all>
 *
 * Outputs GitHub Actions key=value pairs to stdout:
 *   matrix={"core":{"publish":true,"version":"0.2.0","npm_name":"@mlflow/core","dir":"libs/typescript/core"}, ...}
 *   any_publish=true
 *
 * Requirements: node, git (with full history and tags)
 */

import { execSync } from 'child_process';
import { readFileSync } from 'fs';
import path from 'path';

interface PackageInfo {
  npm_name: string;
  dir: string;
}

interface MatrixEntry {
  publish: boolean;
  version: string;
  npm_name: string;
  dir: string;
}

const PACKAGE_REGISTRY: Record<string, PackageInfo> = {
  core: { npm_name: '@mlflow/core', dir: 'libs/typescript/core' },
  openai: {
    npm_name: '@mlflow/openai',
    dir: 'libs/typescript/integrations/openai',
  },
  anthropic: {
    npm_name: '@mlflow/anthropic',
    dir: 'libs/typescript/integrations/anthropic',
  },
  gemini: {
    npm_name: '@mlflow/gemini',
    dir: 'libs/typescript/integrations/gemini',
  },
};

const ALL_PACKAGES = Object.keys(PACKAGE_REGISTRY);

const REPO_ROOT =
  process.env.GITHUB_WORKSPACE ||
  execSync('git rev-parse --show-toplevel', { encoding: 'utf-8' }).trim();

function parseArgs(): string {
  const args = process.argv.slice(2);
  let packages: string | null = null;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--packages' && i + 1 < args.length) {
      packages = args[i + 1];
      i++;
    } else if (args[i] === '--help' || args[i] === '-h') {
      console.error(
        `Usage: npx tsx libs/typescript/scripts/resolve-ts-publish-matrix.ts --packages <comma-separated|all>`,
      );
      console.error(`\nValid package names: all, ${ALL_PACKAGES.join(', ')}`);
      process.exit(0);
    } else {
      console.error(`Error: Unknown argument: ${args[i]}`);
      process.exit(1);
    }
  }

  if (!packages) {
    console.error('Error: --packages is required');
    process.exit(1);
  }

  return packages;
}

function resolveSelected(packagesArg: string): string[] {
  if (packagesArg === 'all') {
    return [...ALL_PACKAGES];
  }

  const selected: string[] = [];
  for (const raw of packagesArg.split(',')) {
    const pkg = raw.trim();
    if (!PACKAGE_REGISTRY[pkg]) {
      console.error(
        `Error: Unknown package: '${pkg}'. Valid values: all, ${ALL_PACKAGES.join(', ')}`,
      );
      process.exit(1);
    }
    selected.push(pkg);
  }

  if (selected.length === 0) {
    console.error('Error: No packages selected.');
    process.exit(1);
  }

  return selected;
}

function git(cmd: string): string {
  return execSync(`git ${cmd}`, { encoding: 'utf-8' }).trim();
}

function readVersion(dir: string): string {
  const pkgJson: { version: string } = JSON.parse(
    readFileSync(path.join(REPO_ROOT, dir, 'package.json'), 'utf-8'),
  );
  return pkgJson.version;
}

function shouldPublish(npmName: string, version: string, dir: string): boolean {
  const tag = `${npmName}@${version}`;

  try {
    git(`rev-parse "refs/tags/${tag}"`);
  } catch {
    console.error(`${npmName}: no existing tag '${tag}' found (first release of this version)`);
    return true;
  }

  const log = git(`log "${tag}..HEAD" --oneline -- "${dir}"`);
  const commitCount = log === '' ? 0 : log.split('\n').length;

  if (commitCount === 0) {
    console.error(`Skipping ${npmName}@${version}: no commits since tag ${tag}`);
    return false;
  }

  console.error(`${npmName}: ${commitCount} commit(s) since ${tag}`);
  return true;
}

function main(): void {
  const packagesArg = parseArgs();
  const selected = resolveSelected(packagesArg);

  console.error(`Selected packages: ${selected.join(' ')}`);

  const matrix: Record<string, MatrixEntry> = {};
  let anyPublish = false;

  for (const pkg of ALL_PACKAGES) {
    const { npm_name, dir } = PACKAGE_REGISTRY[pkg];
    const version = readVersion(dir);
    const publish = selected.includes(pkg) && shouldPublish(npm_name, version, dir);

    if (publish) {
      anyPublish = true;
    }

    matrix[pkg] = { publish, version, npm_name, dir };
  }

  console.log(`matrix=${JSON.stringify(matrix)}`);
  console.log(`any_publish=${anyPublish}`);
}

main();
