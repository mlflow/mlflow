/**
 * Publishes TypeScript SDK packages and creates git tags based on the
 * resolved publish matrix.
 *
 * Usage:
 *   npx tsx libs/typescript/scripts/ts-publish.ts --matrix '<JSON>' [--dry-run]
 *
 * The matrix JSON is produced by resolve-ts-publish-matrix.ts.
 * Packages are published in dependency order: core first, then integrations alphabetically.
 */

import { execSync } from 'child_process';
import path from 'path';

interface PackageEntry {
  publish: boolean;
  version: string;
  npm_name: string;
  dir: string;
}

type Matrix = Record<string, PackageEntry>;

const NETWORK_ERROR_PATTERNS = [
  'ETIMEDOUT',
  'ECONNREFUSED',
  'ECONNRESET',
  'ENOTFOUND',
  'EAI_AGAIN',
];

function parseArgs(): { matrix: Matrix; dryRun: boolean } {
  const args = process.argv.slice(2);
  let matrixJson: string | null = null;
  let dryRun = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--matrix' && i + 1 < args.length) {
      matrixJson = args[i + 1];
      i++;
    } else if (args[i] === '--dry-run') {
      dryRun = true;
    } else if (args[i] === '--help' || args[i] === '-h') {
      console.error(
        `Usage: npx tsx libs/typescript/scripts/ts-publish.ts --matrix '<JSON>' [--dry-run]`,
      );
      process.exit(0);
    } else {
      console.error(`Error: Unknown argument: ${args[i]}`);
      process.exit(1);
    }
  }

  if (!matrixJson) {
    console.error('Error: --matrix is required');
    process.exit(1);
  }

  let matrix: Matrix;
  try {
    matrix = JSON.parse(matrixJson);
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    console.error(`Error: Failed to parse --matrix JSON: ${message}`);
    process.exit(1);
  }

  return { matrix, dryRun };
}

function run(cmd: string, opts: Record<string, unknown> = {}): void {
  console.log(`  > ${cmd}`);
  execSync(cmd, { stdio: 'inherit', ...opts });
}

function tagExists(tag: string): boolean {
  const output = execSync(`git ls-remote --tags origin "refs/tags/${tag}"`, {
    stdio: 'pipe',
    encoding: 'utf-8',
  }).trim();
  return output.length > 0;
}

function npmPublishState(name: string, version: string): 'published' | 'missing' {
  try {
    execSync(`npm view "${name}@${version}" version --json`, {
      stdio: 'pipe',
      encoding: 'utf-8',
    });
    return 'published';
  } catch (error: unknown) {
    const err = error as { stdout?: string; stderr?: string };
    const output = `${err.stdout || ''}\n${err.stderr || ''}`;

    if (NETWORK_ERROR_PATTERNS.some((pattern) => output.includes(pattern))) {
      console.error(
        `Error: Network error while querying npm registry for ${name}@${version}.\n${output}`,
      );
      process.exit(1);
    }

    if (output.includes('E404')) {
      return 'missing';
    }

    console.error(`Error: Unable to query npm registry for ${name}@${version}.\n${output}`);
    process.exit(1);
  }
}

/**
 * Returns package keys in publish order: "core" first, then remaining keys alphabetically.
 */
function publishOrder(matrix: Matrix): string[] {
  const keys = Object.keys(matrix);
  const order: string[] = [];
  if (keys.includes('core')) {
    order.push('core');
  }
  for (const k of keys.sort()) {
    if (k !== 'core') {
      order.push(k);
    }
  }
  return order;
}

function main(): void {
  const { matrix, dryRun } = parseArgs();
  const workspace =
    process.env.GITHUB_WORKSPACE ||
    execSync('git rev-parse --show-toplevel', { encoding: 'utf-8' }).trim();

  if (dryRun) {
    console.log('DRY RUN — packages will not be published to npm\n');
  }

  const ordered = publishOrder(matrix);
  let published = 0;
  let taggedOnly = 0;

  for (const pkg of ordered) {
    const { publish, version, npm_name, dir } = matrix[pkg];
    const tag = `${npm_name}@${version}`;

    if (!publish) {
      console.log(`Skipping ${npm_name} (publish=false)`);
      continue;
    }

    const pkgDir = path.resolve(workspace, dir);
    const hasTag = tagExists(tag);
    const onNpm = npmPublishState(npm_name, version) === 'published';

    if (onNpm && hasTag) {
      console.log(`Skipping ${npm_name}@${version} (already published and tagged)`);
      continue;
    }

    if (!onNpm && hasTag) {
      console.error(
        `Error: ${npm_name}@${version} has git tag '${tag}' but is not published on npm.`,
      );
      process.exit(1);
    }

    if (onNpm && !hasTag) {
      if (dryRun) {
        console.log(`\nWould tag ${tag} (already published on npm).`);
      } else {
        console.log(`\nTagging ${tag} (already published on npm)...`);
        run(`git tag "${tag}"`, { cwd: workspace });
        run(`git push origin "${tag}"`, { cwd: workspace });
        taggedOnly++;
      }
      continue;
    }

    console.log(`\nPublishing ${npm_name}@${version}...`);
    if (dryRun) {
      run('npm publish --dry-run', { cwd: pkgDir });
      continue;
    }

    run('npm publish --provenance --access public', { cwd: pkgDir });
    console.log(`Tagging ${tag}...`);
    run(`git tag "${tag}"`, { cwd: workspace });
    run(`git push origin "${tag}"`, { cwd: workspace });
    published++;
  }

  if (dryRun) {
    console.log('\nDone. Dry run completed.');
    return;
  }

  console.log(
    `\nDone. ${published} package(s) published, ${taggedOnly} existing package(s) tagged.`,
  );
}

main();
