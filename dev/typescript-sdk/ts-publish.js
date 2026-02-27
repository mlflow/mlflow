#!/usr/bin/env node
/**
 * Publishes TypeScript SDK packages and creates git tags based on the
 * resolved publish matrix.
 *
 * Usage:
 *   node dev/ts-publish.js --matrix '<JSON>' [--dry-run]
 *
 * The matrix JSON is produced by resolve-ts-publish-matrix.js.
 * Packages are published in dependency order: core first, then integrations alphabetically.
 */

const { execSync } = require("child_process");
const path = require("path");

function parseArgs() {
  const args = process.argv.slice(2);
  let matrixJson = null;
  let dryRun = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--matrix" && i + 1 < args.length) {
      matrixJson = args[i + 1];
      i++;
    } else if (args[i] === "--dry-run") {
      dryRun = true;
    } else if (args[i] === "--help" || args[i] === "-h") {
      console.error(
        `Usage: node dev/ts-publish.js --matrix '<JSON>' [--dry-run]`
      );
      process.exit(0);
    } else {
      console.error(`Error: Unknown argument: ${args[i]}`);
      process.exit(1);
    }
  }

  if (!matrixJson) {
    console.error("Error: --matrix is required");
    process.exit(1);
  }

  return { matrix: JSON.parse(matrixJson), dryRun };
}

function run(cmd, opts = {}) {
  console.log(`  > ${cmd}`);
  execSync(cmd, { stdio: "inherit", ...opts });
}

/**
 * Returns package keys in publish order: "core" first, then remaining keys alphabetically.
 */
function publishOrder(matrix) {
  const keys = Object.keys(matrix);
  const order = [];
  if (keys.includes("core")) order.push("core");
  for (const k of keys.sort()) {
    if (k !== "core") order.push(k);
  }
  return order;
}

function main() {
  const { matrix, dryRun } = parseArgs();
  const workspace = process.env.GITHUB_WORKSPACE || process.cwd();

  if (dryRun) {
    console.log("DRY RUN — packages will not be published to npm\n");
  }

  const ordered = publishOrder(matrix);
  let published = 0;

  for (const pkg of ordered) {
    const { publish, version, npm_name, dir } = matrix[pkg];

    if (!publish) {
      console.log(`Skipping ${npm_name} (publish=false)`);
      continue;
    }

    const pkgDir = path.resolve(workspace, dir);

    console.log(`\nPublishing ${npm_name}@${version}...`);
    if (dryRun) {
      run("npm publish --dry-run", { cwd: pkgDir });
    } else {
      run("npm publish --provenance --access public", { cwd: pkgDir });
    }

    if (!dryRun) {
      const tag = `${npm_name}@${version}`;
      console.log(`Tagging ${tag}...`);
      run(`git tag "${tag}"`, { cwd: workspace });
      run(`git push origin "${tag}"`, { cwd: workspace });
    }

    published++;
  }

  console.log(`\nDone. ${published} package(s) ${dryRun ? "would be" : ""} published.`);
}

main();
