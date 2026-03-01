#!/usr/bin/env node
/**
 * Publishes TypeScript SDK packages and creates git tags based on the
 * resolved publish matrix.
 *
 * Usage:
 *   node dev/typescript-sdk/ts-publish.js --matrix '<JSON>' [--dry-run]
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
      console.error(`Usage: node dev/typescript-sdk/ts-publish.js --matrix '<JSON>' [--dry-run]`);
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

function commandSucceeds(cmd, opts = {}) {
  try {
    execSync(cmd, { stdio: "pipe", ...opts });
    return true;
  } catch {
    return false;
  }
}

function tagExists(tag, workspace) {
  return commandSucceeds(`git rev-parse --verify "refs/tags/${tag}"`, {
    cwd: workspace,
  });
}

function npmPublishState(name, version) {
  try {
    execSync(`npm view "${name}@${version}" version --json`, {
      stdio: "pipe",
      encoding: "utf-8",
    });
    return "published";
  } catch (error) {
    const output = `${error.stdout || ""}\n${error.stderr || ""}`;
    if (output.includes("E404")) {
      return "missing";
    }
    console.error(`Error: Unable to query npm registry for ${name}@${version}.\n${output}`);
    process.exit(1);
  }
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
  let taggedOnly = 0;

  for (const pkg of ordered) {
    const { publish, version, npm_name, dir } = matrix[pkg];
    const tag = `${npm_name}@${version}`;

    if (!publish) {
      console.log(`Skipping ${npm_name} (publish=false)`);
      continue;
    }

    const pkgDir = path.resolve(workspace, dir);
    const hasTag = tagExists(tag, workspace);
    const onNpm = npmPublishState(npm_name, version) === "published";

    if (onNpm && hasTag) {
      console.log(`Skipping ${npm_name}@${version} (already published and tagged)`);
      continue;
    }

    if (!onNpm && hasTag) {
      console.error(
        `Error: ${npm_name}@${version} has git tag '${tag}' but is not published on npm.`
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
      run("npm publish --dry-run", { cwd: pkgDir });
      continue;
    }

    run("npm publish --provenance --access public", { cwd: pkgDir });
    console.log(`Tagging ${tag}...`);
    run(`git tag "${tag}"`, { cwd: workspace });
    run(`git push origin "${tag}"`, { cwd: workspace });
    published++;
  }

  if (dryRun) {
    console.log("\nDone. Dry run completed.");
    return;
  }

  console.log(
    `\nDone. ${published} package(s) published, ${taggedOnly} existing package(s) tagged.`
  );
}

main();
