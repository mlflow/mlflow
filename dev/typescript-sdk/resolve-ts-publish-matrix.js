#!/usr/bin/env node
/**
 * Resolves the TypeScript SDK publish matrix by checking which packages have
 * new commits since their last release tag.
 *
 * Usage:
 *   node dev/resolve-ts-publish-matrix.js --packages <comma-separated|all>
 *
 * Outputs GitHub Actions key=value pairs to stdout:
 *   matrix={"core":{"publish":true,"version":"0.2.0","npm_name":"@mlflow/core","dir":"libs/typescript/core"}, ...}
 *   any_publish=true
 *
 * Requirements: node, git (with full history and tags)
 */

const { execSync } = require("child_process");
const path = require("path");
const fs = require("fs");

const PACKAGE_REGISTRY = {
  core: { npm_name: "@mlflow/core", dir: "libs/typescript/core" },
  openai: {
    npm_name: "@mlflow/openai",
    dir: "libs/typescript/integrations/openai",
  },
  anthropic: {
    npm_name: "@mlflow/anthropic",
    dir: "libs/typescript/integrations/anthropic",
  },
  gemini: {
    npm_name: "@mlflow/gemini",
    dir: "libs/typescript/integrations/gemini",
  },
};

const ALL_PACKAGES = Object.keys(PACKAGE_REGISTRY);

function parseArgs() {
  const args = process.argv.slice(2);
  let packages = null;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--packages" && i + 1 < args.length) {
      packages = args[i + 1];
      i++;
    } else if (args[i] === "--help" || args[i] === "-h") {
      console.error(
        `Usage: node dev/resolve-ts-publish-matrix.js --packages <comma-separated|all>`
      );
      console.error(`\nValid package names: all, ${ALL_PACKAGES.join(", ")}`);
      process.exit(0);
    } else {
      console.error(`Error: Unknown argument: ${args[i]}`);
      process.exit(1);
    }
  }

  if (!packages) {
    console.error("Error: --packages is required");
    process.exit(1);
  }

  return packages;
}

function resolveSelected(packagesArg) {
  if (packagesArg === "all") {
    return [...ALL_PACKAGES];
  }

  const selected = [];
  for (const raw of packagesArg.split(",")) {
    const pkg = raw.trim();
    if (!PACKAGE_REGISTRY[pkg]) {
      console.error(
        `Error: Unknown package: '${pkg}'. Valid values: all, ${ALL_PACKAGES.join(", ")}`
      );
      process.exit(1);
    }
    selected.push(pkg);
  }

  if (selected.length === 0) {
    console.error("Error: No packages selected.");
    process.exit(1);
  }

  return selected;
}

function git(cmd) {
  return execSync(`git ${cmd}`, { encoding: "utf-8" }).trim();
}

function readVersion(dir) {
  const pkgJson = JSON.parse(fs.readFileSync(path.join(dir, "package.json"), "utf-8"));
  return pkgJson.version;
}

function shouldPublish(npmName, version, dir) {
  const tag = `${npmName}@${version}`;

  try {
    git(`rev-parse "refs/tags/${tag}"`);
  } catch {
    console.error(`${npmName}: no existing tag '${tag}' found (first release of this version)`);
    return true;
  }

  const log = git(`log "${tag}..HEAD" --oneline -- "${dir}"`);
  const commitCount = log === "" ? 0 : log.split("\n").length;

  if (commitCount === 0) {
    console.error(`Skipping ${npmName}@${version}: no commits since tag ${tag}`);
    return false;
  }

  console.error(`${npmName}: ${commitCount} commit(s) since ${tag}`);
  return true;
}

function main() {
  const packagesArg = parseArgs();
  const selected = resolveSelected(packagesArg);

  console.error(`Selected packages: ${selected.join(" ")}`);

  const matrix = {};
  let anyPublish = false;

  for (const pkg of ALL_PACKAGES) {
    const { npm_name, dir } = PACKAGE_REGISTRY[pkg];
    const version = readVersion(dir);
    const publish = selected.includes(pkg) && shouldPublish(npm_name, version, dir);

    if (publish) anyPublish = true;

    matrix[pkg] = { publish, version, npm_name, dir };
  }

  // Output GitHub Actions key=value pairs
  console.log(`matrix=${JSON.stringify(matrix)}`);
  console.log(`any_publish=${anyPublish}`);
}

main();
