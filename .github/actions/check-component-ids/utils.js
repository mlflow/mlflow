const fs = require("fs");
const path = require("path");

const EXTENSIONS = [".js", ".jsx", ".ts", ".tsx"];
// Skip test files — they don't need registered componentIds
const TEST_PATTERN = /\.test\.[jt]sx?$/;

const EXTRACT_PATTERNS = [
  /(?:componentId|data-component-id)=["']([^"']+)["']/g,
  /componentId:\s*["']([^"']+)["']/g,
  // Match static strings inside JSX expressions like componentId={"value"},
  // componentId={cond ?? "fallback"}, componentId={cond ? "a" : "b"}, etc.
  // Uses [^\n}]* to avoid matching across lines.
  /componentId=\{[^\n}]*["']([^"'\n`]+)["'][^\n}]*\}/g,
];

function findFiles(dir) {
  const results = [];
  function walk(d) {
    for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
      const full = path.join(d, entry.name);
      if (entry.isDirectory() && !entry.name.startsWith(".") && entry.name !== "node_modules") {
        walk(full);
      } else if (
        entry.isFile() &&
        EXTENSIONS.some((ext) => full.endsWith(ext)) &&
        !TEST_PATTERN.test(full)
      ) {
        results.push(full);
      }
    }
  }
  walk(dir);
  return results;
}

function extractComponentIds(files) {
  const ids = new Set();
  for (const file of files) {
    const content = fs.readFileSync(file, "utf8");
    for (const pat of EXTRACT_PATTERNS) {
      pat.lastIndex = 0;
      let m;
      while ((m = pat.exec(content)) !== null) {
        ids.add(m[1]);
      }
    }
  }
  return ids;
}

function getRepoRoot(actionDir) {
  return process.env.GITHUB_WORKSPACE || path.join(actionDir, "../../..");
}

/**
 * Extract all static componentIds from the MLflow UI source directory.
 * @param {string} actionDir - path to this action's directory (used to resolve the repo root)
 * @returns {Set<string>} set of componentId strings found in source
 */
function extractComponentIdsFromSource(actionDir) {
  const srcDir = path.resolve(getRepoRoot(actionDir), "mlflow/server/js/src");
  const files = findFiles(srcDir);
  return extractComponentIds(files);
}

// The registry lives under mlflow/server/js so that registry-only updates in
// UI PRs don't escape master.yml's paths-ignore and trigger the Python suite.
const REGISTRY_PATH = "mlflow/server/js/scripts/componentId-registry.js";

/**
 * Resolve the absolute path to the componentId registry.
 * @param {string} actionDir - path to this action's directory (used to resolve the repo root)
 * @returns {string} absolute path to the registry module
 */
function getRegistryPath(actionDir) {
  return path.resolve(getRepoRoot(actionDir), REGISTRY_PATH);
}

/**
 * Build the canonical source text of the componentId registry. This exact
 * output is enforced byte-for-byte by index.js — the registry is excluded
 * from prettier, so the generator's output is the only formatting authority.
 * @param {Set<string>} codeIds - componentIds extracted from source
 * @param {Object} existingDescriptions - id -> description map to preserve
 * @returns {string} registry module source
 */
function buildRegistrySource(codeIds, existingDescriptions) {
  const sorted = [...codeIds].sort();

  // Group by prefix for readability
  const groups = {};
  for (const id of sorted) {
    let prefix;
    if (id.startsWith("codegen_")) {
      prefix = "Codegen (auto-generated)";
    } else if (id.startsWith("mlflow.")) {
      const parts = id.split(".");
      prefix = parts[0] + "." + parts[1];
    } else if (id.startsWith("shared.")) {
      const parts = id.split(".");
      prefix = parts[0] + "." + parts[1];
    } else {
      prefix = "Other";
    }
    if (!groups[prefix]) groups[prefix] = [];
    groups[prefix].push(id);
  }

  let output = `/**
 * Curated registry of all componentIds used in the MLflow UI.
 *
 * Every static componentId string literal in non-test source files must
 * have an entry here. The CI job \`check-component-ids\` verifies this
 * bidirectionally: code IDs must be in the registry, and registry
 * entries must exist in code.
 *
 * Format: key = componentId string, value = optional description of the
 * component (blank by default, especially for generated entries)
 */
module.exports = {\n`;

  for (const gk of Object.keys(groups).sort()) {
    output += `  // -- ${gk} --\n`;
    for (const id of groups[gk]) {
      const escaped = id.replace(/"/g, '\\"');
      const desc = (existingDescriptions[id] || "").replace(/"/g, '\\"');
      output += `  "${escaped}": "${desc}",\n`;
    }
    output += "\n";
  }
  // Drop the blank line after the last group
  output = output.slice(0, -1) + "};\n";
  return output;
}

module.exports = {
  extractComponentIdsFromSource,
  buildRegistrySource,
  getRegistryPath,
  REGISTRY_PATH,
};
