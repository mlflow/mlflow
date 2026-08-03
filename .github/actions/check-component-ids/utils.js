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

/**
 * Extract all static componentIds from the MLflow UI source directory.
 * @param {string} actionDir - path to this action's directory (used to resolve the repo root)
 * @returns {Set<string>} set of componentId strings found in source
 */
function extractComponentIdsFromSource(actionDir) {
  const srcDir = path.resolve(
    process.env.GITHUB_WORKSPACE || path.join(actionDir, "../../.."),
    "mlflow/server/js/src"
  );
  const files = findFiles(srcDir);
  return extractComponentIds(files);
}

module.exports = { extractComponentIdsFromSource };
