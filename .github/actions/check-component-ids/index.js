const fs = require("fs");
const path = require("path");

const registry = require("./componentId-registry");

const SRC_DIR = path.resolve(
  process.env.GITHUB_WORKSPACE || path.join(__dirname, "../../.."),
  "mlflow/server/js/src"
);
const EXTENSIONS = [".js", ".jsx", ".ts", ".tsx"];
// Skip test files — they don't need registered componentIds
const TEST_PATTERN = /\.test\.[jt]sx?$/;

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

const EXTRACT_PATTERNS = [
  /(?:componentId|data-component-id)=["']([^"']+)["']/g,
  /componentId:\s*["']([^"']+)["']/g,
  /componentId=\{["']([^"']+)["']\}/g,
];

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

// --- Main ---
const files = findFiles(SRC_DIR);
const codeIds = extractComponentIds(files);
const registryKeys = new Set(Object.keys(registry));

// Check 1: componentIds in code but not in registry
const unregistered = [...codeIds].filter((id) => !registryKeys.has(id)).sort();

// Check 2: componentIds in registry but not in code (stale)
const stale = [...registryKeys].filter((id) => !codeIds.has(id)).sort();

let failed = false;

if (unregistered.length > 0) {
  failed = true;
  console.error(
    `\n❌ Found ${unregistered.length} componentId(s) in code but NOT in the registry:\n`
  );
  for (const id of unregistered) {
    console.error(`  + ${id}`);
  }
  console.error("\nAdd these to .github/actions/check-component-ids/componentId-registry.js");
}

if (stale.length > 0) {
  failed = true;
  console.error(`\n❌ Found ${stale.length} stale componentId(s) in registry but NOT in code:\n`);
  for (const id of stale) {
    console.error(`  - ${id}`);
  }
  console.error("\nRemove these from .github/actions/check-component-ids/componentId-registry.js");
}

if (failed) {
  process.exit(1);
} else {
  console.log(`✅ componentId registry is in sync. ${registryKeys.size} entries verified.`);
}
