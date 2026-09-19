const fs = require("fs");
const {
  extractComponentIdsFromSource,
  buildRegistrySource,
  getRegistryPath,
  REGISTRY_PATH,
} = require("./utils");

const registryPath = getRegistryPath(__dirname);
const registry = require(registryPath);

// --- Main ---
const codeIds = extractComponentIdsFromSource(__dirname);
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
  console.error(`\nAdd these to ${REGISTRY_PATH}`);
}

if (stale.length > 0) {
  failed = true;
  console.error(`\n❌ Found ${stale.length} stale componentId(s) in registry but NOT in code:\n`);
  for (const id of stale) {
    console.error(`  - ${id}`);
  }
  console.error(`\nRemove these from ${REGISTRY_PATH}`);
}

// Check 3: file must be byte-identical to the generator's output. The
// registry is excluded from prettier, so nothing else normalizes hand-edits.
if (!failed && fs.readFileSync(registryPath, "utf8") !== buildRegistrySource(codeIds, registry)) {
  failed = true;
  console.error("\n❌ Registry file does not match the canonical generator output.");
  console.error("Run: node .github/actions/check-component-ids/regenerate.js");
}

if (failed) {
  process.exit(1);
} else {
  console.log(`✅ componentId registry is in sync. ${registryKeys.size} entries verified.`);
}
