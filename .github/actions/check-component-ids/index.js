const { extractComponentIdsFromSource } = require("./utils");

const registry = require("./componentId-registry");

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
