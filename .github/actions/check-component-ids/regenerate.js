#!/usr/bin/env node

/**
 * Regenerates the componentId registry from source code.
 *
 * Usage (from repo root):
 *   node .github/actions/check-component-ids/regenerate.js
 */

const fs = require("fs");
const path = require("path");
const { extractComponentIdsFromSource, buildRegistrySource } = require("./utils");

const codeIds = extractComponentIdsFromSource(__dirname);

// Load existing registry to preserve descriptions
let existingDescriptions = {};
try {
  existingDescriptions = require("./componentId-registry");
} catch {
  // First run or broken registry — start fresh
}

const output = buildRegistrySource(codeIds, existingDescriptions);

const outPath = path.join(__dirname, "componentId-registry.js");
fs.writeFileSync(outPath, output);
console.log(`✅ Registry regenerated with ${codeIds.size} entries at ${outPath}`);
