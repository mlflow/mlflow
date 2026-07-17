#!/usr/bin/env node

/**
 * Regenerates the componentId registry from source code.
 *
 * Usage (from repo root):
 *   node .github/actions/check-component-ids/regenerate.js
 */

const fs = require("fs");
const path = require("path");
const { extractComponentIdsFromSource } = require("./utils");

const codeIds = extractComponentIdsFromSource(__dirname);
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

// Load existing registry to preserve descriptions
let existingDescriptions = {};
try {
  existingDescriptions = require("./componentId-registry");
} catch {
  // First run or broken registry — start fresh
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
output += "};\n";

const outPath = path.join(__dirname, "componentId-registry.js");
fs.writeFileSync(outPath, output);
console.log(`✅ Registry regenerated with ${sorted.length} entries at ${outPath}`);
