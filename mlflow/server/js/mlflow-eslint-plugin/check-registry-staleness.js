#!/usr/bin/env node

/**
 * Checks for stale entries in the componentId registry — entries that
 * exist in the registry but are no longer found in any source file.
 *
 * Usage: node check-registry-staleness.js
 * Exits with code 1 if stale entries are found.
 */

const fs = require('fs');
const path = require('path');

const registry = require('./componentId-registry');

const SRC_DIR = path.resolve(__dirname, '../src');
const EXTENSIONS = ['.js', '.jsx', '.ts', '.tsx'];

function findFiles(dir) {
  const results = [];
  function walk(d) {
    for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
      const full = path.join(d, entry.name);
      if (entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'node_modules') {
        walk(full);
      } else if (entry.isFile() && EXTENSIONS.some((ext) => full.endsWith(ext))) {
        results.push(full);
      }
    }
  }
  walk(dir);
  return results;
}

// Collect all static componentId values found in source files
const foundIds = new Set();
const patterns = [
  /(?:componentId|data-component-id)=["']([^"']+)["']/g,
  /componentId:\s*["']([^"']+)["']/g,
  /componentId=\{["']([^"']+)["']\}/g,
];

const files = findFiles(SRC_DIR);
for (const file of files) {
  const content = fs.readFileSync(file, 'utf8');
  for (const pat of patterns) {
    pat.lastIndex = 0;
    let m;
    while ((m = pat.exec(content)) !== null) {
      foundIds.add(m[1]);
    }
  }
}

// Find stale registry entries
const registryKeys = Object.keys(registry);
const staleEntries = registryKeys.filter((key) => !foundIds.has(key));

if (staleEntries.length > 0) {
  console.error(`Found ${staleEntries.length} stale componentId(s) in registry:\n`);
  for (const entry of staleEntries) {
    console.error(`  - ${entry}`);
  }
  console.error(
    '\nThese entries exist in componentId-registry.js but were not found in any source file.',
  );
  console.error('Remove them from the registry to fix this.');
  process.exit(1);
} else {
  console.log(`Registry is clean. All ${registryKeys.length} entries found in source code.`);
}
