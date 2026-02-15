#!/usr/bin/env node
/**
 * Script to create missing size/* labels using GitHub REST API
 * Usage: GITHUB_TOKEN=<token> GITHUB_REPOSITORY=mlflow/mlflow node dev/create-size-labels.js
 */

const https = require("https");

const GITHUB_TOKEN = process.env.GITHUB_TOKEN;
const GITHUB_REPOSITORY = process.env.GITHUB_REPOSITORY || "mlflow/mlflow";

if (!GITHUB_TOKEN) {
  console.error("Error: GITHUB_TOKEN environment variable is required");
  process.exit(1);
}

const [owner, repo] = GITHUB_REPOSITORY.split("/");

const labels = [
  { name: "size/XS", color: "ededed", description: "Extra-small PR (0-9 LoC)" },
  { name: "size/S", color: "ededed", description: "Small PR (10-49 LoC)" },
  { name: "size/M", color: "ededed", description: "Medium PR (50-199 LoC)" },
  { name: "size/L", color: "ededed", description: "Large PR (200-499 LoC)" },
  { name: "size/XL", color: "ededed", description: "Extra-large PR (500+ LoC)" },
];

function makeRequest(method, path, data = null) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: "api.github.com",
      port: 443,
      path,
      method,
      headers: {
        "User-Agent": "mlflow-label-creator",
        Authorization: `token ${GITHUB_TOKEN}`,
        Accept: "application/vnd.github.v3+json",
        "Content-Type": "application/json",
      },
    };

    const req = https.request(options, (res) => {
      let body = "";
      res.on("data", (chunk) => (body += chunk));
      res.on("end", () => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          resolve({ statusCode: res.statusCode, data: JSON.parse(body || "{}") });
        } else {
          reject({
            statusCode: res.statusCode,
            message: body,
          });
        }
      });
    });

    req.on("error", reject);
    if (data) {
      req.write(JSON.stringify(data));
    }
    req.end();
  });
}

async function getLabel(name) {
  try {
    const result = await makeRequest(
      "GET",
      `/repos/${owner}/${repo}/labels/${encodeURIComponent(name)}`
    );
    return result.data;
  } catch (error) {
    if (error.statusCode === 404) {
      return null;
    }
    throw error;
  }
}

async function createLabel(label) {
  return makeRequest("POST", `/repos/${owner}/${repo}/labels`, label);
}

async function updateLabel(name, label) {
  return makeRequest("PATCH", `/repos/${owner}/${repo}/labels/${encodeURIComponent(name)}`, label);
}

async function main() {
  console.log(`Creating/updating size/* labels in ${owner}/${repo}...\n`);

  for (const label of labels) {
    try {
      const existing = await getLabel(label.name);

      if (existing) {
        console.log(`Label ${label.name} exists, updating...`);
        await updateLabel(label.name, {
          color: label.color,
          description: label.description,
        });
        console.log(`✓ Updated ${label.name}`);
      } else {
        console.log(`Label ${label.name} doesn't exist, creating...`);
        await createLabel(label);
        console.log(`✓ Created ${label.name}`);
      }
    } catch (error) {
      console.error(`✗ Error processing ${label.name}:`, error.message || error);
      process.exit(1);
    }
  }

  console.log("\n✓ All size/* labels have been created/updated successfully!");

  // Verify the labels
  console.log("\nVerifying labels...");
  const allLabels = await makeRequest("GET", `/repos/${owner}/${repo}/labels?per_page=100`);
  const sizeLabels = allLabels.data.filter((l) => l.name.startsWith("size/"));

  console.log("\nCurrent size/* labels:");
  for (const label of sizeLabels) {
    console.log(`  ${label.name}: ${label.description || "(no description)"}`);
  }
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
