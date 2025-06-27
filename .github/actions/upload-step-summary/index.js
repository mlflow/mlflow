module.exports = async ({ context, core }) => {
  const fs = require("fs");
  const path = require("path");
  const os = require("os");
  try {
    const summaryPath = process.env.GITHUB_STEP_SUMMARY;

    if (!summaryPath) {
      core.error("GITHUB_STEP_SUMMARY environment variable is not set");
      core.setOutput("exists", "false");
      return;
    }

    const runId = context.runId || "unknown";
    const runAttempt = process.env.GITHUB_RUN_ATTEMPT || "1";
    const outputFileName = `job-summary-${runId}-${runAttempt}.md`;
    const outputFilePath = path.join(os.tmpdir(), outputFileName);
    const summaryContents = [];

    const summaryDir = path.dirname(summaryPath);
    const files = fs.readdirSync(summaryDir);

    core.info(`Summary directory: ${summaryDir}`);
    core.info(`All files in summary directory: ${files.join(", ")}`);

    const summaryFiles = files.filter((file) => file.startsWith("step_summary_")).sort();
    core.info(`Summary files found: ${summaryFiles.join(", ")}`);

    for (const file of summaryFiles) {
      const filePath = path.join(summaryDir, file);
      const stat = fs.statSync(filePath);

      if (!stat.isFile() || stat.size === 0) {
        continue;
      }

      const content = fs.readFileSync(filePath, "utf8");
      summaryContents.push(content);
    }

    if (summaryContents.length === 0) {
      core.warning("No summary content found");
      core.setOutput("exists", "false");
      return;
    }

    const mergedContent = summaryContents.join("\n---\n\n");
    fs.writeFileSync(outputFilePath, mergedContent, "utf8");

    core.setOutput("exists", "true");
    core.setOutput("markdown_path", outputFilePath);
  } catch (error) {
    core.setFailed(`Action failed: ${error.message}`);
  }
};
