const { execSync } = require("child_process");

function exec(cmd) {
  console.log(`> ${cmd}`);
  return execSync(cmd, { stdio: "inherit" });
}

function execWithOutput(cmd) {
  console.log(`> ${cmd}`);
  return execSync(cmd, { encoding: "utf8" });
}

function getDaysAgo(days) {
  const date = new Date();
  date.setDate(date.getDate() - days);
  return date.toISOString().replace(/\.\d{3}Z$/, "Z");
}

function getTimestamp() {
  return new Date().toISOString().replace(/[:.]/g, "-").slice(0, -5);
}

module.exports = async ({ github, context }) => {
  // Note: We intentionally avoid early exits to maximize test coverage on PRs
  // This allows testing the entire workflow except git push and PR creation

  // Run uv lock with --exclude-newer flag to avoid potentially unstable releases
  const uvOutput = execWithOutput(`uv lock --upgrade --exclude-newer "${getDaysAgo(3)}"`);

  const { repo, owner } = context.repo;
  const PR_TITLE = "Update `uv.lock`";

  // Configure git user before any git operations
  exec("git config user.name 'mlflow-app[bot]'");
  exec("git config user.email 'mlflow-app[bot]@users.noreply.github.com'");

  const branchName = `uv-lock-update-${getTimestamp()}`;
  exec(`git checkout -b ${branchName}`);
  exec("git add uv.lock");
  exec('git commit -s --allow-empty -m "Update uv.lock"'); // --allow-empty in case uv.lock is unchanged
  exec(`git push --dry-run origin ${branchName}`); // Dry run to verify push would succeed

  // Search for existing PR
  const { data: searchResults } = await github.rest.search.issuesAndPullRequests({
    q: `repo:${owner}/${repo} is:pr is:open base:master "${PR_TITLE}" in:title`,
    per_page: 1,
  });

  if (context.eventName === "pull_request") {
    console.log("In pull request mode, not pushing changes or creating a new PR");
    return;
  } else if (searchResults.total_count > 0) {
    console.log(`An open PR already exists: ${searchResults.items[0].html_url}`);
    return;
  }

  // Push branch and create PR
  exec(`git push origin ${branchName}`);
  const runUrl = `https://github.com/${owner}/${repo}/actions/runs/${context.runId}`;
  const { data: pr } = await github.rest.pulls.create({
    owner,
    repo,
    title: PR_TITLE,
    head: branchName,
    base: "master",
    body: `This PR was created automatically to update \`uv.lock\`.

### \`uv lock\` output

\`\`\`
${uvOutput.trim()}
\`\`\`

Created by: ${runUrl}
`,
    labels: ["team-review"],
  });
  console.log(`Created PR: ${pr.html_url}`);
};
