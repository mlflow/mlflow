const { spawnSync } = require("child_process");

function exec(cmd, args) {
  console.log(`> ${cmd} ${args.join(" ")}`);

  const result = spawnSync(cmd, args, {
    stdio: "inherit",
  });

  if (result.status !== 0) {
    throw new Error(`Command failed with exit code ${result.status}: ${cmd} ${args.join(" ")}`);
  }

  return result;
}

function execWithOutput(cmd, args) {
  console.log(`> ${cmd} ${args.join(" ")}`);

  const result = spawnSync(cmd, args, {
    encoding: "utf8",
    stdio: ["pipe", "pipe", "pipe"],
  });

  const output = (result.stdout || "") + (result.stderr || "");
  if (result.status !== 0) {
    throw new Error(
      `Command failed with exit code ${result.status}: ${cmd} ${args.join(" ")}\nOutput: ${output}`
    );
  }

  return output;
}

function getTimestamp() {
  return new Date().toISOString().replace(/[:.]/g, "-").slice(0, -5);
}

module.exports = async ({ github, context }) => {
  // Note: We intentionally avoid early exits to maximize test coverage on PRs
  // This allows testing the entire workflow except git push and PR creation

  const uvLockOutput = execWithOutput("uv", ["lock", "--upgrade"]);
  console.log(`uv lock output:\n${uvLockOutput}`);

  // Check if uv.lock has changes
  const gitStatus = execWithOutput("git", ["status", "--porcelain", "uv.lock"]);
  const hasChanges = gitStatus.trim() !== "";

  const branchName = `uv-lock-update-${getTimestamp()}`;
  exec("git", ["config", "user.name", "mlflow-app[bot]"]);
  exec("git", ["config", "user.email", "mlflow-app[bot]@users.noreply.github.com"]);
  exec("git", ["checkout", "-b", branchName]);
  // `git add` succeeds even if there are no changes
  exec("git", ["add", "uv.lock"]);
  // `--allow-empty` in case `uv.lock` is unchanged
  exec("git", ["commit", "-s", "--allow-empty", "-m", "Update uv.lock"]);
  // `--dry-run` to avoid actual push but verify it would succeed
  const isPr = context.eventName === "pull_request";
  const args = isPr ? ["--dry-run"] : [];
  exec("git", ["push", ...args, "origin", branchName]);

  // Search for existing PR
  const PR_TITLE = "Update `uv.lock`";
  const { repo, owner } = context.repo;
  const { data: searchResults } = await github.rest.search.issuesAndPullRequests({
    q: `repo:${owner}/${repo} is:pr is:open base:master "${PR_TITLE}" in:title`,
    per_page: 1,
  });

  if (isPr) {
    console.log("In pull request mode, not pushing changes or creating a new PR");
    return;
  } else if (searchResults.total_count > 0) {
    console.log(`An open PR already exists: ${searchResults.items[0].html_url}`);
    return;
  } else if (!hasChanges) {
    console.log("No changes to uv.lock, not creating a PR");
    return;
  }

  // Create PR
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
${uvLockOutput.trim()}
\`\`\`

Created by: ${runUrl}
`,
  });
  console.log(`Created PR: ${pr.html_url}`);

  // Add team-review label to request review from the team
  await github.rest.issues.addLabels({
    owner,
    repo,
    issue_number: pr.number,
    labels: ["team-review"],
  });
  console.log("Added team-review label to the PR");
};
