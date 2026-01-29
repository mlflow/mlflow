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

function generatePrBody(uvLockOutput, runUrl) {
  return `This PR was created automatically to update \`uv.lock\`.

### \`uv lock\` output

\`\`\`
${uvLockOutput.trim()}
\`\`\`

Created by: ${runUrl}
`;
}

module.exports = async ({ github, context }) => {
  const uvLockOutput = execWithOutput("uv", ["lock", "--upgrade"]);
  console.log(`uv lock output:\n${uvLockOutput}`);

  // Check if uv.lock has changes
  const gitStatus = execWithOutput("git", ["status", "--porcelain", "uv.lock"]);
  const hasChanges = gitStatus.trim() !== "";

  const isPr = context.eventName === "pull_request";
  if (isPr) {
    console.log("In pull request mode, exiting early");
    return;
  }

  if (!hasChanges) {
    console.log("No changes to uv.lock, exiting early");
    return;
  }

  // Search for existing PR
  const PR_TITLE = "Update `uv.lock`";
  const { repo, owner } = context.repo;
  const { data: searchResults } = await github.rest.search.issuesAndPullRequests({
    q: `repo:${owner}/${repo} is:pr is:open base:master "${PR_TITLE}" in:title`,
    per_page: 1,
  });

  const runUrl = `https://github.com/${owner}/${repo}/actions/runs/${context.runId}`;

  exec("git", ["config", "user.name", "mlflow-app[bot]"]);
  exec("git", ["config", "user.email", "mlflow-app[bot]@users.noreply.github.com"]);
  exec("git", ["add", "uv.lock"]);
  exec("git", ["commit", "-s", "-m", "Update uv.lock"]);

  if (searchResults.total_count > 0) {
    // Existing PR found - update it
    const existingPr = searchResults.items[0];
    console.log(`An open PR already exists: ${existingPr.html_url}`);

    const { data: prDetails } = await github.rest.pulls.get({
      owner,
      repo,
      pull_number: existingPr.number,
    });
    const existingBranch = prDetails.head.ref;

    exec("git", ["push", "--force", "origin", `HEAD:${existingBranch}`]);
    console.log(`Force pushed changes to existing branch: ${existingBranch}`);

    await github.rest.pulls.update({
      owner,
      repo,
      pull_number: existingPr.number,
      body: generatePrBody(uvLockOutput, runUrl),
    });
    console.log("Updated PR description");
    return;
  }

  // Create new PR
  const branchName = `uv-lock-update-${getTimestamp()}`;
  exec("git", ["push", "origin", `HEAD:${branchName}`]);
  const { data: pr } = await github.rest.pulls.create({
    owner,
    repo,
    title: PR_TITLE,
    head: branchName,
    base: "master",
    body: generatePrBody(uvLockOutput, runUrl),
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
