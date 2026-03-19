const ACTIVITY_WINDOW_MS = 14 * 24 * 60 * 60 * 1000;
const MAX_REPOS_TO_DISPLAY = 10;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function getRecentActivity(github, username) {
  const windowStart = new Date(Date.now() - ACTIVITY_WINDOW_MS);
  const dateString = windowStart.toISOString().slice(0, 10);
  const query = `type:pr author:${username} created:>${dateString}`;
  const items = await github.paginate(github.rest.search.issuesAndPullRequests, {
    q: query,
    per_page: 100,
  });
  const repoCounts = new Map();
  for (const item of items) {
    const repoFullName = item.repository_url.replace("https://api.github.com/repos/", "");
    if (!repoCounts.has(repoFullName)) {
      repoCounts.set(repoFullName, { open: 0, closed: 0, merged: 0 });
    }
    const counts = repoCounts.get(repoFullName);
    if (item.pull_request?.merged_at) {
      counts.merged++;
    } else if (item.state === "closed") {
      counts.closed++;
    } else {
      counts.open++;
    }
  }
  return { totalPRs: items.length, repoCount: repoCounts.size, repoBreakdown: repoCounts };
}

async function getRecentActivitySection(github, username) {
  const { totalPRs, repoCount, repoBreakdown } = await getRecentActivity(github, username);
  if (totalPRs === 0) {
    return "";
  }
  const prLabel = totalPRs === 1 ? "PR" : "PRs";
  const repoLabel = repoCount === 1 ? "repo" : "repos";
  const total = ({ open, closed, merged }) => open + closed + merged;
  const sortedRepos = [...repoBreakdown.entries()]
    .sort((a, b) => total(b[1]) - total(a[1]))
    .slice(0, MAX_REPOS_TO_DISPLAY);
  const tableRows = sortedRepos
    .map(
      ([repo, counts]) =>
        `| [${repo}](https://github.com/${repo}/pulls/${username}) | ${counts.open} | ${
          counts.closed
        } | ${counts.merged} | ${total(counts)} |`
    )
    .join("\n");
  const topNote = repoCount > MAX_REPOS_TO_DISPLAY ? ` (showing top ${MAX_REPOS_TO_DISPLAY})` : "";
  return `
<details><summary>PR author's recent activity</summary>

In the last 14 days, @${username} opened **${totalPRs} ${prLabel}** across **${repoCount} ${repoLabel}**${topNote}:

| Repository | Open | Closed | Merged | Total |
| ---------- | ---- | ------ | ------ | ----- |
${tableRows}

</details>`;
}

async function getDcoCheck(github, owner, repo, sha) {
  const backoffs = [0, 2, 4, 6, 8];
  const numAttempts = backoffs.length;
  for (const [index, backoff] of backoffs.entries()) {
    await sleep(backoff * 1000);
    const resp = await github.rest.checks.listForRef({
      owner,
      repo,
      ref: sha,
      app_id: 1861, // ID of the DCO check app
    });

    const { check_runs } = resp.data;
    if (check_runs.length > 0 && check_runs[0].status === "completed") {
      return check_runs[0];
    }
    console.log(`[Attempt ${index + 1}/${numAttempts}]`, "The DCO check hasn't completed yet.");
  }
}

module.exports = async ({ context, github }) => {
  const { owner, repo } = context.repo;
  const { number: issue_number } = context.issue;
  const { sha, label } = context.payload.pull_request.head;
  const { user, body } = context.payload.pull_request;
  const messages = [];

  const title = "Install mlflow from this PR";
  // Check if an install comment already exists
  const comments = await github.paginate(github.rest.issues.listComments, {
    owner,
    repo,
    issue_number,
  });
  const installCommentExists = comments.some((comment) => comment.body.includes(title));

  if (!installCommentExists) {
    let activitySection = "";
    const memberAssociations = ["MEMBER", "OWNER", "COLLABORATOR"];
    if (
      user.type !== "Bot" &&
      !memberAssociations.includes(context.payload.pull_request.author_association)
    ) {
      try {
        activitySection = await getRecentActivitySection(github, user.login);
      } catch (e) {
        console.log("Failed to fetch recent activity:", e);
      }
    }
    const devToolsComment = `
<details><summary>${title}</summary>
<p>

#### Install mlflow from this PR

\`\`\`bash
# mlflow
pip install git+https://github.com/mlflow/mlflow.git@refs/pull/${issue_number}/merge
# mlflow-skinny
pip install git+https://github.com/mlflow/mlflow.git@refs/pull/${issue_number}/merge#subdirectory=libs/skinny
\`\`\`

For Databricks, use the following command:

\`\`\`bash
%sh curl -LsSf https://raw.githubusercontent.com/mlflow/mlflow/HEAD/dev/install-skinny.sh | sh -s pull/${issue_number}/merge
\`\`\`

</p>
</details>
${activitySection}
`.trim();
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number,
      body: devToolsComment,
    });
  }

  // Exit early if the PR author is a bot
  if (user.type === "Bot") {
    return;
  }

  const dcoCheck = await getDcoCheck(github, owner, repo, sha);
  if (dcoCheck && dcoCheck.conclusion !== "success") {
    messages.push(
      "#### &#x274C; DCO check\n\n" +
        "The DCO check failed. " +
        `Please sign off your commit(s) by following the instructions [here](${dcoCheck.html_url}). ` +
        "See https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md#sign-your-work for more " +
        "details."
    );
  }

  if (label.endsWith(":master")) {
    messages.push(
      "#### &#x274C; PR branch check\n\n" +
        "This PR was filed from the master branch in your fork, which is not recommended " +
        "and may cause our CI checks to fail. Please close this PR and file a new PR from " +
        "a non-master branch."
    );
  }

  if (!(body || "").includes("How should the PR be classified in the release notes?")) {
    messages.push(
      "#### &#x274C; Invalid PR template\n\n" +
        "The PR description is missing required sections. " +
        "Please use the [PR template](https://raw.githubusercontent.com/mlflow/mlflow/master/.github/pull_request_template.md)."
    );
  }

  if (messages.length > 0) {
    const body =
      `@${user.login} Thank you for the contribution! Could you fix the following issue(s)? Otherwise, this PR may be automatically closed.\n\n` +
      messages.join("\n\n");
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number,
      body,
    });
  }
};
