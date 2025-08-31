function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
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

  const title = "&#x1F6E0 DevTools &#x1F6E0";
  if (body && !body.includes(title)) {
    const codespacesBadge = `[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/${user.login}/mlflow/pull/${issue_number}?quickstart=1)`;
    const newSection = `
<details><summary>${title}</summary>
<p>

${codespacesBadge}

#### Install mlflow from this PR

\`\`\`
# mlflow
pip install git+https://github.com/mlflow/mlflow.git@refs/pull/${issue_number}/merge
# mlflow-skinny
pip install git+https://github.com/mlflow/mlflow.git@refs/pull/${issue_number}/merge#subdirectory=libs/skinny
\`\`\`

For Databricks, use the following command:

\`\`\`
%sh curl -LsSf https://raw.githubusercontent.com/mlflow/mlflow/HEAD/dev/install-skinny.sh | sh -s pull/${issue_number}/merge
\`\`\`

</p>
</details>
`.trim();
    await github.rest.pulls.update({
      owner,
      repo,
      pull_number: issue_number,
      body: `${newSection}\n\n${body}`,
    });
  }

  const dcoCheck = await getDcoCheck(github, owner, repo, sha);
  if (dcoCheck && dcoCheck.conclusion !== "success") {
    messages.push(
      "#### &#x26a0; DCO check\n\n" +
        "The DCO check failed. " +
        `Please sign off your commit(s) by following the instructions [here](${dcoCheck.html_url}). ` +
        "See https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md#sign-your-work for more " +
        "details."
    );
  }

  if (label.endsWith(":master")) {
    messages.push(
      "#### &#x26a0; PR branch check\n\n" +
        "This PR was filed from the master branch in your fork, which is not recommended " +
        "and may cause our CI checks to fail. Please close this PR and file a new PR from " +
        "a non-master branch."
    );
  }

  if (!(body || "").includes("How should the PR be classified in the release notes?")) {
    messages.push(
      "#### &#x26a0; Invalid PR template\n\n" +
        "This PR does not appear to have been filed using the MLflow PR template. " +
        "Please copy the PR template from [here](https://raw.githubusercontent.com/mlflow/mlflow/master/.github/pull_request_template.md) " +
        "and fill it out."
    );
  }

  if (messages.length > 0) {
    const body =
      `@${user.login} Thank you for the contribution! Could you fix the following issue(s)?\n\n` +
      messages.join("\n\n");
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number,
      body,
    });
  }
};
