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
  const { user } = context.payload.pull_request;
  const messages = [""];

  const dcoCheck = await getDcoCheck(github, owner, repo, sha);
  if (dcoCheck.conclusion !== "success") {
    messages.push(
      "The DCO check failed. " +
        `Please sign off your commit(s) by following the instructions [here](${dcoCheck.html_url}). ` +
        "See https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md#sign-your-work for more " +
        "details."
    );
  }

  if (label.endsWith(":master")) {
    messages.push(
      "This PR is filed from the master branch in your fork, which is not recommended " +
        "and may cause our CI checks to fail. Please close this PR and file a new PR from " +
        "a non-master branch."
    );
  }

  if (messages.length > 1) {
    const body = `@${user.login} Thank you for the contribution!` + messages.join("\n\n---\n\n");
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number,
      body,
    });
  }
};
