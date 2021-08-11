module.exports = async ({ context, github }) => {
  const { owner, repo } = context.repo;
  const { number: issue_number } = context.issue;
  const { sha: ref, user } = context.payload.pull_request.head;

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async function getDcoCheck() {
    const backoffs = [0, 2, 4, 6, 8];
    const numAttempts = backoffs.length;
    for (const [index, backoff] of backoffs.entries()) {
      await sleep(backoff * 1000);
      const resp = await github.checks.listForRef({
        owner,
        repo,
        ref,
        app_id: 1861, // ID of the DCO check app
      });

      const { check_runs } = resp.data;
      if (check_runs.length > 0 && check_runs[0].status === "completed") {
        return check_runs[0];
      }
      console.log(
        `[Attempt ${index + 1}/${numAttempts}]`,
        "The DCO check hasn't completed yet."
      );
    }
  }

  const dcoCheck = await getDcoCheck();
  if (dcoCheck.conclusion !== "success") {
    const body = [
      `@${user.login} Thanks for the contribution! The DCO check failed. `,
      `Please sign off your commits by following the instructions here: ${dcoCheck.html_url}. `,
      "See https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.rst#sign-your-work for more ",
      "details.",
    ].join("");
    await github.issues.createComment({
      owner,
      repo,
      issue_number,
      body,
    });
  }
};
