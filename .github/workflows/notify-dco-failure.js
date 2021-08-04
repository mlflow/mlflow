module.exports = async ({ context, github }) => {
  const { sha: ref } = context.payload.pull_request.head;
  const { owner, repo } = context.repo;
  const { number: issue_number } = context.issue;
  console.log(context, sha);

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async function getDcoCheck() {
    for (const sec of [0, 2, 4, 6, 8]) {
      await sleep(sec * 1000);
      const resp = await github.checks.listForRef({
        owner,
        repo,
        ref,
        app_id: 1861, // ID of the DCO check app
      });

      const { check_runs } = resp.data;
      if (check_runs.length > 0) {
        return check_runs[0];
      }
    }
  }

  const dcoCheck = await getDcoCheck();
  const { html_url, conclusion } = dcoCheck;

  if (conclusion === "success") {
    const body = `The DCO check failed. Please sign off your commits:\n${html_url}`;
    await github.issues.createComment({
      owner,
      repo,
      issue_number,
      body,
    });
  }
};
