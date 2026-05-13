async function main({ context, github }) {
  const { comment } = context.payload;
  const { owner, repo } = context.repo;
  const pull_number = context.issue.number;

  const { data: pr } = await github.rest.pulls.get({ owner, repo, pull_number });
  const flavorsMatch = comment.body.match(/\/(?:cross-version-test|cvt)\s+([^\n]+)\n?/);
  if (!flavorsMatch) {
    return;
  }

  // Run the workflow
  const flavors = flavorsMatch[1];
  const workflow_id = "cross-version-tests.yml";
  const { data: run } = await github.request(
    "POST /repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches",
    {
      owner,
      repo,
      workflow_id,
      ref: pr.base.ref,
      return_run_details: true,
      inputs: {
        repository: `${owner}/${repo}`,
        ref: pr.merge_commit_sha,
        flavors,
      },
    }
  );

  await github.rest.issues.createComment({
    owner,
    repo,
    issue_number: pull_number,
    body: `Cross-version test run started: ${run.html_url}`,
  });
}

module.exports = {
  main,
};
