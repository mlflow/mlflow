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
  const uuid = Array.from({ length: 16 }, () => Math.floor(Math.random() * 16).toString(16)).join(
    ""
  );
  const workflow_id = "cross-version-tests.yml";
  await github.rest.actions.createWorkflowDispatch({
    owner,
    repo,
    workflow_id,
    ref: pr.base.ref,
    inputs: {
      repository: `${owner}/${repo}`,
      ref: pr.merge_commit_sha,
      flavors,
      // The response of create-workflow-dispatch request doesn't contain the ID of the triggered
      // workflow run. We need to pass a unique identifier to the workflow run and find the run by
      // the identifier. See https://github.com/orgs/community/discussions/9752 for more details.
      uuid,
    },
  });

  // Find the triggered workflow run
  let run;
  const maxAttempts = 5;
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise((resolve) => setTimeout(resolve, 5000));

    const { data: runs } = await github.rest.actions.listWorkflowRunsForRepo({
      owner,
      repo,
      workflow_id,
      event: "workflow_dispatch",
    });
    run = runs.workflow_runs.find((run) => run.name.includes(uuid));
    if (run) {
      break;
    }
  }

  if (!run) {
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: pull_number,
      body: "Failed to find the triggered workflow run.",
    });
    return;
  }

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
