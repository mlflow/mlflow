module.exports = async ({ context, github }) => {
  const owner = context.repo.owner;
  const repo = context.repo.repo;
  const prNumber = context.payload.pull_request.number;
  const headSha = context.payload.pull_request.head.sha;

  // Get all workflow runs associated with the PR.
  const prRuns = await github.paginate(github.rest.actions.listWorkflowRunsForRepo, {
    owner,
    repo,
    head_sha: headSha,
    event: "pull_request",
    per_page: 100,
  });

  const unfinishedRuns = prRuns.filter((run) => run.status !== "completed");

  // Cancel the runs
  for (const run of unfinishedRuns) {
    try {
      // Some runs may have already completed, so we need to handle errors.
      await github.rest.actions.cancelWorkflowRun({
        owner,
        repo,
        run_id: run.id,
      });
      console.log(`Cancelled run ${run.id}`);
    } catch (error) {
      console.error(`Failed to cancel run ${run.id}`, error);
    }
  }
};
