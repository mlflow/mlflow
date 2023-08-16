module.exports = async ({ context, github }) => {
  const owner = context.repo.owner;
  const repo = context.repo.repo;
  const headSha = context.payload.pull_request.head.sha;
  const prRuns = await github.paginate(github.rest.actions.listWorkflowRunsForRepo, {
    owner,
    repo,
    head_sha: headSha,
    event: "pull_request",
    per_page: 100,
  });
  const unfinishedRuns = prRuns.filter(
    ({ status, name }) =>
      // `post-merge` job in `release-note` workflow should not be cancelled
      status !== "completed" && name !== "release-note"
  );
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
