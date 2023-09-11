module.exports = async ({ context, github, workflow_id }) => {
  const { owner, repo } = context.repo;
  const { data: workflowRunsData } = await github.rest.actions.listWorkflowRuns({
    owner,
    repo,
    workflow_id,
    event: "schedule",
  });

  if (workflowRunsData.total_count === 0) {
    return;
  }

  const { id: run_id, conclusion } = workflowRunsData.workflow_runs[0];
  if (conclusion === "success") {
    return;
  }

  const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
    owner,
    repo,
    run_id,
  });
  const failedJobs = jobs.filter((job) => job.conclusion !== "success");
  if (failedJobs.length === 0) {
    return;
  }

  await github.rest.actions.reRunWorkflowFailedJobs({
    repo,
    owner,
    run_id,
  });
};
