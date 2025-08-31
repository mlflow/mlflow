const fs = require("fs");

function computeExecutionTimeInSeconds(started_at, completed_at) {
  const startedAt = new Date(started_at);
  const completedAt = new Date(completed_at);
  return (completedAt - startedAt) / 1000;
}

async function download({ github, context }) {
  const allArtifacts = await github.rest.actions.listWorkflowRunArtifacts({
    owner: context.repo.owner,
    repo: context.repo.repo,
    run_id: context.payload.workflow_run.id,
  });

  const matchArtifact = allArtifacts.data.artifacts.find((artifact) => {
    return artifact.name == "pr_number";
  });

  const download = await github.rest.actions.downloadArtifact({
    owner: context.repo.owner,
    repo: context.repo.repo,
    artifact_id: matchArtifact.id,
    archive_format: "zip",
  });

  fs.writeFileSync(`${process.env.GITHUB_WORKSPACE}/pr_number.zip`, Buffer.from(download.data));
}

async function rerun({ github, context }) {
  const pull_number = Number(fs.readFileSync("./pr_number"));
  const {
    repo: { owner, repo },
  } = context;

  const { data: pr } = await github.rest.pulls.get({
    owner,
    repo,
    pull_number,
  });

  const checkRuns = await github.paginate(github.rest.checks.listForRef, {
    owner,
    repo,
    ref: pr.head.sha,
  });
  const runIdsToRerun = checkRuns
    // Select failed/cancelled github action runs
    .filter(
      ({ name, status, conclusion, started_at, completed_at, app: { slug } }) =>
        slug === "github-actions" &&
        status === "completed" &&
        (conclusion === "failure" || conclusion === "cancelled") &&
        name.toLowerCase() !== "rerun" && // Prevent recursive rerun
        (name.toLowerCase() === "protect" || // Always rerun protect job
          computeExecutionTimeInSeconds(started_at, completed_at) <= 60) // Rerun jobs that took less than 60 seconds (e.g. Maintainer approval check)
    )
    .map(
      ({
        // Example: https://github.com/mlflow/mlflow/actions/runs/10675586265/job/29587793829
        //                                                        ^^^^^^^^^^^ run_id
        html_url,
      }) => html_url.match(/\/actions\/runs\/(\d+)/)[1]
    );

  const uniqueRunIds = [...new Set(runIdsToRerun)];
  const promises = uniqueRunIds.map(async (run_id) => {
    console.log(`Rerunning https://github.com/${owner}/${repo}/actions/runs/${run_id}`);
    try {
      await github.rest.actions.reRunWorkflowFailedJobs({
        repo,
        owner,
        run_id,
      });
    } catch (error) {
      console.error(`Failed to rerun workflow for run_id ${run_id}:`, error);
    }
  });
  await Promise.all(promises);
}

module.exports = {
  download,
  rerun,
};
