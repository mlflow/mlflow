const createCommitStatus = async (context, github, sha, state) => {
  const { workflow, runId } = context;
  const { owner, repo } = context.repo;
  const target_url = `https://github.com/${owner}/${repo}/actions/runs/${runId}`;
  await github.rest.repos.createCommitStatus({
    owner,
    repo,
    sha,
    state,
    target_url,
    description: sha,
    context: workflow,
  });
};

const shouldAutoformat = (comment) => {
  return comment.body.trim() === "/autoformat";
};

const getPullInfo = async (context, github) => {
  const { owner, repo } = context.repo;
  const pull_number = context.issue.number;
  const pr = await github.rest.pulls.get({ owner, repo, pull_number });
  const {
    sha: head_sha,
    ref: head_ref,
    repo: { full_name },
  } = pr.data.head;
  const { sha: base_sha, ref: base_ref, repo: base_repo } = pr.data.base;
  return {
    repository: full_name,
    pull_number,
    head_sha,
    head_ref,
    base_sha,
    base_ref,
    base_repo: base_repo.full_name,
    author_association: pr.data.author_association,
  };
};

const createReaction = async (context, github) => {
  const { owner, repo } = context.repo;
  const { id: comment_id } = context.payload.comment;
  await github.rest.reactions.createForIssueComment({
    owner,
    repo,
    comment_id,
    content: "rocket",
  });
};

const createStatus = async (context, github, core) => {
  const { head_sha, head_ref, repository } = await getPullInfo(context, github);
  if (repository === "mlflow/mlflow" && head_ref === "master") {
    core.setFailed("Running autoformat bot against master branch of mlflow/mlflow is not allowed.");
  }
  await createCommitStatus(context, github, head_sha, "pending");
};

const updateStatus = async (context, github, sha, needs) => {
  const failed = Object.values(needs).some(({ result }) => result === "failure");
  const state = failed ? "failure" : "success";
  await createCommitStatus(context, github, sha, state);
};

const fetchWorkflowRuns = async ({ context, github, head_sha }) => {
  const { owner, repo } = context.repo;
  const SLEEP_DURATION_MS = 5000;
  const MAX_RETRIES = 5;
  let prevRuns = [];
  for (let i = 0; i < MAX_RETRIES; i++) {
    console.log(`Attempt ${i + 1} to fetch workflow runs`);
    const runs = await github.paginate(github.rest.actions.listWorkflowRunsForRepo, {
      owner,
      repo,
      head_sha,
      status: "action_required",
      actor: "mlflow-app[bot]",
    });

    // If the number of runs has not changed since the last attempt,
    // we can assume that all the workflow runs have been created.
    if (runs.length > 0 && runs.length === prevRuns.length) {
      return runs;
    }

    prevRuns = runs;
    await new Promise((resolve) => setTimeout(resolve, SLEEP_DURATION_MS));
  }
  return prevRuns;
};

const approveWorkflowRuns = async (context, github, head_sha) => {
  const { owner, repo } = context.repo;
  const workflowRuns = await fetchWorkflowRuns({ context, github, head_sha });
  const approvePromises = workflowRuns.map((run) =>
    github.rest.actions.approveWorkflowRun({
      owner,
      repo,
      run_id: run.id,
    })
  );
  const results = await Promise.allSettled(approvePromises);
  for (const result of results) {
    if (result.status === "rejected") {
      console.error(`Failed to approve run: ${result.reason}`);
    }
  }
};

const checkMaintainerAccess = async (context, github) => {
  const { owner, repo } = context.repo;
  const pull_number = context.issue.number;
  const { runId } = context;
  const pr = await github.rest.pulls.get({ owner, repo, pull_number });

  // Skip maintainer access check for copilot bot PRs
  // Copilot bot creates PRs that are owned by the repository and don't need the same permission model
  if (
    pr.data.user?.type?.toLowerCase() === "bot" &&
    pr.data.user?.login?.toLowerCase() === "copilot"
  ) {
    console.log(`Skipping maintainer access check for copilot bot PR #${pull_number}`);
    return;
  }

  const isForkPR = pr.data.head.repo.full_name !== pr.data.base.repo.full_name;
  if (isForkPR && !pr.data.maintainer_can_modify) {
    const workflowRunUrl = `https://github.com/${owner}/${repo}/actions/runs/${runId}`;

    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: pull_number,
      body: `❌ **Autoformat failed**: The "Allow edits and access to secrets by maintainers" checkbox must be checked for autoformat to work properly.

Please:
1. Check the "Allow edits and access to secrets by maintainers" checkbox on this pull request
2. Comment \`/autoformat\` again

This permission is required for the autoformat bot to push changes to your branch.

**Details:** [View workflow run](${workflowRunUrl})`,
    });

    throw new Error(
      'The "Allow edits and access to secrets by maintainers" checkbox must be checked for autoformat to work properly.'
    );
  }
};

module.exports = {
  shouldAutoformat,
  getPullInfo,
  createReaction,
  createStatus,
  updateStatus,
  approveWorkflowRuns,
  checkMaintainerAccess,
};
