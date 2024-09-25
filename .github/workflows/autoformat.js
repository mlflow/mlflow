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

const isNewCommand = (comment) => {
  return comment.body.trim() === "/autoformat";
};

const isOldCommand = (comment) => {
  return /^@mlflow-automation\s+autoformat$/.test(comment.body.trim());
};

const shouldAutoformat = (comment) => {
  return isNewCommand(comment) || isOldCommand(comment);
};

const getPullInformation = async (context, github) => {
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

  if (isOldCommand(context.payload.comment)) {
    await github.rest.issues.createComment({
      repo: context.repo.repo,
      owner: context.repo.owner,
      issue_number: context.issue.number,
      body: "The command `@mlflow-automation autoformat` has been deprecated and will be removed soon. Please use `/autoformat` instead.",
    });
  }
};

const createStatus = async (context, github, core) => {
  const { head_sha, head_ref, repository } = await getPullInformation(context, github);
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

const isMlflowMaintainer = (commentAuthorAssociation) => {
  return ["OWNER", "MEMBER", "COLLABORATOR"].includes(commentAuthorAssociation);
};

module.exports = {
  isMlflowMaintainer,
  shouldAutoformat,
  getPullInformation,
  createReaction,
  createStatus,
  updateStatus,
};
