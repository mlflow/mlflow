const { execSync } = require("child_process");

const shouldSync = (comment) => {
  return comment.body.trim() === "/sync";
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
    maintainer_can_modify: pr.data.maintainer_can_modify,
    mergeable: pr.data.mergeable,
    mergeable_state: pr.data.mergeable_state,
  };
};

const updateTriggerComment = async (context, github, commentId, message) => {
  const { owner, repo } = context.repo;
  const { runId } = context;
  const workflowRunUrl = `https://github.com/${owner}/${repo}/actions/runs/${runId}`;

  const updatedBody = `/sync

---

${message}

**Details:** [View workflow run](${workflowRunUrl})`;

  await github.rest.issues.updateComment({
    owner,
    repo,
    comment_id: commentId,
    body: updatedBody,
  });
};

const createInitialReaction = async (context, github) => {
  const { owner, repo } = context.repo;
  const { id: comment_id } = context.payload.comment;
  const { runId } = context;

  // Add rocket reaction
  await github.rest.reactions.createForIssueComment({
    owner,
    repo,
    comment_id,
    content: "rocket",
  });

  // Update the trigger comment with workflow link
  const workflowRunUrl = `https://github.com/${owner}/${repo}/actions/runs/${runId}`;
  const updatedBody = `/sync

---

⏳ [Sync workflow started](${workflowRunUrl})`;

  await github.rest.issues.updateComment({
    owner,
    repo,
    comment_id,
    body: updatedBody,
  });
};

const isAuthorAllowed = ({ author_association, user }) => {
  return (
    ["owner", "member", "collaborator"].includes(author_association.toLowerCase()) ||
    // Allow Copilot and mlflow-app bot to run this workflow
    (user &&
      user.type.toLowerCase() === "bot" &&
      ["copilot", "mlflow-app[bot]"].includes(user.login.toLowerCase()))
  );
};

const validateAuthorPermissions = async (context, github, core, triggerCommentId) => {
  const { comment } = context.payload;

  if (
    !isAuthorAllowed({
      author_association: comment.author_association,
      user: comment.user,
    })
  ) {
    const message = `❌ **Sync failed**: Only repository owners, members, or collaborators can use the /sync command. @${comment.user.login} (${comment.author_association}) does not have sufficient permissions.`;

    await updateTriggerComment(context, github, triggerCommentId, message);
    core.setFailed(`User ${comment.user.login} does not have sufficient permissions`);
  }
};

const validatePRConditions = async (context, github, pullInfo, triggerCommentId) => {
  // Check if it's a fork PR
  const isForkPR = pullInfo.repository !== pullInfo.base_repo;

  // For fork PRs, check if maintainers can modify
  if (isForkPR && !pullInfo.maintainer_can_modify) {
    const message = `❌ **Sync failed**: For fork PRs, the "Allow edits and access to secrets by maintainers" checkbox must be checked.

Please:
1. Check the "Allow edits and access to secrets by maintainers" checkbox on this pull request
2. Comment \`/sync\` again`;

    await updateTriggerComment(context, github, triggerCommentId, message);
    throw new Error("Fork PR does not allow maintainer edits");
  }

  // Check for merge conflicts
  if (pullInfo.mergeable === false) {
    const message = `❌ **Sync failed**: This PR has merge conflicts that must be resolved before syncing.

Please resolve the conflicts and then comment \`/sync\` again.`;

    await updateTriggerComment(context, github, triggerCommentId, message);
    throw new Error("PR has merge conflicts");
  }

  // Check if mergeable state is unknown (GitHub is still computing)
  if (pullInfo.mergeable === null) {
    // Wait a bit and check again
    await new Promise((resolve) => setTimeout(resolve, 5000));

    const { owner, repo } = context.repo;
    const updatedPR = await github.rest.pulls.get({
      owner,
      repo,
      pull_number: pullInfo.pull_number,
    });

    if (updatedPR.data.mergeable === false) {
      const message = `❌ **Sync failed**: This PR has merge conflicts that must be resolved before syncing.

Please resolve the conflicts and then comment \`/sync\` again.`;

      await updateTriggerComment(context, github, triggerCommentId, message);
      throw new Error("PR has merge conflicts");
    }
  }
};

const generateRandomRemoteName = (prefix) => {
  const randomSuffix = Math.random().toString(36).substring(2, 8);
  return `${prefix}-${randomSuffix}`;
};

const performSync = async (context, github, pullInfo, botToken, triggerCommentId) => {
  try {
    // Configure git
    execSync('git config user.name "mlflow-app[bot]"');
    execSync('git config user.email "mlflow-app[bot]@users.noreply.github.com"');

    // We're already on the PR head branch, just need to fetch and merge base
    const baseRepoUrl = `https://github.com/${pullInfo.base_repo}.git`;

    console.log(`Setting up base remote and fetching base branch`);

    // Use random remote name to avoid conflicts
    const baseRemoteName = generateRandomRemoteName("base");

    execSync(`git remote add ${baseRemoteName} ${baseRepoUrl}`);
    execSync(`git fetch ${baseRemoteName} ${pullInfo.base_ref}`);

    console.log(`Merging base branch: ${pullInfo.base_ref}`);
    execSync(`git merge ${baseRemoteName}/${pullInfo.base_ref} --no-edit`);

    console.log(`Pushing updated branch`);
    execSync(`git push origin HEAD`);

    console.log("Sync completed successfully");

    // Clean up the remote
    execSync(`git remote remove ${baseRemoteName}`);
  } catch (error) {
    console.error("Sync failed:", error.message);

    const message = `❌ **Sync failed**: An error occurred during the sync process.

Error: \`${error.message}\``;

    await updateTriggerComment(context, github, triggerCommentId, message);
    throw error;
  }
};

const updateFinalStatus = async (context, github, success, triggerCommentId) => {
  if (success) {
    const message = `✅ **Sync completed**: The PR branch has been successfully updated with the latest changes from the base branch.`;
    await updateTriggerComment(context, github, triggerCommentId, message);
  }
  // For failures, the error message should already be updated in performSync
};

module.exports = {
  shouldSync,
  getPullInfo,
  updateTriggerComment,
  createInitialReaction,
  validateAuthorPermissions,
  validatePRConditions,
  performSync,
  updateFinalStatus,
};
