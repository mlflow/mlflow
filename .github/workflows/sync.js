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
  return ["owner", "member", "collaborator"].includes(author_association.toLowerCase());
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

const performSync = async (context, github, pullInfo, botToken, triggerCommentId) => {
  const { owner, repo } = context.repo;

  try {
    console.log(`Attempting to update PR branch using GitHub API`);

    await github.rest.pulls.updateBranch({
      owner,
      repo,
      pull_number: pullInfo.pull_number,
      expected_head_sha: pullInfo.head_sha,
    });

    console.log("Sync completed successfully");
  } catch (error) {
    console.error("Sync failed:", error.message);

    let message = `❌ **Sync failed**: Unable to update PR branch.`;

    // Provide specific error messages based on common failure scenarios
    if (error.message.includes("merge conflict") || error.message.includes("conflict")) {
      message += `\n\nThis PR has merge conflicts that must be resolved before syncing. Please resolve the conflicts and try again.`;
    } else if (error.message.includes("not fast-forward") || error.message.includes("behind")) {
      message += `\n\nThe PR branch cannot be fast-forwarded. This typically happens when there are conflicting changes.`;
    } else if (error.message.includes("expected_head_sha")) {
      message += `\n\nThe PR branch has been updated since the sync started. Please try the \`/sync\` command again.`;
    } else {
      message += `\n\nError: \`${error.message}\``;
    }

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
  performSync,
  updateFinalStatus,
};
