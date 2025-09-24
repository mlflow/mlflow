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
    pull_number,
    head_sha,
    author_association: pr.data.author_association,
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

const isAuthorAllowed = ({ author_association }) => {
  return ["owner", "member", "collaborator"].includes(author_association.toLowerCase());
};

const validateAuthorPermissions = async (context, github, core, triggerCommentId) => {
  const { comment } = context.payload;

  if (
    !isAuthorAllowed({
      author_association: comment.author_association,
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

    // Update with success message
    const successMessage = `✅ **Sync completed**: The PR branch has been successfully updated with the latest changes from the base branch.`;
    await updateTriggerComment(context, github, triggerCommentId, successMessage);
  } catch (error) {
    console.error("Sync failed:", error.message);

    const message = `❌ **Sync failed**: Unable to update PR branch.

Error: \`${error.message}\``;

    await updateTriggerComment(context, github, triggerCommentId, message);
    throw error;
  }
};

module.exports = {
  shouldSync,
  getPullInfo,
  updateTriggerComment,
  validateAuthorPermissions,
  performSync,
};
