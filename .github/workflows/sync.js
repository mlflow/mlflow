const shouldSync = (comment) => {
  return comment.body.trim() === "/sync";
};

const getPullInfo = async (context, github) => {
  const { owner, repo } = context.repo;
  const pull_number = context.issue.number;
  const pr = await github.rest.pulls.get({ owner, repo, pull_number });
  const { sha: head_sha } = pr.data.head;
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

const performSync = async (context, github, pullInfo, botToken) => {
  const { owner, repo } = context.repo;
  const { comment } = context.payload;

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
    await updateTriggerComment(context, github, comment.id, successMessage);
  } catch (error) {
    console.error("Sync failed:", error.message);

    const message = `❌ **Sync failed**: Unable to update PR branch.

Error: \`${error.message}\``;

    await updateTriggerComment(context, github, comment.id, message);
    throw error;
  }
};

module.exports = {
  shouldSync,
  getPullInfo,
  updateTriggerComment,
  isAuthorAllowed,
  performSync,
};
