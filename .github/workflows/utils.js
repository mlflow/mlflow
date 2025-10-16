/**
 * Appends a message to an existing GitHub comment.
 *
 * @param {object} context - GitHub Actions context object
 * @param {object} github - GitHub REST API client
 * @param {number} commentId - Comment ID to update
 * @param {string} message - Message to append
 */
async function appendToComment(context, github, commentId, message) {
  const { owner, repo } = context.repo;

  // Refetch the comment to get the latest body
  const { data: comment } = await github.rest.issues.getComment({
    owner,
    repo,
    comment_id: commentId,
  });

  const updatedBody = `${comment.body}\n\n---\n${message}`;

  await github.rest.issues.updateComment({
    owner,
    repo,
    comment_id: commentId,
    body: updatedBody,
  });
}

module.exports = { appendToComment };
