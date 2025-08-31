function isAllowed({ author_association, user }) {
  return (
    ["owner", "member", "collaborator"].includes(author_association.toLowerCase()) ||
    // Allow Copilot and mlflow-app bot to run this workflow
    (user &&
      user.type.toLowerCase() === "bot" &&
      ["copilot", "mlflow-app[bot]"].includes(user.login.toLowerCase()))
  );
}

/**
 * Create a comment on the issue/PR with the failure message
 * @param {Object} github - The github object from the action context
 * @param {Object} context - The context object from the action
 * @param {string} message - The message to add as a comment
 * @returns {Promise<void>}
 */
async function createFailureComment(github, context, message) {
  try {
    await github.rest.issues.createComment({
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: context.issue.number,
      body: `❌ **Validation Failed**: ${message}`,
    });
  } catch (error) {
    // Log the error but don't fail the workflow because of comment creation issues
    console.error(`Error creating comment: ${error.message}`);
  }
}

module.exports = async ({ context, github, core }) => {
  if (context.eventName === "issue_comment") {
    const { comment } = context.payload;
    if (
      !isAllowed({
        author_association: comment.author_association,
        user: comment.user,
      })
    ) {
      const message = `This workflow can only be triggered by a repository owner, member, or collaborator. @${comment.user.login} (${comment.author_association}) does not have sufficient permissions.`;
      await createFailureComment(github, context, message);
      core.setFailed(message);
    }

    const { data: pullRequest } = await github.rest.pulls.get({
      owner: context.repo.owner,
      repo: context.repo.repo,
      pull_number: context.issue.number,
    });
    if (
      !isAllowed({
        author_association: pullRequest.author_association,
        user: pullRequest.user,
      })
    ) {
      const message = `This workflow can only be triggered on PRs filed by a repository owner, member, or collaborator. @${pullRequest.user.login} (${pullRequest.author_association}) does not have sufficient permissions.`;
      await createFailureComment(github, context, message);
      core.setFailed(message);
    }
  } else {
    const message = `This workflow does not support the "${context.eventName}" event type.`;
    await createFailureComment(github, context, message);
    core.setFailed(message);
  }
};
