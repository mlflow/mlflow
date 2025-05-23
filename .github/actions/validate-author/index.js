function isAllowed({ author_association, user }) {
  return (
    ["owner", "member", "collaborator"].includes(author_association.toLowerCase()) ||
    // Allow Copilot to run this workflow
    (user && user.login.toLowerCase() === "copilot" && user.type.toLowerCase() === "bot")
  );
}

/**
 * Create a comment on the issue/PR with the failure message
 * @param {Object} github - The github object from the action context
 * @param {Object} context - The context object from the action
 * @param {string} message - The message to add as a comment
 */
async function createFailureComment(github, context, message) {
  try {
    await github.rest.issues.createComment({
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: context.issue.number,
      body: `❌ **Validation Failed**: ${message}`
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
      const message = `${comment.author_association} is not allowed to use this workflow.`;
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
      const message = `This workflow is not allowed to run on PRs from ${pullRequest.author_association}.`;
      await createFailureComment(github, context, message);
      core.setFailed(message);
    }
  } else {
    const message = `Unsupported event: ${context.eventName}`;
    await createFailureComment(github, context, message);
    core.setFailed(message);
  }
};
