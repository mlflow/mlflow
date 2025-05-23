function isAllowed({ author_association, user }) {
  return (
    ["owner", "member", "collaborator"].includes(author_association.toLowerCase()) ||
    // Allow Copilot to run this workflow
    (user && user.login.toLowerCase() === "copilot" && user.type.toLowerCase() === "bot")
  );
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
      core.setFailed(`${comment.author_association} is not allowed to use this workflow.`);
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
      core.setFailed(
        `This workflow is not allowed to run on PRs from ${pullRequest.author_association}.`
      );
    }
  } else {
    core.setFailed(`Unsupported event: ${context.eventName}`);
  }
};
