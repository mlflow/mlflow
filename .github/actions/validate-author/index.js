function isAllowed(author_association) {
  return ["owner", "member", "collaborator"].includes(author_association.toLowerCase());
}

module.exports = async ({ context, github, core }) => {
  if (context.eventName === "issue_comment") {
    if (!isAllowed(context.payload.comment.author_association)) {
      core.setFailed(`${comment.author_association} is not allowed to use this workflow.`);
    }

    const { data: pullRequest } = await github.rest.pulls.get({
      owner: context.repo.owner,
      repo: context.repo.repo,
      pull_number: context.issue.number,
    });
    if (!isAllowed(pullRequest.author_association)) {
      core.setFailed(
        `This workflow is not allowed to run on PRs from ${pullRequest.author_association}.`
      );
    }
  } else {
    core.setFailed(`Unsupported event: ${context.eventName}`);
  }
};
