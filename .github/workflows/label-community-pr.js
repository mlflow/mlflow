async function getMaintainers({ github, context }) {
  const collaborators = await github.paginate(github.rest.repos.listCollaborators, {
    owner: context.repo.owner,
    repo: context.repo.repo,
  });
  return collaborators
    .filter(({ role_name }) => ["admin", "maintain"].includes(role_name))
    .map(({ login }) => login)
    .sort();
}

module.exports = async ({ context, github }) => {
  const { owner, repo } = context.repo;
  const { number: issue_number } = context.issue;
  const author = context.payload.pull_request.user.login;

  const maintainers = await getMaintainers({ github, context });

  if (!maintainers.includes(author)) {
    await github.rest.issues.addLabels({
      owner,
      repo,
      issue_number,
      labels: ["community-pr"],
    });

    console.log(`Added 'community-pr' label to PR #${issue_number}`);
  }
};
