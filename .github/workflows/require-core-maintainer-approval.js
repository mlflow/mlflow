async function getMaintainers({ github, context }) {
  const collaborators = await github.paginate(github.rest.repos.listCollaborators, {
    owner: context.repo.owner,
    repo: context.repo.repo,
  });
  return collaborators
    .filter(({ role_name }) => ["admin", "maintain"].includes(role_name))
    .map(({ login }) => login);
}

module.exports = async ({ github, context, core }) => {
  const maintainers = await getMaintainers({ github, context });
  const reviews = await github.paginate(github.rest.pulls.listReviews, {
    owner: context.repo.owner,
    repo: context.repo.repo,
    pull_number: context.issue.number,
  });
  const maintainerApproved = reviews.some(
    ({ state, user: { login } }) => state === "APPROVED" && maintainers.includes(login)
  );
  if (!maintainerApproved) {
    const message = `This PR requires an approval from at least one of the core maintainers: ${maintainers.join(
      ", "
    )}.`;
    core.setFailed(message);
  }
};
