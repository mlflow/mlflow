module.exports = async ({ github, context, core }) => {
  // Fetch all collaborators with admin or maintain role
  const collaborators = await github.paginate(github.rest.repos.listCollaborators, {
    owner: context.repo.owner,
    repo: context.repo.repo,
  });

  // Filter to keep only admin and maintain roles
  const coreMaintainers = collaborators
    .filter(({ role_name }) => role_name === "admin" || role_name === "maintain")
    .map(({ login }) => login);

  console.log("Core maintainers (admin or maintain role):", coreMaintainers);

  const CORE_MAINTAINERS = new Set(coreMaintainers);

  const reviews = await github.paginate(github.rest.pulls.listReviews, {
    owner: context.repo.owner,
    repo: context.repo.repo,
    pull_number: context.issue.number,
  });
  const maintainerApproved = reviews.some(
    ({ state, user: { login } }) => state === "APPROVED" && CORE_MAINTAINERS.has(login)
  );
  if (!maintainerApproved) {
    const maintainerList = Array.from(CORE_MAINTAINERS)
      .map((maintainer) => `${maintainer}`)
      .join(", ");
    const message = `This PR requires an approval from at least one of core maintainers: ${maintainerList}.`;
    core.setFailed(message);
  }
};
