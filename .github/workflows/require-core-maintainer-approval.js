const CORE_MAINTAINERS = new Set([
  "B-Step62",
  "BenWilson2",
  "daniellok-db",
  "dbczumar",
  "gabrielfu",
  "harupy",
  "serena-ruan",
  "TomeHirata",
  "WeichenXu123",
  "xq-yin",
]);

module.exports = async ({ github, context, core }) => {
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
