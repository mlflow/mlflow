const CORE_MAINTAINERS = new Set([
  "B-Step62",
  "BenWilson2",
  "daniellok-db",
  "harupy",
  "serena-ruan",
  "WeichenXu123",
]);

module.exports = async ({ github, context, core }) => {
  const { data: reviews } = await github.rest.pulls.listReviews({
    owner: context.repo.owner,
    repo: context.repo.repo,
    pull_number: context.issue.number,
  });

  const maintainerApprovals = reviews.filter(
    (review) => review.state === "APPROVED" && CORE_MAINTAINERS.has(review.user.login)
  );

  if (maintainerApprovals.length === 0) {
    core.setFailed("No core maintainers have approved this pull request");
  }
};
