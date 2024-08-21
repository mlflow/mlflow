const CORE_MAINTAINERS = new Set([
  "B-Step62",
  "BenWilson2",
  "daniellok-db",
  "dbczumar",
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
    const maintainerList = CORE_MAINTAINERS.join(", ");
    core.setFailed(
      `This PR requires an approval from at leas one of core maintainers [${maintainerList}].`
    );
  }
};
