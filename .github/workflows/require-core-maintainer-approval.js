const CORE_MAINTAINERS = new Set([
  "B-Step62",
  "BenWilson2",
  "daniellok-db",
  "dbczumar",
  "gabrielfu",
  "harupy",
  "serena-ruan",
  "WeichenXu123",
  "xq-yin",
]);

module.exports = async ({ github, context, core }) => {
  const { data: reviews } = await github.rest.pulls.listReviews({
    owner: context.repo.owner,
    repo: context.repo.repo,
    pull_number: context.issue.number,
  });
  const maintainerApproved = reviews.some(
    ({ state, user: { login } }) => state === "APPROVED" && CORE_MAINTAINERS.has(login)
  );
  if (!maintainerApproved) {
    const marker = "<!-- MAINTAINER_APPROVAL -->";
    const comments = await github.paginate(github.rest.issues.listComments, {
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: context.issue.number,
    });
    const maintainers = Array.from(CORE_MAINTAINERS)
      .map((maintainer) => `\`${maintainer}\``)
      .join(", ");
    const message = `This PR requires approval from at least one core maintainer (${maintainers}). If you're not sure who to request a review from, assign \`mlflow-automation\`.`;
    if (!comments.some(({ body }) => body.includes(marker))) {
      await github.rest.issues.createComment({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: context.issue.number,
        body: `${message}\n\n${marker}`,
      });
    }
    core.setFailed(message);
  }
};
