const { getCloseReason } = require("./auto-close-pr.js");

module.exports = async ({ context, github }) => {
  const closeReason = await getCloseReason({ github, context });
  if (closeReason) {
    console.log("PR will be auto-closed. Skipping labeling.");
    return;
  }

  const { owner, repo } = context.repo;
  const number = context.payload.pull_request.number;

  const result = await github.graphql(
    `query($owner: String!, $repo: String!, $number: Int!) {
      repository(owner: $owner, name: $repo) {
        pullRequest(number: $number) {
          closingIssuesReferences(first: 10) {
            nodes {
              number
            }
          }
        }
      }
    }`,
    { owner, repo, number }
  );

  const issues = result.repository.pullRequest.closingIssuesReferences.nodes;
  for (const { number: issue_number } of issues) {
    await github.rest.issues.addLabels({
      owner,
      repo,
      issue_number,
      labels: ["has-closing-pr"],
    });
  }
};
