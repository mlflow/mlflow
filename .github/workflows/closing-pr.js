// Regular expressions to capture a closing syntax in the PR body
// https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue
const CLOSING_SYNTAX_PATTERNS = [
  /(?:close|fixe?|resolve)[sd]?\s+(?:mlflow\/mlflow)?#(\d+)/i,
  /(?:close|fixe?|resolve)[sd]?\s+(?:https?:\/\/github.com\/mlflow\/mlflow\/issues\/)(\d+)/i,
];
const HAS_CLOSING_PR_LABEL = 'has-closing-pr';

module.exports = async ({ context, github }) => {
  const { body } = context.payload.pull_request;
  // Find a closing syntax after removing comments
  const commentsExcluded = body.replace(/<!--(.+?)-->/gs, '');
  const match = CLOSING_SYNTAX_PATTERNS.map((pattern) => pattern.match(commentsExcluded)).find(
    (match) => match,
  );
  if (match) {
    const issue_number = match[1];
    const { owner, repo } = context.repo;
    await github.issues.addLabels({
      owner,
      repo,
      issue_number,
      labels: [HAS_CLOSING_PR_LABEL],
    });
  }
};
