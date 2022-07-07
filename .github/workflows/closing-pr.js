// Regular expressions to capture a closing syntax in the PR body
// https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue
const CLOSING_SYNTAX_PATTERNS = [
  /(?:close|fixe?|resolve)[sd]?\s+(?:mlflow\/mlflow)?#(\d+)/gi,
  /(?:close|fixe?|resolve)[sd]?\s+(?:https?:\/\/github.com\/mlflow\/mlflow\/issues\/)(\d+)/gi,
];
const HAS_CLOSING_PR_LABEL = 'has-closing-pr';

const getIssuesToLabel = (body) => {
  const commentsExcluded = body.replace(/<!--(.+?)-->/gs, ''); // remove comments
  const matches = CLOSING_SYNTAX_PATTERNS.flatMap((pattern) =>
    Array.from(commentsExcluded.matchAll(pattern)),
  );
  const issueNumbers = matches.map((match) => match[1]);
  return [...new Set(issueNumbers)];
};

module.exports = async ({ context, github }) => {
  const { body } = context.payload.pull_request;
  getIssuesToLabel(body).forEach(async (issue_number) => {
    await github.issues.addLabels({
      owner,
      repo,
      issue_number,
      labels: [HAS_CLOSING_PR_LABEL],
    });
  });
};

const main = () => {
  const body1 = 'Close #123';
  console.log(getIssuesToLabel(body1));
  const body2 = `
Fix mlflow/mlflow#456
Resolve https://github.com/mlflow/mlflow/issues/789
`;
  console.log(getIssuesToLabel(body2));
};

if (require.main === module) {
  main();
}
