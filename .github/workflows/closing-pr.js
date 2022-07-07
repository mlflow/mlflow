// Regular expressions to capture a closing syntax in the PR body
// https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue
const CLOSING_SYNTAX_PATTERNS = [
  /(?:close|fixe?|resolve)[sd]?\s+(?:mlflow\/mlflow)?#(\d+)/gi,
  /(?:close|fixe?|resolve)[sd]?\s+(?:https?:\/\/github.com\/mlflow\/mlflow\/issues\/)(\d+)/gi,
];
const HAS_CLOSING_PR_LABEL = 'has-closing-pr';

const getIssuesToClose = (body) => {
  const commentsExcluded = body.replace(/<!--(.+?)-->/gs, ''); // remove comments
  const matches = CLOSING_SYNTAX_PATTERNS.flatMap((pattern) =>
    Array.from(commentsExcluded.matchAll(pattern)),
  );
  const issueNumbers = matches.map((match) => match[1]);
  return [...new Set(issueNumbers)].sort();
};

const arraysEqual = (a1, a2) => {
  return JSON.stringify(a1) == JSON.stringify(a2);
};

const assertArrayEqual = (a1, a2) => {
  if (!arraysEqual(a1, a2)) {
    throw `[${a1}] !== [${a2}]`;
  }
};

const test = () => {
  const body1 = 'Close #123';
  assertArrayEqual(getIssuesToClose(body1), ['123']);

  const body2 = `
Fix mlflow/mlflow#123
Resolve https://github.com/mlflow/mlflow/issues/456
`;
  assertArrayEqual(getIssuesToClose(body2), ['123', '456']);

  const body3 = `
Fix #123
Close #123
`;
  assertArrayEqual(getIssuesToClose(body3), ['123']);

  const body4 = 'Relates to #123';
  assertArrayEqual(getIssuesToClose(body4), []);

  const body5 = '<!-- close #123 -->';
  assertArrayEqual(getIssuesToClose(body5), []);
};

// `node .github/workflows/closing-pr.js` runs this block
if (require.main === module) {
  test();
}

module.exports = async ({ context, github }) => {
  const { body } = context.payload.pull_request;
  getIssuesToClose(body).forEach(async (issue_number) => {
    await github.issues.addLabels({
      owner,
      repo,
      issue_number,
      labels: [HAS_CLOSING_PR_LABEL],
    });
  });
};
