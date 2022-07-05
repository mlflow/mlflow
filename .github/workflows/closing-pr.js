// https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue
const CLOSING_SYNTAX_REGEX = /(?:close|fixe?|resolve)[sd]?\s+#(\d+)/i;
const HAS_CLOSING_PR_LABEL = 'has-closing-pr';

module.exports = async ({ context, github }) => {
  const { body } = context.payload.pull_request;
  // Find a closing syntax after removing comments
  const match = body.replace(/<!--(.+?)-->/gs, '').match(CLOSING_SYNTAX_REGEX);
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
