// Auto-close PRs that attempt to close an issue without the "ready" label.
// Skips PRs that reference multiple issues (ambiguous intent).
// Only enforces on issues created on or after 2026-03-10.

const CLOSING_SYNTAX_PATTERNS = [
  /(?:(?:close|fixe|resolve)[sd]?|fix)\s+(?:mlflow\/mlflow)?#(\d+)/gi,
  /(?:(?:close|fixe|resolve)[sd]?|fix)\s+(?:https?:\/\/github.com\/mlflow\/mlflow\/issues\/)(\d+)/gi,
  // Bare references (contributors might omit the closing keyword)
  /(?:mlflow\/mlflow)?#(\d+)/gi,
  /https?:\/\/github.com\/mlflow\/mlflow\/issues\/(\d+)/gi,
];
const READY_LABEL = "ready";
// The date we introduced the "ready" label policy; skip older issues.
const CUTOFF_DATE = new Date("2026-03-10T00:00:00Z");

const getIssuesToClose = (body) => {
  // Strip HTML comments and fenced code blocks to avoid false matches
  const commentsExcluded = body.replace(/<!--(.+?)-->/gs, "").replace(/```[\s\S]*?```/g, "");
  const matches = CLOSING_SYNTAX_PATTERNS.flatMap((pattern) =>
    Array.from(commentsExcluded.matchAll(pattern))
  );
  const issueNumbers = matches.map((match) => match[1]);
  return [...new Set(issueNumbers)].sort();
};

module.exports = async ({ context, github }) => {
  const { body } = context.payload.pull_request;
  const prNumber = context.payload.pull_request.number;
  const { owner, repo } = context.repo;

  const issues = getIssuesToClose(body || "");

  if (issues.length === 0) {
    console.log("No closing issue references found. Skipping.");
    return;
  }

  if (issues.length > 1) {
    console.log(`Multiple issues referenced (${issues.join(", ")}). Skipping.`);
    return;
  }

  const issueNumber = issues[0];
  console.log(`PR #${prNumber} references issue #${issueNumber}`);

  const { data: issue } = await github.rest.issues.get({
    owner,
    repo,
    issue_number: issueNumber,
  });

  // Skip if it's actually a PR
  if (issue.pull_request) {
    console.log(`#${issueNumber} is a pull request, not an issue. Skipping.`);
    return;
  }

  // Skip issues created before the cutoff date
  if (new Date(issue.created_at) < CUTOFF_DATE) {
    console.log(`Issue #${issueNumber} was created before ${CUTOFF_DATE.toISOString()}. Skipping.`);
    return;
  }

  const hasReadyLabel = issue.labels.some((label) => label.name === READY_LABEL);
  if (hasReadyLabel) {
    console.log(`Issue #${issueNumber} has the "${READY_LABEL}" label. No action needed.`);
    return;
  }

  console.log(
    `Issue #${issueNumber} is missing the "${READY_LABEL}" label. Closing PR #${prNumber}.`
  );

  await github.rest.issues.createComment({
    owner,
    repo,
    issue_number: prNumber,
    body: [
      `This PR was automatically closed because the referenced issue #${issueNumber} hasn't been triaged yet (missing the \`ready\` label).`,
      "This doesn't mean your contribution won't be merged. Once a maintainer reviews the issue and applies the `ready` label, feel free to reopen this PR.",
      "Please do not force-push to or delete this branch, as that will make this PR unable to be reopened.",
    ].join(" "),
  });

  await github.rest.pulls.update({
    owner,
    repo,
    pull_number: prNumber,
    state: "closed",
  });

  console.log(`PR #${prNumber} closed.`);
};
