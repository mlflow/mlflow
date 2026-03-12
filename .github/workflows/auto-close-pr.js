// Auto-close PRs that attempt to close an issue without the "ready" label.
// Only targets PRs with closing keywords (Closes/Fixes/Resolves) because
// "Relates to #123" PRs don't claim to solve the issue and shouldn't be blocked.
// Skips PRs that reference multiple issues (ambiguous intent).
// Only enforces on issues created on or after 2026-03-10.

const READY_LABEL = "ready";
// The date we introduced the "ready" label policy; skip older issues.
const CUTOFF_DATE = new Date("2026-03-10T00:00:00Z");

const QUERY = `
  query($owner: String!, $repo: String!, $number: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $number) {
        closingIssuesReferences(first: 10) {
          nodes {
            number
            createdAt
            labels(first: 50) {
              nodes { name }
            }
          }
        }
      }
    }
  }
`;

module.exports = async ({ context, github }) => {
  const prNumber = context.payload.pull_request.number;
  const { owner, repo } = context.repo;

  const response = await github.graphql(QUERY, { owner, repo, number: prNumber });
  const issues = response.repository.pullRequest.closingIssuesReferences.nodes;

  if (issues.length === 0) {
    console.log("No closing issue references found. Skipping.");
    return;
  }

  if (issues.length > 1) {
    console.log(
      `Multiple issues referenced (${issues.map((i) => `#${i.number}`).join(", ")}). Skipping.`
    );
    return;
  }

  const issue = issues[0];
  console.log(`PR #${prNumber} references issue #${issue.number}`);

  // Skip issues created before the cutoff date
  if (new Date(issue.createdAt) < CUTOFF_DATE) {
    console.log(
      `Issue #${issue.number} was created before ${CUTOFF_DATE.toISOString()}. Skipping.`
    );
    return;
  }

  const hasReadyLabel = issue.labels.nodes.some((label) => label.name === READY_LABEL);
  if (hasReadyLabel) {
    console.log(`Issue #${issue.number} has the "${READY_LABEL}" label. No action needed.`);
    return;
  }

  console.log(
    `Issue #${issue.number} is missing the "${READY_LABEL}" label. Closing PR #${prNumber}.`
  );

  await github.rest.issues.createComment({
    owner,
    repo,
    issue_number: prNumber,
    body: [
      `This PR was automatically closed because #${issue.number} is missing the \`${READY_LABEL}\` label.`,
      "Once a maintainer triages the issue and applies the label, feel free to reopen this PR.",
      "Please do not force-push to or delete the PR branch so this PR can be reopened.",
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
