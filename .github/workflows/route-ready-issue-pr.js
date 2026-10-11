const { getCloseReason } = require("./auto-close-pr.js");
const runTeamReview = require("./team-review.js");

const READY_LABEL = "ready";
const TEAM_REVIEW_LABEL = "team-review";

const QUERY = `
  query($owner: String!, $repo: String!, $number: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $number) {
        closingIssuesReferences(first: 10) {
          nodes {
            number
            state
            labels(first: 50) {
              nodes { name }
            }
            assignees(first: 10) {
              nodes { login }
            }
          }
        }
      }
    }
  }
`;

function getPrLabels(pr) {
  return (pr.labels ?? []).map((label) => (typeof label === "string" ? label : label.name));
}

async function main({ context, github }) {
  const pr = context.payload.pull_request;
  if (pr.user.type === "Bot") return;

  if (await getCloseReason({ github, context })) {
    console.log(`PR #${pr.number} does not satisfy the auto-close policy. Skipping routing.`);
    return;
  }

  const { owner, repo } = context.repo;
  const result = await github.graphql(QUERY, { owner, repo, number: pr.number });
  const issues = result.repository.pullRequest.closingIssuesReferences.nodes;
  if (issues.length !== 1) {
    console.log(`PR #${pr.number} closes ${issues.length} issues. Skipping routing.`);
    return;
  }

  const issue = issues[0];
  const labels = issue.labels.nodes.map((label) => label.name);
  if (issue.state !== "OPEN" || !labels.includes(READY_LABEL)) {
    console.log(`Issue #${issue.number} is not an open ready issue. Skipping routing.`);
    return;
  }

  const prAuthor = pr.user.login;
  const assignees = issue.assignees.nodes.map((assignee) => assignee.login);
  if (assignees.length > 0 && !assignees.includes(prAuthor)) {
    console.log(`Issue #${issue.number} is assigned to another contributor. Skipping routing.`);
    return;
  }

  if (assignees.length === 0) {
    await github.rest.issues.addAssignees({
      owner,
      repo,
      issue_number: issue.number,
      assignees: [prAuthor],
    });
    console.log(`Assigned issue #${issue.number} to @${prAuthor}.`);
  }

  if (pr.draft) {
    console.log(`PR #${pr.number} is a draft. Deferring team review until it is ready.`);
    return;
  }

  if (getPrLabels(pr).includes(TEAM_REVIEW_LABEL)) {
    console.log(`PR #${pr.number} already has the "${TEAM_REVIEW_LABEL}" label.`);
    return;
  }

  await github.rest.issues.addLabels({
    owner,
    repo,
    issue_number: pr.number,
    labels: [TEAM_REVIEW_LABEL],
  });
  console.log(`Added the "${TEAM_REVIEW_LABEL}" label to PR #${pr.number}.`);

  // Labels added with GITHUB_TOKEN do not trigger a follow-up workflow run.
  await runTeamReview({ github, context });
}

module.exports = { main };
