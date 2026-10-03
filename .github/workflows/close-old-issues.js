const MS_PER_DAY = 24 * 60 * 60 * 1000;
const CUTOFF_DAYS = 180;
const EXCLUDED_LABELS = new Set(["security"]);
const MAX_ISSUES_PER_RUN = 100;

const QUERY = `
  query($searchQuery: String!) {
    rateLimit { remaining resetAt }
    search(query: $searchQuery, type: ISSUE, first: ${MAX_ISSUES_PER_RUN}) {
      nodes {
        ... on Issue {
          number
          createdAt
          url
          labels(first: 100) { nodes { name } }
          reactions { totalCount }
          timelineItems(itemTypes: [CROSS_REFERENCED_EVENT], first: 100) {
            nodes {
              ... on CrossReferencedEvent {
                willCloseTarget
                source {
                  __typename
                  ... on PullRequest {
                    number
                    state
                    url
                  }
                }
              }
            }
          }
        }
      }
    }
  }
`;

function getLabels(issue) {
  return issue.labels?.nodes?.map((label) => label.name) ?? [];
}

function getClosingPullRequests(issue) {
  return (issue.timelineItems?.nodes ?? [])
    .filter(
      (event) =>
        event.willCloseTarget &&
        event.source?.__typename === "PullRequest" &&
        event.source?.state === "OPEN"
    )
    .map((event) => event.source);
}

function isOlderThanCutoff(issue, cutoffTime) {
  return new Date(issue.createdAt).getTime() < cutoffTime;
}

module.exports = async ({ context, github }) => {
  const { owner, repo } = context.repo;
  const dryRun = process.env.DRY_RUN !== "false";
  const closeMessage = process.env.CLOSE_MESSAGE?.trim();

  if (!closeMessage) {
    throw new Error("CLOSE_MESSAGE is required.");
  }
  if (!dryRun && closeMessage.startsWith("TODO:")) {
    throw new Error("Replace the CLOSE_MESSAGE placeholder before closing issues in production.");
  }

  const cutoffTime = Date.now() - CUTOFF_DAYS * MS_PER_DAY;
  const cutoffDate = new Date(cutoffTime).toISOString().slice(0, 10);
  const searchQuery = `repo:${owner}/${repo} is:issue is:open created:<${cutoffDate}`;
  const response = await github.graphql(QUERY, { searchQuery });
  const { remaining, resetAt } = response.rateLimit;
  console.log(`Rate limit: ${remaining} remaining, resets at ${resetAt}`);

  const oldIssues = response.search.nodes
    .filter((issue) => isOlderThanCutoff(issue, cutoffTime))
    .filter((issue) => !getLabels(issue).some((label) => EXCLUDED_LABELS.has(label)))
    .filter((issue) => issue.reactions.totalCount === 0);

  console.log(
    `Found ${oldIssues.length} open issues older than ${CUTOFF_DAYS} days without reactions.`
  );

  for (const issue of oldIssues) {
    const pullRequests = getClosingPullRequests(issue);

    for (const pullRequest of pullRequests) {
      if (dryRun) {
        console.log(`[dry run] Would close PR #${pullRequest.number} for issue #${issue.number}`);
        continue;
      }

      await github.rest.pulls.update({
        owner,
        repo,
        pull_number: pullRequest.number,
        state: "closed",
      });
      await github.rest.issues.createComment({
        owner,
        repo,
        issue_number: pullRequest.number,
        body: `${closeMessage}\n\nLinked issue: ${issue.url}`,
      });
      console.log(`Closed PR #${pullRequest.number} linked to issue #${issue.number}.`);
    }

    if (dryRun) {
      console.log(`[dry run] Would close issue #${issue.number}`);
      continue;
    }

    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: issue.number,
      body: closeMessage,
    });
    await github.rest.issues.update({
      owner,
      repo,
      issue_number: issue.number,
      state: "closed",
      state_reason: "not_planned",
    });
    console.log(`Closed issue #${issue.number}.`);
  }

  console.log(
    dryRun
      ? "Dry run completed without closing issues or pull requests."
      : `Closed ${oldIssues.length} old issues and their linked pull requests.`
  );
};
