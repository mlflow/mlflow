// Custom stale PR workflow using GitHub Timeline API
// Detects last human activity (ignoring bot events) and closes inactive PRs

const MS_PER_DAY = 24 * 60 * 60 * 1000;
const STALE_DAYS = 30;
const MAX_CLOSES = 50;
const closeMessage = (days) =>
  `Closing due to inactivity (no activity for ${days} days). Feel free to reopen if still relevant.`;

// GraphQL query to fetch open PRs with timeline data
const QUERY = `
  query($cursor: String) {
    rateLimit { remaining resetAt }
    repository(owner: "mlflow", name: "mlflow") {
      pullRequests(states: OPEN, first: 50, after: $cursor) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          number
          url
          createdAt
          authorAssociation
          closingIssuesReferences(first: 1) {
            totalCount
          }
          timelineItems(
            last: 10
            itemTypes: [ISSUE_COMMENT, PULL_REQUEST_REVIEW, PULL_REQUEST_COMMIT, REOPENED_EVENT]
          ) {
            nodes {
              __typename
              ... on IssueComment {
                createdAt
                author { __typename }
              }
              ... on PullRequestReview {
                createdAt
                author { __typename }
              }
              ... on PullRequestCommit {
                commit {
                  committedDate
                  author { user { __typename } }
                }
              }
              ... on ReopenedEvent {
                createdAt
                actor { __typename }
              }
            }
          }
        }
      }
    }
  }
`;

const isBot = (author) => !author || author.__typename === "Bot";

const getEventDate = (item) => {
  if (item.__typename === "PullRequestCommit") {
    const user = item.commit?.author?.user;
    return user && !isBot(user) ? item.commit.committedDate : null;
  }
  if (item.__typename === "ReopenedEvent") {
    return !isBot(item.actor) ? item.createdAt : null;
  }
  return !isBot(item.author) ? item.createdAt : null;
};

const getLastHumanActivity = (pr) => {
  const items = pr.timelineItems.nodes || [];
  const item = items.findLast((i) => getEventDate(i));
  return new Date(item ? getEventDate(item) : pr.createdAt);
};

const isStale = (lastActivityDate) => {
  return (Date.now() - lastActivityDate) / MS_PER_DAY > STALE_DAYS;
};

const shouldProcessPR = (pr) => {
  // Skip community PRs
  const memberAssociations = ["MEMBER", "OWNER", "COLLABORATOR"];
  if (!memberAssociations.includes(pr.authorAssociation)) {
    return false;
  }

  // Skip PRs that close issues
  if (pr.closingIssuesReferences.totalCount > 0) {
    return false;
  }

  return true;
};

module.exports = async ({ context, github }) => {
  const { owner, repo } = context.repo;
  let closeCount = 0;

  try {
    let cursor = null;
    let hasNextPage = true;

    while (hasNextPage) {
      const response = await github.graphql(QUERY, { cursor });
      const { remaining, resetAt } = response.rateLimit;
      console.log(`Rate limit: ${remaining} remaining, resets at ${resetAt}`);

      const { nodes, pageInfo } = response.repository.pullRequests;
      hasNextPage = pageInfo.hasNextPage;
      cursor = pageInfo.endCursor;

      for (const pr of nodes) {
        if (closeCount >= MAX_CLOSES) {
          console.log(`Reached close limit (${MAX_CLOSES}). Stopping.`);
          return;
        }

        if (!shouldProcessPR(pr)) {
          continue;
        }

        const lastActivity = getLastHumanActivity(pr);
        if (!isStale(lastActivity)) {
          continue;
        }

        const days = Math.floor((Date.now() - lastActivity) / MS_PER_DAY);
        console.log(`Closing PR ${pr.url} (inactive for ${days} days)`);

        await github.rest.issues.createComment({
          owner,
          repo,
          issue_number: pr.number,
          body: closeMessage(days),
        });
        await github.rest.pulls.update({
          owner,
          repo,
          pull_number: pr.number,
          state: "closed",
        });
        closeCount++;
      }
    }

    console.log(`Closed ${closeCount} stale PRs.`);
  } catch (error) {
    if (error.status === 429 || error.message?.includes("rate limit")) {
      console.log(`Rate limit hit after closing ${closeCount} PRs. Exiting gracefully.`);
      return;
    }
    throw error;
  }
};
