// Custom stale PR workflow using GitHub Timeline API
// Detects last human activity (ignoring bot events) and closes inactive PRs

const STALE_DAYS = 30;
// TODO: Increase once we're confident the workflow works correctly
const MAX_CLOSES = 10;
const CLOSE_MESSAGE = "Closing due to inactivity. Feel free to reopen if still relevant.";

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
          createdAt
          authorAssociation
          closingIssuesReferences(first: 1) {
            totalCount
          }
          timelineItems(
            last: 5
            itemTypes: [ISSUE_COMMENT, PULL_REQUEST_REVIEW, PULL_REQUEST_COMMIT]
          ) {
            nodes {
              __typename
              ... on IssueComment {
                createdAt
                author {
                  __typename
                  login
                }
              }
              ... on PullRequestReview {
                createdAt
                author {
                  __typename
                  login
                }
              }
              ... on PullRequestCommit {
                commit {
                  committedDate
                  author {
                    user {
                      __typename
                      login
                    }
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

const isBot = (author) => !author || author.__typename === "Bot";

const getLastHumanActivity = (pr) => {
  const items = pr.timelineItems.nodes || [];

  for (let i = items.length - 1; i >= 0; i--) {
    const item = items[i];

    if (item.__typename === "PullRequestCommit") {
      const user = item.commit?.author?.user;
      if (user && !isBot(user)) {
        return new Date(item.commit.committedDate);
      }
    } else if (!isBot(item.author)) {
      return new Date(item.createdAt);
    }
  }

  // No human activity found, fall back to PR creation date
  return new Date(pr.createdAt);
};

const isStale = (lastActivityDate) => {
  const now = new Date();
  const daysSinceActivity = (now - lastActivityDate) / (1000 * 60 * 60 * 24);
  return daysSinceActivity > STALE_DAYS;
};

const shouldProcessPR = (pr) => {
  // Skip community PRs — only close internal (org member) PRs
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
    const iterator = github.graphql.paginate.iterator(QUERY);

    for await (const response of iterator) {
      const { remaining, resetAt } = response.rateLimit;
      console.log(`Rate limit: ${remaining} remaining, resets at ${resetAt}`);

      for (const pr of response.repository.pullRequests.nodes) {
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

        const days = Math.floor((Date.now() - lastActivity) / 86400000);
        console.log(`Closing PR #${pr.number} (inactive for ${days} days)`);

        await github.rest.issues.createComment({
          owner,
          repo,
          issue_number: pr.number,
          body: CLOSE_MESSAGE,
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
