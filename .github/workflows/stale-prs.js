// Custom stale PR workflow using GitHub Timeline API
// Detects last human activity (ignoring bot events) and closes inactive PRs

const STALE_DAYS = 30;
const MAX_OPERATIONS = 100;
const CLOSE_MESSAGE = "Closing due to inactivity. Feel free to reopen if still relevant.";

// GraphQL query to fetch open PRs with timeline data
const QUERY = `
  query($cursor: String) {
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
          timelineItems(
            last: 5
            itemTypes: [ISSUE_COMMENT, PULL_REQUEST_REVIEW, PULL_REQUEST_COMMIT]
          ) {
            nodes {
              __typename
              ... on IssueComment {
                createdAt
                author {
                  login
                }
              }
              ... on PullRequestReview {
                createdAt
                author {
                  login
                }
              }
              ... on PullRequestCommit {
                commit {
                  committedDate
                  author {
                    user {
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

const isBot = (login) => {
  return login && login.endsWith("[bot]");
};

const getLastHumanActivity = (pr) => {
  const items = pr.timelineItems.nodes || [];

  // Iterate in reverse to find the most recent human activity
  for (let i = items.length - 1; i >= 0; i--) {
    const item = items[i];

    if (item.__typename === "IssueComment") {
      const login = item.author?.login;
      if (login && !isBot(login)) {
        return new Date(item.createdAt);
      }
    } else if (item.__typename === "PullRequestReview") {
      const login = item.author?.login;
      if (login && !isBot(login)) {
        return new Date(item.createdAt);
      }
    } else if (item.__typename === "PullRequestCommit") {
      const login = item.commit?.author?.user?.login;
      if (login && !isBot(login)) {
        return new Date(item.commit.committedDate);
      }
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
  // Only process PRs from org members
  const memberAssociations = ["MEMBER", "OWNER", "COLLABORATOR"];
  return memberAssociations.includes(pr.authorAssociation);
};

module.exports = async ({ context, github }) => {
  const { owner, repo } = context.repo;
  let operationsCount = 0;

  try {
    console.log("Fetching open pull requests...");

    // Use paginate to fetch all open PRs
    const iterator = github.graphql.paginate.iterator(QUERY);

    for await (const response of iterator) {
      const prs = response.repository.pullRequests.nodes;

      for (const pr of prs) {
        // Check if we've hit the operations limit
        if (operationsCount >= MAX_OPERATIONS) {
          console.log(`Reached maximum operations limit (${MAX_OPERATIONS}). Stopping.`);
          return;
        }

        // Skip community PRs (non-org members)
        if (!shouldProcessPR(pr)) {
          continue;
        }

        // Get last human activity
        const lastActivity = getLastHumanActivity(pr);

        // Check if PR is stale
        if (isStale(lastActivity)) {
          const daysSinceActivity = Math.floor((new Date() - lastActivity) / (1000 * 60 * 60 * 24));

          console.log(`Closing PR #${pr.number} (inactive for ${daysSinceActivity} days)`);

          try {
            // Add comment
            await github.rest.issues.createComment({
              owner,
              repo,
              issue_number: pr.number,
              body: CLOSE_MESSAGE,
            });

            // Close PR
            await github.rest.pulls.update({
              owner,
              repo,
              pull_number: pr.number,
              state: "closed",
            });

            operationsCount++;
          } catch (error) {
            // Check for rate limit error
            if (error.status === 429) {
              console.log("Rate limit reached. Exiting gracefully.");
              return;
            }
            throw error;
          }
        }
      }
    }

    console.log(`Processed ${operationsCount} stale PRs.`);
  } catch (error) {
    // Handle rate limit errors gracefully
    if (error.status === 429 || error.message?.includes("rate limit")) {
      console.log("Rate limit error encountered. Exiting gracefully.");
      return;
    }
    throw error;
  }
};
