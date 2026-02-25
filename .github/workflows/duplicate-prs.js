// Label duplicate community PRs that reference the same issue
// Only considers PRs opened in the last 14 days
// Keeps the oldest PR and labels newer ones as duplicates

const MS_PER_DAY = 24 * 60 * 60 * 1000;
const DAYS_TO_CONSIDER = 14;
const DUPLICATE_LABEL = "duplicate";

const duplicateMessage = (issueNumber, keeperPR) =>
  `This PR appears to reference the same issue (#${issueNumber}) as #${keeperPR} (opened earlier). If your change is already covered, please consider closing this PR.`;

// GraphQL query to fetch open PRs created in the last 14 days
const QUERY = `
  query($cursor: String, $searchQuery: String!) {
    rateLimit { remaining resetAt }
    search(query: $searchQuery, type: ISSUE, first: 50, after: $cursor) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        ... on PullRequest {
          number
          createdAt
          url
          author { login }
          authorAssociation
          closingIssuesReferences(first: 10) {
            nodes {
              number
            }
          }
        }
      }
    }
  }
`;

const shouldProcessPR = (pr) => {
  // Only process community PRs (skip maintainer PRs)
  const memberAssociations = ["MEMBER", "OWNER", "COLLABORATOR"];
  if (memberAssociations.includes(pr.authorAssociation)) {
    return false;
  }
  return true;
};

const getIssueReferences = (pr) => {
  const references = pr.closingIssuesReferences?.nodes || [];
  return references.map((node) => node.number);
};

module.exports = async ({ context, github }) => {
  const { owner, repo } = context.repo;

  try {
    // Calculate the date 14 days ago
    const fourteenDaysAgo = new Date(Date.now() - DAYS_TO_CONSIDER * MS_PER_DAY);
    const dateString = fourteenDaysAgo.toISOString().slice(0, 10);
    const searchQuery = `repo:${owner}/${repo} is:pr is:open created:>${dateString}`;

    console.log(`Searching for PRs: ${searchQuery}`);

    let cursor = null;
    let hasNextPage = true;
    const allPRs = [];

    // Fetch all open PRs from the last 14 days
    while (hasNextPage) {
      const response = await github.graphql(QUERY, { cursor, searchQuery });
      const { remaining, resetAt } = response.rateLimit;
      console.log(`Rate limit: ${remaining} remaining, resets at ${resetAt}`);

      const { nodes, pageInfo } = response.search;
      hasNextPage = pageInfo.hasNextPage;
      cursor = pageInfo.endCursor;

      allPRs.push(...nodes);
    }

    console.log(`Found ${allPRs.length} open PRs from the last ${DAYS_TO_CONSIDER} days`);

    // Filter to community PRs only
    const communityPRs = allPRs.filter(shouldProcessPR);
    console.log(`${communityPRs.length} are community PRs`);

    // Group PRs by the single issue they reference
    // Skip PRs that reference multiple issues (ambiguous intent)
    const prsByIssue = new Map();

    for (const pr of communityPRs) {
      const issueRefs = getIssueReferences(pr);

      if (issueRefs.length === 0) {
        // PR doesn't reference any issue, skip it
        continue;
      }

      if (issueRefs.length > 1) {
        // PR references multiple issues, skip it (ambiguous)
        console.log(
          `Skipping PR #${pr.number}: references multiple issues (${issueRefs.join(", ")})`
        );
        continue;
      }

      // PR references exactly one issue
      const issueNumber = issueRefs[0];
      if (!prsByIssue.has(issueNumber)) {
        prsByIssue.set(issueNumber, []);
      }
      prsByIssue.get(issueNumber).push(pr);
    }

    console.log(`Found ${prsByIssue.size} issues with associated PRs`);

    // Process each issue that has multiple PRs
    let labelCount = 0;
    for (const [issueNumber, prs] of prsByIssue.entries()) {
      if (prs.length <= 1) {
        // Only one PR for this issue, no duplicates
        continue;
      }

      console.log(`Issue #${issueNumber} has ${prs.length} PRs`);

      // Sort PRs by creation date (oldest first)
      prs.sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt));

      // Keep the oldest PR, label the rest as duplicates
      const [keeper, ...duplicates] = prs;
      console.log(`  Keeping PR #${keeper.number} (oldest, created ${keeper.createdAt})`);

      for (const pr of duplicates) {
        console.log(`  Labeling PR #${pr.number} as duplicate (created ${pr.createdAt})`);

        // Add duplicate label
        await github.rest.issues.addLabels({
          owner,
          repo,
          issue_number: pr.number,
          labels: [DUPLICATE_LABEL],
        });

        // Post comment encouraging author to close if duplicate
        await github.rest.issues.createComment({
          owner,
          repo,
          issue_number: pr.number,
          body: duplicateMessage(issueNumber, keeper.number),
        });

        labelCount++;
      }
    }

    console.log(`Labeled ${labelCount} duplicate PRs.`);
  } catch (error) {
    if (error.status === 429 || error.message?.includes("rate limit")) {
      console.log(`Rate limit hit. Exiting gracefully.`);
      return;
    }
    throw error;
  }
};
