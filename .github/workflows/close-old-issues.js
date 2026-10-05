const fs = require("fs");
const MS_PER_DAY = 24 * 60 * 60 * 1000;
const CUTOFF_DAYS = 180;
const EXCLUDED_LABELS = new Set(["security"]);

const QUERY = `
  query($searchQuery: String!, $cursor: String) {
    rateLimit { remaining resetAt }
    search(query: $searchQuery, type: ISSUE, first: 100, after: $cursor) {
      issueCount
      pageInfo {
        hasNextPage
        endCursor
      }
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

function isEligible(issue, cutoffTime) {
  return (
    new Date(issue.createdAt).getTime() < cutoffTime &&
    issue.reactions.totalCount === 0 &&
    !getLabels(issue).some((label) => EXCLUDED_LABELS.has(label))
  );
}

function isRateLimitError(error) {
  return error.status === 429 || error.message?.includes("rate limit");
}

function writeSummary({ dryRun, rateLimited, issueNumbers, pullRequestNumbers }) {
  if (!process.env.GITHUB_STEP_SUMMARY) {
    return;
  }

  const lines = ["## Old issue cleanup", ""];
  if (rateLimited) {
    lines.push(
      `Stopped early on the GitHub API rate limit after processing ${issueNumbers.length} issues and ${pullRequestNumbers.length} pull requests; the next run resumes.`
    );
  } else if (dryRun) {
    lines.push(
      `Dry run: found ${issueNumbers.length} eligible issues and ${pullRequestNumbers.length} linked pull requests. Nothing was closed.`
    );
  } else {
    lines.push(
      `Closed ${issueNumbers.length} issues and ${pullRequestNumbers.length} linked pull requests.`
    );
  }

  if (pullRequestNumbers.length > 0) {
    lines.push("", "### Pull requests", "");
    for (const number of pullRequestNumbers) {
      lines.push(`- #${number}`);
    }
  }

  if (issueNumbers.length > 0) {
    lines.push("", "### Issues", "");
    for (const number of issueNumbers) {
      lines.push(`- #${number}`);
    }
  }

  fs.appendFileSync(process.env.GITHUB_STEP_SUMMARY, `${lines.join("\n")}\n`);
}

module.exports = async ({ context, github }) => {
  const { owner, repo } = context.repo;
  const dryRun = process.env.DRY_RUN !== "false";
  const issueMessage = process.env.ISSUE_CLOSE_MESSAGE?.trim();
  const pullRequestMessage = process.env.PR_CLOSE_MESSAGE?.trim();

  if (!issueMessage || !pullRequestMessage) {
    throw new Error("ISSUE_CLOSE_MESSAGE and PR_CLOSE_MESSAGE are required.");
  }

  const cutoffTime = Date.now() - CUTOFF_DAYS * MS_PER_DAY;
  const cutoffDate = new Date(cutoffTime).toISOString().slice(0, 10);
  // Label and reaction exclusions live in the search query so every result
  // page holds only eligible issues; the GraphQL search API still caps any
  // single query at 1000 results, but closing those makes progress for the
  // next run.
  const searchQuery = `repo:${owner}/${repo} is:issue is:open created:<${cutoffDate} -label:security reactions:0`;
  const processedPullRequests = new Set();
  const closedIssueNumbers = [];
  const closedPullRequestNumbers = [];
  let cursor = null;
  let hasNextPage = true;
  const eligibleIssues = [];

  // Collect everything before closing: closing issues mid-pagination would
  // remove them from the open-issues result set and shift the cursor.
  while (hasNextPage) {
    const response = await github.graphql(QUERY, { searchQuery, cursor });
    const { remaining, resetAt } = response.rateLimit;
    console.log(`Rate limit: ${remaining} remaining, resets at ${resetAt}`);

    const { issueCount, pageInfo, nodes } = response.search;
    console.log(
      `Search matched ${issueCount} open issues older than ${CUTOFF_DAYS} days without reactions.`
    );

    for (const issue of nodes) {
      if (isEligible(issue, cutoffTime)) {
        eligibleIssues.push(issue);
      }
    }

    hasNextPage = pageInfo.hasNextPage;
    cursor = pageInfo.endCursor;
  }

  console.log(`Found ${eligibleIssues.length} eligible issues to close.`);

  let rateLimited = false;

  try {
    for (const issue of eligibleIssues) {
      for (const pullRequest of getClosingPullRequests(issue)) {
        if (processedPullRequests.has(pullRequest.number)) {
          continue;
        }
        processedPullRequests.add(pullRequest.number);

        if (!dryRun) {
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
            body: `${pullRequestMessage}\n\nLinked issue: ${issue.url}`,
          });
          console.log(`Closed PR #${pullRequest.number} linked to issue #${issue.number}.`);
        }
        closedPullRequestNumbers.push(pullRequest.number);
      }

      if (!dryRun) {
        await github.rest.issues.createComment({
          owner,
          repo,
          issue_number: issue.number,
          body: issueMessage,
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
      closedIssueNumbers.push(issue.number);
    }
  } catch (error) {
    if (!isRateLimitError(error)) {
      throw error;
    }
    rateLimited = true;
  }

  writeSummary({
    dryRun,
    rateLimited,
    issueNumbers: closedIssueNumbers,
    pullRequestNumbers: closedPullRequestNumbers,
  });

  const verb = dryRun ? "Would close" : "Closed";
  const suffix = rateLimited ? " (stopped early by the rate limit; the next run resumes)" : "";
  console.log(
    `${verb} ${closedIssueNumbers.length} issues and ${
      closedPullRequestNumbers.length
    } linked pull requests${dryRun ? " in a dry run" : ""}${suffix}.`
  );
};
