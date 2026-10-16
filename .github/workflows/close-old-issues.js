const fs = require("fs");
const MS_PER_DAY = 24 * 60 * 60 * 1000;
const CUTOFF_DAYS = 180;
const EXCLUDED_LABELS = new Set(["security"]);

const SEARCH_QUERY = `
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
            pageInfo { hasNextPage endCursor }
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

const TIMELINE_QUERY = `
  query($owner: String!, $repo: String!, $number: Int!, $cursor: String) {
    repository(owner: $owner, name: $repo) {
      issue(number: $number) {
        timelineItems(itemTypes: [CROSS_REFERENCED_EVENT], first: 100, after: $cursor) {
          pageInfo { hasNextPage endCursor }
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
`;

function getLabels(issue) {
  return issue.labels?.nodes?.map((label) => label.name) ?? [];
}

// Widely-linked issues can have more than 100 cross-reference events, so the
// first page fetched with the search may be truncated; paginate the tail.
async function getClosingPullRequests(github, owner, repo, issue) {
  const timeline = issue.timelineItems ?? { nodes: [] };
  let nodes = timeline.nodes ?? [];
  let cursor = timeline.pageInfo?.hasNextPage ? timeline.pageInfo.endCursor : null;

  if (cursor) {
    console.log(`Issue #${issue.number} has more than 100 cross-references; fetching the rest.`);
  }
  while (cursor) {
    const response = await github.graphql(TIMELINE_QUERY, {
      owner,
      repo,
      number: issue.number,
      cursor,
    });
    const page = response.repository.issue.timelineItems;
    nodes = nodes.concat(page.nodes);
    cursor = page.pageInfo.hasNextPage ? page.pageInfo.endCursor : null;
  }

  return nodes
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

function writeSummary({ dryRun, failed, rateLimited, issueNumbers, pullRequestNumbers }) {
  if (!process.env.GITHUB_STEP_SUMMARY) {
    return;
  }

  const lines = ["## Old issue cleanup", ""];
  if (rateLimited) {
    lines.push(
      `Stopped early on the GitHub API rate limit after processing ${issueNumbers.length} issues and ${pullRequestNumbers.length} pull requests; the next run resumes.`
    );
  } else if (failed) {
    lines.push(
      `Stopped early on an error after processing ${issueNumbers.length} issues and ${pullRequestNumbers.length} pull requests; check the run logs.`
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
  // page holds only eligible issues.
  const searchQuery = `repo:${owner}/${repo} is:issue is:open created:<${cutoffDate} -label:security reactions:0`;
  const linkedPullRequestUrls = new Map();
  const closedIssueNumbers = [];
  const closedPullRequestNumbers = [];
  let cursor = null;
  let hasNextPage = true;
  const eligibleIssues = [];
  let matchedCount = 0;
  let collectedCount = 0;

  // Collect everything before closing: closing issues mid-pagination would
  // remove them from the open-issues result set and shift the cursor.
  while (hasNextPage) {
    const response = await github.graphql(SEARCH_QUERY, { searchQuery, cursor });
    const { remaining, resetAt } = response.rateLimit;
    console.log(`Rate limit: ${remaining} remaining, resets at ${resetAt}`);

    const { issueCount, pageInfo, nodes } = response.search;
    matchedCount = issueCount;
    collectedCount += nodes.length;
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

  if (collectedCount < matchedCount) {
    console.log(
      `Collected ${collectedCount} of ${matchedCount} matched issues; the GraphQL search API caps a single query at 1000 results. Closing these makes progress and the next run continues.`
    );
  }

  console.log(`Found ${eligibleIssues.length} eligible issues to close.`);

  let rateLimited = false;
  let failed = false;

  try {
    // Map every linked PR to all eligible issues that reference it before any
    // mutation, so each PR can be closed and commented back-to-back while the
    // comment still lists every linked issue.
    for (const issue of eligibleIssues) {
      for (const pullRequest of await getClosingPullRequests(github, owner, repo, issue)) {
        const urls = linkedPullRequestUrls.get(pullRequest.number) ?? [];
        urls.push(issue.url);
        linkedPullRequestUrls.set(pullRequest.number, urls);
      }
    }

    for (const [pullRequestNumber, urls] of linkedPullRequestUrls) {
      if (dryRun) {
        const linked = `linked issue${urls.length > 1 ? "s" : ""}: ${urls.join(", ")}`;
        console.log(`[dry run] Would close PR #${pullRequestNumber} (${linked})`);
        closedPullRequestNumbers.push(pullRequestNumber);
        continue;
      }

      await github.rest.pulls.update({
        owner,
        repo,
        pull_number: pullRequestNumber,
        state: "closed",
      });
      await github.rest.issues.createComment({
        owner,
        repo,
        issue_number: pullRequestNumber,
        body: `${pullRequestMessage}\n\nLinked issue${urls.length > 1 ? "s" : ""}: ${urls.join(
          ", "
        )}`,
      });
      console.log(
        `Closed PR #${pullRequestNumber} linked to ${urls.length} issue${
          urls.length > 1 ? "s" : ""
        }.`
      );
      closedPullRequestNumbers.push(pullRequestNumber);
    }

    for (const issue of eligibleIssues) {
      if (dryRun) {
        console.log(`[dry run] Would close issue #${issue.number}`);
        closedIssueNumbers.push(issue.number);
        continue;
      }

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
      closedIssueNumbers.push(issue.number);
      console.log(`Closed issue #${issue.number}.`);
    }
  } catch (error) {
    failed = true;
    if (!isRateLimitError(error)) {
      throw error;
    }
    rateLimited = true;
  } finally {
    writeSummary({
      dryRun,
      failed,
      rateLimited,
      issueNumbers: closedIssueNumbers,
      pullRequestNumbers: closedPullRequestNumbers,
    });
  }

  const verb = dryRun ? "Would close" : "Closed";
  const suffix = rateLimited ? " (stopped early by the rate limit; the next run resumes)" : "";
  console.log(
    `${verb} ${closedIssueNumbers.length} issues and ${
      closedPullRequestNumbers.length
    } linked pull requests${dryRun ? " in a dry run" : ""}${suffix}.`
  );
};
