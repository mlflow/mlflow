module.exports = async ({ github, context }) => {
  const {
    repo: { owner, repo },
  } = context;

  const MERGE_INTERVAL_MS = 5000; // 5 seconds pause after a merge
  const MAX_RETRIES = 3;
  const RETRY_INTERVAL_MS = 10000; // 10 seconds

  async function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function logRateLimit() {
    const { data: rateLimit } = await github.rest.rateLimit.get();
    console.log(`Rate limit remaining: ${rateLimit.resources.core.remaining}`);
    console.log(
      `Rate limit resets at: ${new Date(rateLimit.resources.core.reset * 1000).toISOString()}`
    );
  }

  async function waitUntilMergeable(prNumber) {
    for (let i = 0; i < MAX_RETRIES; i++) {
      const pullRequest = await github.rest.pulls
        .get({
          owner,
          repo,
          pull_number: prNumber,
        })
        .then((res) => res.data);

      if (pullRequest.mergeable !== null) {
        return pullRequest.mergeable;
      }

      console.log(`Waiting for mergeability calculation for PR #${prNumber}...`);
      await sleep(RETRY_INTERVAL_MS);
    }
    return false;
  }

  async function areAllChecksPassed(sha) {
    const { data: checkRuns } = await github.rest.checks.listForRef({
      owner,
      repo,
      ref: sha,
    });
    return checkRuns.check_runs.every(({ conclusion }) =>
      ["success", "skipped"].includes(conclusion)
    );
  }

  // Get date from a month ago in ISO format
  const oneMonthAgo = new Date();
  oneMonthAgo.setMonth(oneMonthAgo.getMonth() - 1);
  const sinceDate = oneMonthAgo.toISOString();

  // List PRs with the "automerge" label created within the last month
  const { data: issues } = await github.rest.issues.listForRepo({
    owner,
    repo,
    labels: "automerge",
    since: sinceDate,
  });

  // Filter for pull requests from the list of issues
  const pullRequests = issues.filter((issue) => issue.pull_request);

  for (const pr of pullRequests) {
    if (!(await waitUntilMergeable(pr.number))) {
      console.log(`PR #${pr.number} is not mergeable. Skipping...`);
      await logRateLimit();
      continue;
    }

    if (await areAllChecksPassed(pullRequest.head.sha)) {
      try {
        await github.rest.pulls.merge({
          owner,
          repo,
          pull_number: pr.number,
        });
        console.log(`Merged PR #${pr.number}`);

        await sleep(MERGE_INTERVAL_MS);
        await logRateLimit();
      } catch (error) {
        console.log(`Failed to merge PR #${pr.number}. Reason: ${error.message}`);
      }
    } else {
      console.log(`Checks not ready for PR #${pr.number}. Skipping merge.`);
      await logRateLimit();
    }
  }
};
