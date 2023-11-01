module.exports = async ({ github, context }) => {
  const {
    repo: { owner, repo },
  } = context;

  const MERGE_INTERVAL_MS = 5000; // 5 seconds pause after a merge
  const PR_FETCH_RETRY_INTERVAL_MS = 5000; // 5 seconds
  const PR_FETCH_MAX_ATTEMPTS = 10;

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
    for (let i = 0; i < PR_FETCH_MAX_ATTEMPTS; i++) {
      const pr = await github.rest.pulls
        .get({
          owner,
          repo,
          pull_number: prNumber,
        })
        .then((res) => res.data);

      if (pr.merged) {
        return null;
      }

      if (pr.mergeable && pr.mergeable_state === "clean") {
        return pr;
      }

      console.log(`Waiting for GitHub to recalculate mergeability of PR #${prNumber}...`);
      await sleep(PR_FETCH_RETRY_INTERVAL_MS);
    }
    return null;
  }

  async function allChecksPassed(ref) {
    const checkRuns = await github.paginate(github.rest.checks.listForRef, {
      owner,
      repo,
      ref,
    });
    return checkRuns.check_runs.every(({ conclusion }) =>
      ["success", "skipped"].includes(conclusion)
    );
  }

  // List PRs with the "automerge" label updated in the last two weeks
  const twoWeeksAgo = new Date();
  twoWeeksAgo.setDate(twoWeeksAgo.getDate() - 14);
  const { data: issues } = await github.rest.issues.listForRepo({
    owner,
    repo,
    state: "open",
    sort: "updated",
    direction: "asc",
    per_page: 30,
    labels: "automerge",
    since: twoWeeksAgo.toISOString(),
  });
  // Exclude issues
  const prs = issues.filter((issue) => issue.pull_request);

  for (const { number } of prs) {
    const pr = await waitUntilMergeable(number);
    if (pr === null) {
      console.log(`PR #${pr.number} is not mergeable. Skipping...`);
      continue;
    }

    if (!(await allChecksPassed(pr.head.sha))) {
      console.log(`Not all checks passed for PR #${pr.number}. Skipping...`);
      continue;
    }

    try {
      console.log(`Would merge PR #${pr.number}`);
      // await github.rest.pulls.merge({
      //   owner,
      //   repo,
      //   pull_number: pr.number,
      // });
      console.log(`Merged PR #${pr.number}`);
      await sleep(MERGE_INTERVAL_MS);
    } catch (error) {
      console.log(`Failed to merge PR #${pr.number}. Reason: ${error.message}`);
    }
  }

  await logRateLimit();
};
