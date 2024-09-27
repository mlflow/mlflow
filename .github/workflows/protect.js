function getSleepLength(iterationCount, numPendingJobs) {
  if (iterationCount <= 5 && numPendingJobs <= 5) {
    // It's likely that this job was triggered with other quick jobs.
    // To minimize the wait time, shorten the polling interval for the first 5 iterations.
    return 5 * 1000; // 5 seconds
  }
  return (numPendingJobs <= 3 ? 1 : 5) * 60 * 1000; // 1 minute or 5 minutes
}
module.exports = async ({ github, context }) => {
  const {
    repo: { owner, repo },
  } = context;
  const { sha } = context.payload.pull_request.head;

  const STATE = {
    pending: "pending",
    success: "success",
    failure: "failure",
  };

  async function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function logRateLimit() {
    const { data: rateLimit } = await github.rest.rateLimit.get();
    console.log(`Rate limit remaining: ${rateLimit.resources.core.remaining}`);
  }

  async function fetchChecks(ref) {
    // Check runs (e.g., GitHub Actions)
    const checkRuns = (
      await github.paginate(github.rest.checks.listForRef, {
        owner,
        repo,
        ref,
      })
    ).filter(({ name }) => name !== "protect");

    const latestRuns = {};
    for (const run of checkRuns) {
      const { name } = run;
      if (!latestRuns[name] || new Date(run.started_at) > new Date(latestRuns[name].started_at)) {
        latestRuns[name] = run;
      }
    }
    const runs = Object.values(latestRuns).map(({ name, status, conclusion }) => ({
      name,
      status:
        status !== "completed"
          ? STATE.pending
          : conclusion === "success" || conclusion === "skipped"
          ? STATE.success
          : STATE.failure,
    }));

    // Commit statues (e.g., CircleCI checks)
    const commitStatuses = await github.paginate(github.rest.repos.listCommitStatusesForRef, {
      owner,
      repo,
      ref,
    });

    const latestStatuses = {};
    for (const status of commitStatuses) {
      const { context } = status;
      if (
        !latestStatuses[context] ||
        new Date(status.created_at) > new Date(latestStatuses[context].created_at)
      ) {
        latestStatuses[context] = status;
      }
    }

    const statuses = Object.values(latestStatuses).map(({ context, state }) => ({
      name: context,
      status:
        state === "pending" ? STATE.pending : state === "success" ? STATE.success : STATE.failure,
    }));

    return [...runs, ...statuses];
  }

  const start = new Date();
  let iterationCount = 0;
  const TIMEOUT = 120 * 60 * 1000; // 2 hours
  while (new Date() - start < TIMEOUT) {
    ++iterationCount;
    const checks = await fetchChecks(sha);
    const longest = Math.max(...checks.map(({ name }) => name.length));
    checks.forEach(({ name, status }) => {
      const icon = status === STATE.success ? "✅" : status === STATE.failure ? "❌" : "🕒";
      console.log(`- ${name.padEnd(longest)}: ${icon} ${status}`);
    });

    if (checks.some(({ status }) => status === STATE.failure)) {
      throw new Error(
        "This job ensures that all checks except for this one have passed to prevent accidental auto-merges."
      );
    }

    if (checks.length > 0 && checks.every(({ status }) => status === STATE.success)) {
      console.log("All checks passed");
      return;
    }

    await logRateLimit();
    const pendingJobs = checks.filter(({ status }) => status === STATE.pending);
    const sleepLength = getSleepLength(iterationCount, pendingJobs.length);
    await sleep(sleepLength);
  }

  throw new Error("Timeout");
};
