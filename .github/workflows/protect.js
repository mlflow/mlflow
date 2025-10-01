function getSleepLength(iterationCount, numPendingJobs) {
  if (iterationCount <= 5 && numPendingJobs <= 5) {
    // It's likely that this job was triggered with other quick jobs.
    // To minimize the wait time, shorten the polling interval for the first 5 iterations.
    return 5 * 1000; // 5 seconds
  }
  // If the number of pending jobs is small, poll more frequently to reduce wait time.
  return (numPendingJobs <= 7 ? 30 : 5 * 60) * 1000;
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

  function isNewerRun(newRun, existingRun) {
    // Returns true if newRun should replace existingRun
    if (!existingRun) return true;

    // Higher run_attempt takes priority (re-runs)
    if (newRun.run_attempt > existingRun.run_attempt) return true;

    // For same run_attempt, use newer created_at as tiebreaker
    if (
      newRun.run_attempt === existingRun.run_attempt &&
      new Date(newRun.created_at) > new Date(existingRun.created_at)
    ) {
      return true;
    }

    return false;
  }

  async function fetchChecks(ref) {
    // Check runs (e.g., DCO check, but excluding GitHub Actions)
    const checkRuns = (
      await github.paginate(github.rest.checks.listForRef, {
        owner,
        repo,
        ref,
        filter: "latest",
      })
    ).filter(({ app }) => app?.slug !== "github-actions");

    const latestCheckRuns = {};
    for (const run of checkRuns) {
      const { name } = run;
      if (
        !latestCheckRuns[name] ||
        new Date(run.started_at) > new Date(latestCheckRuns[name].started_at)
      ) {
        latestCheckRuns[name] = run;
      }
    }
    const checks = Object.values(latestCheckRuns).map(({ name, status, conclusion }) => ({
      name,
      status:
        status !== "completed"
          ? STATE.pending
          : conclusion === "success" || conclusion === "skipped"
          ? STATE.success
          : STATE.failure,
    }));

    // Workflow runs (e.g., GitHub Actions)
    const workflowRuns = (
      await github.paginate(github.rest.actions.listWorkflowRunsForRepo, {
        owner,
        repo,
        head_sha: ref,
      })
    ).filter(({ path }) => path !== ".github/workflows/protect.yml");

    // Deduplicate workflow runs by path and event, keeping the latest attempt
    const latestRuns = {};
    for (const run of workflowRuns) {
      const { path, event } = run;
      const key = `${path}-${event}`;
      if (isNewerRun(run, latestRuns[key])) {
        latestRuns[key] = run;
      }
    }

    // Fetch jobs for each workflow run
    const runs = [];
    for (const run of Object.values(latestRuns)) {
      // Fetch jobs for this workflow run
      const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
        owner,
        repo,
        run_id: run.id,
      });

      // Process each job as a separate check
      for (const job of jobs) {
        const runName = run.path.replace(".github/workflows/", "");
        runs.push({
          name: `${job.name} (${runName}, attempt ${run.run_attempt})`,
          status:
            job.status !== "completed"
              ? STATE.pending
              : job.conclusion === "success" || job.conclusion === "skipped"
              ? STATE.success
              : STATE.failure,
        });
      }
    }

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

    return [...checks, ...runs, ...statuses].sort((a, b) => a.name.localeCompare(b.name));
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
    console.log(`Sleeping for ${sleepLength / 1000} seconds (${pendingJobs.length} pending jobs)`);
    await sleep(sleepLength);
  }

  throw new Error("Timeout");
};
