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

    // If they are different workflow runs, prefer the one with a higher ID (auto-incrementing)
    if (newRun.id !== existingRun.id) {
      return newRun.id > existingRun.id;
    }

    // Same workflow run: higher run_attempt takes priority (re-runs)
    return newRun.run_attempt > existingRun.run_attempt;
  }
  function isJobFailed({ status, conclusion }) {
    return (
      conclusion === "cancelled" ||
      (status === "completed" && conclusion !== "success" && conclusion !== "skipped")
    );
  }

  async function fetchChecks(ref) {
    // Check runs (e.g., DCO check, but excluding GitHub Actions)
    const checkRuns = (
      await github.paginate(github.rest.checks.listForRef, {
        owner,
        repo,
        ref,
        filter: "latest",
        per_page: 100,
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
      pendingJobs: 0,
      status:
        conclusion === "cancelled"
          ? STATE.failure
          : status !== "completed"
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
        per_page: 100,
      })
    ).filter(
      ({ path, event }) =>
        // Exclude this workflow to avoid self-checking
        path !== ".github/workflows/protect.yml" &&
        // Exclude dynamic workflows (GitHub-managed, e.g., Copilot code review)
        event !== "dynamic"
    );

    // Deduplicate workflow runs by path and event, keeping the latest attempt
    const latestRuns = {};
    for (const run of workflowRuns) {
      const { path, event } = run;
      const key = `${path}-${event}`;
      if (isNewerRun(run, latestRuns[key])) {
        latestRuns[key] = run;
      }
    }

    for (const run of Object.values(latestRuns)) {
      const runName = run.path.replace(".github/workflows/", "");
      if (run.status === "completed") {
        // Use run-level status directly (0 extra API calls).
        checks.push({
          name: `${run.name} (${runName}, attempt ${run.run_attempt})`,
          pendingJobs: 0,
          status:
            run.conclusion === "cancelled"
              ? STATE.failure
              : run.conclusion === "success" || run.conclusion === "skipped"
              ? STATE.success
              : STATE.failure,
        });
      } else {
        let failed = false;
        let pendingJobs = 0;
        const jobsIter = github.paginate.iterator(github.rest.actions.listJobsForWorkflowRun, {
          owner,
          repo,
          run_id: run.id,
          per_page: 100,
        });
        for await (const { data: jobs } of jobsIter) {
          if (jobs.some(isJobFailed)) {
            failed = true;
            break;
          }
          pendingJobs += jobs.filter(({ status }) => status !== "completed").length;
        }
        checks.push({
          name: `${run.name} (${runName}, attempt ${run.run_attempt})`,
          pendingJobs: failed ? 0 : pendingJobs,
          status: failed ? STATE.failure : STATE.pending,
        });
      }
    }

    return checks;
  }

  const start = new Date();
  let iterationCount = 0;
  const TIMEOUT = 120 * 60 * 1000; // 2 hours
  await logRateLimit();
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
    const pendingJobs = checks
      .filter(({ status }) => status === STATE.pending)
      .reduce((sum, check) => sum + check.pendingJobs, 0);
    const sleepLength = getSleepLength(iterationCount, pendingJobs);
    console.log(`Sleeping for ${sleepLength / 1000} seconds (${pendingJobs} pending jobs)`);
    await sleep(sleepLength);
  }

  throw new Error("Timeout");
};
