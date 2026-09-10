function getSleepLength(iterationCount, pendingWorkflowRuns) {
  if (iterationCount <= 2) {
    return 15 * 1000;
  }
  return (pendingWorkflowRuns <= 7 ? 30 : 5 * 60) * 1000;
}
module.exports = async ({ github, context }) => {
  let rateLimitRemaining;
  github.hook.after("request", (response) => {
    rateLimitRemaining = response.headers["x-ratelimit-remaining"];
  });

  const {
    repo: { owner, repo },
  } = context;
  const pullRequest = context.payload.pull_request;
  const { sha } = pullRequest.head;

  // TODO: Remove this once stacked PRs support force-merging.
  // The `unprotect` label bypasses this check on stacked PRs, which can't be
  // force-merged like regular PRs (`gh pr merge --admin`). The `stack` property
  // is only present while the PR belongs to a stack.
  if (pullRequest.labels.some(({ name }) => name === "unprotect")) {
    if (pullRequest.stack) {
      console.log("The `unprotect` label is present on a stacked PR. Skipping this check.");
      return;
    }
    console.log("Ignoring the `unprotect` label: it is only valid on stacked PRs.");
  }

  const STATE = {
    pending: "pending",
    success: "success",
    skipped: "skipped",
    failure: "failure",
  };

  const IGNORED_WORKFLOWS = new Set([".github/workflows/rerun.yml"]);

  async function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
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
    const checks = Object.values(latestCheckRuns).map(({ name, status, conclusion, html_url }) => ({
      name,
      url: html_url,
      status:
        conclusion === "cancelled"
          ? STATE.failure
          : status !== "completed"
          ? STATE.pending
          : conclusion === "success"
          ? STATE.success
          : conclusion === "skipped"
          ? STATE.skipped
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
        event !== "dynamic" &&
        !IGNORED_WORKFLOWS.has(path)
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
      checks.push({
        name: `${run.name} (${runName}, attempt ${run.run_attempt})`,
        url: `${run.html_url}/attempts/${run.run_attempt}`,
        status:
          run.status !== "completed"
            ? STATE.pending
            : run.conclusion === "cancelled"
            ? STATE.failure
            : run.conclusion === "success"
            ? STATE.success
            : run.conclusion === "skipped"
            ? STATE.skipped
            : STATE.failure,
      });
    }

    return {
      checks,
      pendingWorkflowRuns: Object.values(latestRuns).filter(({ status }) => status !== "completed")
        .length,
    };
  }

  const start = new Date();
  let iterationCount = 0;
  const TIMEOUT = 120 * 60 * 1000; // 2 hours
  while (new Date() - start < TIMEOUT) {
    ++iterationCount;
    const { checks, pendingWorkflowRuns } = await fetchChecks(sha);
    if (rateLimitRemaining !== undefined) {
      console.log(`Rate limit remaining: ${rateLimitRemaining}`);
    }
    const longest = Math.max(...checks.map(({ name }) => name.length));
    checks.forEach(({ name, status, url }) => {
      const icon =
        status === STATE.success
          ? "✅"
          : status === STATE.skipped
          ? "⏭️"
          : status === STATE.failure
          ? "❌"
          : "🕒";
      console.log(`- ${name.padEnd(longest)}: ${icon} ${status}${url ? ` (${url})` : ""}`);
    });

    if (checks.some(({ status }) => status === STATE.failure)) {
      throw new Error(
        "This job ensures that all checks except for this one have passed to prevent accidental auto-merges."
      );
    }

    if (
      checks.length > 0 &&
      checks.every(({ status }) => status === STATE.success || status === STATE.skipped)
    ) {
      console.log("All checks passed or were skipped");
      return;
    }

    const sleepLength = getSleepLength(iterationCount, pendingWorkflowRuns);
    console.log(
      `Sleeping for ${sleepLength / 1000} seconds (${pendingWorkflowRuns} pending workflows)`
    );
    await sleep(sleepLength);
  }

  throw new Error("Timeout");
};
