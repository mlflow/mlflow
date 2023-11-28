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
  const MINUTE = 1000 * 60;
  const TIMEOUT = 120 * MINUTE; // 2 hours
  while (new Date() - start < TIMEOUT) {
    const checks = await fetchChecks(sha);
    const longest = Math.max(...checks.map(({ name }) => name.length));
    checks.forEach(({ name, status }) => {
      const icon = status === STATE.success ? "✅" : status === STATE.failure ? "❌" : "🕒";
      console.log(`- ${name.padEnd(longest)}: ${icon} ${status}`);
    });

    if (checks.some(({ status }) => status === STATE.failure)) {
      throw new Error("Found failed job(s)");
    }

    if (checks.length > 0 && checks.every(({ status }) => status === STATE.success)) {
      console.log("All checks passed");
      return;
    }

    await logRateLimit();
    await sleep(3 * MINUTE);
  }

  throw new Error("Timeout");
};
