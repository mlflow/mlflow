module.exports = async ({ github, context }) => {
  const {
    repo: { owner, repo },
  } = context;
  const { ref } = context.payload.pull_request.head;

  async function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function logRateLimit() {
    const { data: rateLimit } = await github.rest.rateLimit.get();
    console.log(`Rate limit remaining: ${rateLimit.resources.core.remaining}`);
  }

  async function allChecksPassed(ref) {
    const checkRuns = await github.paginate(github.rest.checks.listForRef, {
      owner,
      repo,
      ref,
    });

    const latestRuns = {};
    for (const run of checkRuns) {
      const { name } = run;
      if (
        !latestRuns[name] ||
        new Date(run.started_at) > new Date(latestRuns[name].started_at)
      ) {
        latestRuns[name] = run;
      }
    }
    Object.values(latestRuns).forEach(({ name, status, conclusion }) => {
      console.log(
        `- checkrun: name: ${name}, latest status: ${status}, conclusion: ${conclusion}`
      );
    });

    const checksPassed = Object.values(latestRuns).every(({ conclusion }) =>
      ["success", "skipped"].includes(conclusion)
    );

    // Commit statues (e.g., CircleCI checks)
    const commitStatuses = await github.paginate(
      github.rest.repos.getCombinedStatusForRef,
      {
        owner,
        repo,
        ref,
      }
    );

    const latestStatuses = {};
    for (const status of commitStatuses) {
      const { context } = status;
      if (
        !latestStatuses[context] ||
        new Date(status.created_at) >
          new Date(latestStatuses[context].created_at)
      ) {
        latestStatuses[context] = status;
      }
    }

    Object.values(latestStatuses).forEach(({ context, state }) => {
      console.log(
        `- commit status: context: ${context}, latest state: ${state}`
      );
    });

    const statusesPassed = Object.values(latestStatuses).every(
      ({ state }) => state === "success"
    );

    return checksPassed && statusesPassed;
  }

  const start = new Date();
  const MINUTE = 1000 * 60;
  while (true) {
    if (allChecksPassed(ref)) {
      break;
    }

    if (new Date() - start > 180 * MINUTE) {
      throw new Error("Timeout");
    }

    await logRateLimit();
    await sleep(3 * MINUTE);
  }
};
