const assert = require("node:assert/strict");
const test = require("node:test");

const protect = require("./protect");

async function runProtect({ author, branch, workflowRuns }) {
  const checks = { listForRef: async () => ({ data: [] }) };
  const actions = {
    listWorkflowRunsForRepo: async () => ({
      data: typeof workflowRuns === "function" ? workflowRuns() : workflowRuns,
    }),
    listJobsForWorkflowRun: async () => ({ data: [] }),
  };
  const github = {
    hook: { after: () => {} },
    rest: { checks, actions },
    paginate: async (request) => (await request()).data,
  };
  github.paginate.iterator = (request) => ({
    async *[Symbol.asyncIterator]() {
      yield { data: (await request()).data };
    },
  });

  let failure;
  await protect({
    github,
    context: {
      repo: { owner: "mlflow", repo: "mlflow" },
      payload: {
        pull_request: {
          number: 1,
          user: { login: author },
          head: { ref: branch, sha: "abc123" },
        },
      },
    },
    core: {
      setFailed: (message) => (failure = message),
      warning: () => {},
    },
  });
  return failure;
}

function successfulWorkflow(path) {
  return {
    id: 1,
    run_attempt: 1,
    path,
    event: "pull_request_target",
    name: "Workflow",
    status: "completed",
    conclusion: "success",
    html_url: "https://example.com/run/1",
  };
}

test("Protect fails triage bot PRs when Approval never starts", async () => {
  const originalSetTimeout = global.setTimeout;
  global.setTimeout = (callback) => queueMicrotask(callback);
  try {
    const failure = await runProtect({
      author: "github-actions[bot]",
      branch: "triage/issue-123",
      workflowRuns: [successfulWorkflow(".github/workflows/tests.yml")],
    });

    assert.match(failure, /Approval workflow did not start/);
  } finally {
    global.setTimeout = originalSetTimeout;
  }
});

test("Protect completes when the triage bot PR's Approval run succeeds", async () => {
  const failure = await runProtect({
    author: "github-actions[bot]",
    branch: "triage/issue-123",
    workflowRuns: [
      successfulWorkflow(".github/workflows/tests.yml"),
      successfulWorkflow(".github/workflows/approval.yml"),
    ],
  });

  assert.equal(failure, undefined);
});

test("Protect waits through a startup race for the triage Approval run", async () => {
  const originalSetTimeout = global.setTimeout;
  global.setTimeout = (callback) => queueMicrotask(callback);
  let polls = 0;
  try {
    const failure = await runProtect({
      author: "github-actions[bot]",
      branch: "triage/issue-123",
      workflowRuns: () =>
        ++polls === 1
          ? [successfulWorkflow(".github/workflows/tests.yml")]
          : [
              successfulWorkflow(".github/workflows/tests.yml"),
              successfulWorkflow(".github/workflows/approval.yml"),
            ],
    });

    assert.equal(failure, undefined);
    assert.equal(polls, 2);
  } finally {
    global.setTimeout = originalSetTimeout;
  }
});

test("Protect does not require Approval for either unmatched predicate condition", async () => {
  for (const { author, branch } of [
    { author: "contributor", branch: "triage/issue-123" },
    { author: "github-actions[bot]", branch: "feature/issue-123" },
  ]) {
    const failure = await runProtect({
      author,
      branch,
      workflowRuns: [successfulWorkflow(".github/workflows/tests.yml")],
    });
    assert.equal(failure, undefined);
  }
});
