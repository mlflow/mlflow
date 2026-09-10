const assert = require("node:assert/strict");
const test = require("node:test");

const protect = require("./protect.js");

function createGithub(workflowRuns) {
  let workflowRunIndex = 0;
  const requests = [];
  return {
    requests,
    hook: { after() {} },
    rest: {
      actions: { listWorkflowRunsForRepo: {} },
      checks: { listForRef: {} },
    },
    paginate: async (_request, params) => {
      requests.push(params);
      return params.head_sha ? workflowRuns[workflowRunIndex++] : [];
    },
  };
}

function context() {
  return {
    repo: { owner: "owner", repo: "repo" },
    payload: {
      pull_request: {
        head: { sha: "sha" },
        labels: [],
      },
    },
  };
}

test("polls unfinished workflows without fetching jobs", async () => {
  const github = createGithub([
    [{ id: 1, path: "workflow.yml", event: "pull_request", status: "queued" }],
    [{ id: 1, path: "workflow.yml", event: "pull_request", status: "queued" }],
    [{ id: 1, path: "workflow.yml", event: "pull_request", status: "queued" }],
    [
      {
        id: 1,
        path: "workflow.yml",
        event: "pull_request",
        status: "completed",
        conclusion: "success",
        name: "Workflow",
        run_attempt: 1,
        html_url: "https://example.com",
      },
    ],
  ]);
  const timeouts = [];
  const originalSetTimeout = global.setTimeout;
  global.setTimeout = (callback, delay) => {
    timeouts.push(delay);
    callback();
  };

  try {
    await protect({ github, context: context() });
  } finally {
    global.setTimeout = originalSetTimeout;
  }

  assert.deepEqual(timeouts, [15_000, 15_000, 30_000]);
  assert.equal(
    github.requests.some(({ run_id }) => run_id),
    false
  );
});

test("blocks completed workflow failures", async () => {
  const github = createGithub([
    [
      {
        id: 1,
        path: "workflow.yml",
        event: "pull_request",
        status: "completed",
        conclusion: "failure",
        name: "Workflow",
        run_attempt: 1,
        html_url: "https://example.com",
      },
    ],
  ]);

  await assert.rejects(() => protect({ github, context: context() }), /all checks/);
});
