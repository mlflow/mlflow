const assert = require("node:assert/strict");
const test = require("node:test");

const approval = require("./approval");

async function runApproval({ author, branch, files = [], reviews = [] }) {
  const collaborators = [{ login: "core-maintainer", role_name: "maintain" }];
  const listCollaborators = async () => collaborators;
  const listReviews = async () => reviews;
  const listFiles = async () => files;
  const github = {
    rest: {
      repos: { listCollaborators },
      pulls: { listReviews, listFiles },
    },
    paginate: async (request) => request(),
  };
  let failure;

  await approval({
    github,
    context: {
      payload: {
        pull_request: { user: { login: author }, head: { ref: branch } },
      },
      repo: { owner: "mlflow", repo: "mlflow" },
      issue: { number: 1 },
    },
    core: { setFailed: (message) => (failure = message) },
  });

  return failure;
}

test("triage bot PR requires core-maintainer approval", async () => {
  const failure = await runApproval({
    author: "github-actions[bot]",
    branch: "triage/issue-123",
  });

  assert.match(failure, /core maintainers/);
});

test("accepted core-maintainer approval passes a triage bot PR", async () => {
  const failure = await runApproval({
    author: "github-actions[bot]",
    branch: "triage/issue-123",
    reviews: [
      {
        state: "APPROVED",
        user: { login: "core-maintainer", type: "User" },
      },
    ],
  });

  assert.equal(failure, undefined);
});

test("a Copilot comment does not approve a triage bot PR", async () => {
  const failure = await runApproval({
    author: "github-actions[bot]",
    branch: "triage/issue-123",
    reviews: [
      {
        state: "COMMENTED",
        user: { login: "copilot[bot]", type: "Bot" },
      },
    ],
  });

  assert.match(failure, /core maintainers/);
});

test("triage branch from an exempt author keeps the existing path exemption", async () => {
  const failure = await runApproval({
    author: "daniellok-db",
    branch: "triage/issue-123",
    files: [{ filename: "mlflow/server/js/src/index.ts" }],
    reviews: [
      {
        state: "APPROVED",
        user: { login: "contributor", type: "User" },
      },
    ],
  });

  assert.equal(failure, undefined);
});

test("github-actions bot on a non-triage branch keeps the normal approval policy", async () => {
  const failure = await runApproval({
    author: "github-actions[bot]",
    branch: "feature/issue-123",
  });

  assert.match(failure, /core maintainers/);
});

test("existing bot author exemptions remain unchanged", async () => {
  for (const author of ["mlflow-app[bot]", "dependabot[bot]"]) {
    const failure = await runApproval({ author, branch: "triage/issue-123" });
    assert.equal(failure, undefined);
  }
});
