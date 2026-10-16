const { execFileSync } = require("node:child_process");
const { readFileSync } = require("node:fs");
const { sanitizeInput } = require("../workflows/sanitize.js");

function ghApi(endpoint, ...args) {
  return JSON.parse(
    execFileSync("gh", ["api", endpoint, ...args], {
      encoding: "utf8",
      maxBuffer: 64 * 1024 * 1024,
    })
  );
}

function parseCommand(body) {
  const match = /^\/triage(?:\s+([\s\S]+))?$/.exec(body.trim());
  if (!match) throw new Error("Use /triage followed by your instructions.");
  return match[1]?.trim() ?? "";
}

function findOpenPr(repo, issueNumber, api = ghApi) {
  const pulls = api(`repos/${repo}/pulls?state=open&per_page=100`, "--paginate", "--slurp").flat();
  return pulls.find(
    (pr) =>
      pr.head.repo?.full_name === repo && pr.head.ref.startsWith(`triage/issue-${issueNumber}-`)
  );
}

function prepareFollowup({ repo, issue, comment, api = ghApi }) {
  if (issue.pull_request || issue.state !== "open")
    throw new Error("Triage requires an open issue.");
  if (comment.issue_url !== issue.url || comment.user.type !== "User") {
    throw new Error("The command must be a user comment on this issue.");
  }
  const instructions = parseCommand(comment.body);
  const permission = api(
    `repos/${repo}/collaborators/${encodeURIComponent(comment.user.login)}/permission`
  );
  if (!["admin", "maintain"].includes(permission.role_name)) {
    throw new Error("Only repository admins and maintainers can request follow-up triage.");
  }
  const existing = findOpenPr(repo, issue.number, api);
  if (existing) throw new Error(`A triage PR already exists: ${existing.html_url}`);

  const comments = api(
    `repos/${repo}/issues/${issue.number}/comments?per_page=100`,
    "--paginate",
    "--slurp"
  ).flat();
  const discussion = comments
    .filter((candidate) => comment.id === undefined || candidate.id < comment.id)
    .sort((a, b) => a.id - b.id)
    .map(
      (candidate) =>
        `## Comment by ${candidate.user?.login ?? "deleted user"} (${candidate.created_at})\n\n` +
        `${candidate.html_url}\n\n${candidate.body ?? ""}`
    );
  return {
    body: [`## Issue description\n\n${issue.body ?? ""}`, ...discussion].join("\n\n---\n\n"),
    followup: {
      instructions: sanitizeInput(instructions),
      command_url: comment.html_url,
      maintainer: comment.user.login,
    },
  };
}

function prepareContext({ repo, issue, event, comment, runUrl, api = ghApi }) {
  let request = event.comment;
  if (!request && comment?.trim()) {
    const text = comment.trim();
    request = {
      body: text.startsWith("/triage") ? text : `/triage ${text}`,
      issue_url: issue.url,
      html_url: runUrl,
      user: event.sender,
    };
  }
  const context = request
    ? prepareFollowup({ repo, issue, comment: request, api })
    : { body: issue.body ?? "", followup: null };
  return { title: issue.title, repository: repo, issue_number: issue.number, ...context };
}

module.exports = { parseCommand, findOpenPr, prepareContext };

if (require.main === module) {
  try {
    const mode = process.argv[2];
    if (mode === "prepare") {
      const event = JSON.parse(readFileSync(process.env.GITHUB_EVENT_PATH, "utf8"));
      const issue = JSON.parse(readFileSync(process.argv[3], "utf8"));
      const context = prepareContext({
        repo: process.env.REPO,
        issue,
        event,
        comment: process.env.COMMENT,
        runUrl: process.env.RUN_URL,
      });
      process.stdout.write(JSON.stringify(context));
    } else if (mode === "check-pr") {
      const existing = findOpenPr(process.env.REPO, process.env.ISSUE_NUMBER);
      if (existing) throw new Error(`A triage PR already exists: ${existing.html_url}`);
    } else {
      throw new Error("Usage: node .github/triage/github.js <prepare|check-pr> [path]");
    }
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
