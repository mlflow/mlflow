const { execFileSync } = require("node:child_process");
const { appendFileSync, readFileSync } = require("node:fs");
const { sanitizeInput } = require("./sanitize.js");

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

function escapeMarkup(text) {
  return text.replace(/[&<>"']/g, (character) => {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" }[character];
  });
}

function findOpenPr(repo, issueNumber, api = ghApi) {
  const owner = repo.split("/")[0];
  const head = encodeURIComponent(`${owner}:triage/issue-${issueNumber}`);
  const pulls = api(`repos/${repo}/pulls?state=open&head=${head}&per_page=100`);
  return pulls.find((pr) => pr.head.repo?.full_name === repo);
}

// In a dry run, an existing triage PR is reported instead of failing the run, since nothing
// would be posted or opened.
function checkExistingPr({ repo, issueNumber, dryRun = false, api = ghApi }) {
  const existing = findOpenPr(repo, issueNumber, api);
  if (!existing) return;
  const message = `A triage PR already exists: ${existing.html_url}`;
  if (!dryRun) throw new Error(message);
  console.error(`Dry run: ${message}`);
  if (process.env.GITHUB_STEP_SUMMARY) {
    appendFileSync(process.env.GITHUB_STEP_SUMMARY, `Dry run: ${message}\n`);
  }
}

function prepareFollowup({ repo, issue, comment, dryRun = false, api = ghApi }) {
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
  checkExistingPr({ repo, issueNumber: issue.number, dryRun, api });

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
        `<comment author="${escapeMarkup(candidate.user?.login ?? "deleted user")}" ` +
        `created_at="${escapeMarkup(candidate.created_at)}" url="${escapeMarkup(
          candidate.html_url
        )}">\n` +
        `${escapeMarkup(sanitizeInput(candidate.body ?? ""))}\n</comment>`
    );
  return {
    body: [
      `<issue>\n${escapeMarkup(sanitizeInput(issue.body ?? ""))}\n</issue>`,
      ...discussion,
    ].join("\n\n"),
    followup: {
      instructions: sanitizeInput(instructions),
      author: comment.user.login,
    },
  };
}

function prepareContext({ repo, issue, event, comment, dryRun = false, api = ghApi }) {
  let request = event.comment;
  if (!request && comment?.trim()) {
    const text = comment.trim();
    request = {
      body: text.startsWith("/triage") ? text : `/triage ${text}`,
      issue_url: issue.url,
      user: event.sender,
    };
  }
  const context = request
    ? prepareFollowup({ repo, issue, comment: request, dryRun, api })
    : { body: issue.body ?? "", followup: null };
  return { title: issue.title, repository: repo, issue_number: issue.number, ...context };
}

module.exports = { parseCommand, findOpenPr, checkExistingPr, prepareContext };

if (require.main === module) {
  try {
    const mode = process.argv[2];
    const dryRun = process.env.DRY_RUN === "true";
    if (mode === "prepare") {
      const event = JSON.parse(readFileSync(process.env.GITHUB_EVENT_PATH, "utf8"));
      const issue = JSON.parse(readFileSync(process.argv[3], "utf8"));
      const context = prepareContext({
        repo: process.env.REPO,
        issue,
        event,
        comment: process.env.COMMENT,
        dryRun,
      });
      process.stdout.write(JSON.stringify(context));
    } else if (mode === "check-pr") {
      checkExistingPr({
        repo: process.env.REPO,
        issueNumber: process.env.ISSUE_NUMBER,
        dryRun,
      });
    } else {
      throw new Error(
        "Usage: node .claude/skills/triage/scripts/utils.js <prepare|check-pr> [path]"
      );
    }
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
