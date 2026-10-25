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

const TRIAGE_MARKER = /^[ \t]*<!-- triage(?:[ \t]|\r?\n|$)/m;

function parseIssueFrontmatter(body) {
  const text = body.replace(/\r\n/g, "\n");
  const starts = [...text.matchAll(/^[ \t]*<!-- triage(?:[ \t]|\n|$)/gm)];
  if (!starts.length) return {};
  const blocks = [...text.matchAll(/^[ \t]*<!-- triage[ \t]*\n([\s\S]*?)^[ \t]*-->[ \t]*$/gm)];
  if (starts.length !== 1 || blocks.length !== 1) {
    throw new Error("Expected one complete triage frontmatter comment.");
  }

  const fields = new Map();
  for (const rawLine of blocks[0][1].split("\n")) {
    const line = rawLine.trim();
    if (!line) continue;
    const match = /^([a-z][a-z0-9_]*):[ \t]*(\S+)$/.exec(line);
    if (!match) throw new Error("Invalid triage frontmatter field.");
    const [, key, value] = match;
    if (fields.has(key)) throw new Error(`Duplicate triage field: ${key}`);
    fields.set(key, value);
  }

  const options = Object.fromEntries(fields);
  if (Object.keys(options).some((key) => key !== "skip")) {
    throw new Error("Only skip is supported in triage frontmatter.");
  }
  if (options.skip === undefined) return {};
  if (!["true", "false"].includes(options.skip)) {
    throw new Error("Triage skip must be true or false.");
  }
  return { skip: options.skip === "true" };
}

function isTrustedActor({ repo, actor, api = ghApi }) {
  if (!actor?.login) return false;
  let permission;
  try {
    permission = api(`repos/${repo}/collaborators/${encodeURIComponent(actor.login)}/permission`);
  } catch (error) {
    const detail = `${error.message}\n${error.stderr ?? ""}`;
    if (/HTTP 404\b|404 Not Found/.test(detail)) return false;
    throw error;
  }
  return ["admin", "maintain"].includes(permission.role_name);
}

function resolveIssueOptions({ repo, eventName, event, api = ghApi }) {
  if (eventName !== "issues" || event?.action !== "opened") return {};
  const issue = event.issue;
  const body = issue?.body ?? "";
  if (!TRIAGE_MARKER.test(body)) return {};
  if (!isTrustedActor({ repo, actor: issue.user, api })) return {};
  return parseIssueFrontmatter(body);
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
  if (comment.issue_url !== issue.url) {
    throw new Error("The command must be a comment on this issue.");
  }
  const instructions = parseCommand(comment.body);
  if (!isTrustedActor({ repo, actor: comment.user, api })) {
    throw new Error("Only repository admins and maintainers can request follow-up triage.");
  }
  checkExistingPr({ repo, issueNumber: issue.number, dryRun, api });

  const comments = api(
    `repos/${repo}/issues/${issue.number}/comments?per_page=100`,
    "--paginate",
    "--slurp"
  ).flat();
  const precedingComments = comments
    .filter((candidate) => comment.id === undefined || candidate.id < comment.id)
    .sort((a, b) => a.id - b.id)
    .map((candidate) => ({
      author: candidate.user?.login ?? "deleted user",
      created_at: candidate.created_at,
      url: candidate.html_url,
      body: sanitizeInput(candidate.body ?? ""),
    }));
  return {
    body: issue.body ?? "",
    comments: precedingComments,
    followup: {
      instructions: sanitizeInput(instructions),
      author: comment.user.login,
    },
  };
}

function prepareContext({
  repo,
  issue,
  event,
  eventName = process.env.GITHUB_EVENT_NAME,
  comment,
  dryRun = false,
  api = ghApi,
}) {
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
    : { body: issue.body ?? "", comments: [], followup: null };
  const triage_options = resolveIssueOptions({ repo, eventName, event, api });
  return {
    title: issue.title,
    repository: repo,
    issue_number: issue.number,
    triage_options,
    ...context,
  };
}

module.exports = {
  parseCommand,
  parseIssueFrontmatter,
  isTrustedActor,
  resolveIssueOptions,
  findOpenPr,
  checkExistingPr,
  prepareContext,
};

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
