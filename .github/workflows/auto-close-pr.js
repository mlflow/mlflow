// Auto-close PRs based on linked-issue policy:
//   1. PRs that attempt to close an issue without the "ready" label.
//   2. PRs that don't link to any issue and change more than LOC_THRESHOLD
//      lines.
// Skips PRs that reference multiple issues (ambiguous intent).
// Only enforces on issues/PRs created on or after 2026-03-10.

const fs = require("fs");
const path = require("path");

const READY_LABEL = "ready";
const PR_TEMPLATE_PATH = ".github/pull_request_template.md";
// The date we introduced the "ready" label policy; skip older issues/PRs.
const CUTOFF_DATE = new Date("2026-03-10T00:00:00Z");
// PRs with more than this many LOC changed must link to an issue.
const LOC_THRESHOLD = 100;

const QUERY = `
  query($owner: String!, $repo: String!, $number: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $number) {
        closingIssuesReferences(first: 10) {
          nodes {
            number
            createdAt
            labels(first: 50) {
              nodes { name }
            }
            assignees(first: 10) {
              nodes { login }
            }
          }
        }
      }
    }
  }
`;

function getTemplateHeadings() {
  const templatePath = path.join(process.env.GITHUB_WORKSPACE, PR_TEMPLATE_PATH);
  try {
    return fs
      .readFileSync(templatePath, "utf8")
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => /^#+\s/.test(line));
  } catch (err) {
    throw new Error(`Failed to read PR template at ${templatePath}: ${err.message}`);
  }
}

function hasIssueReference(body) {
  if (!body) return false;
  // Strip fenced and inline code blocks so references mentioned inside code
  // samples don't count.
  const stripped = body
    .replace(/```[\s\S]*?```/g, "")
    .replace(/~~~[\s\S]*?~~~/g, "")
    .replace(/`[^`\n]*`/g, "");
  // Match `#123`, `owner/repo#123`, or an issue/PR URL.
  const shortRef = /(?:[\w.-]+\/[\w.-]+)?#\d+/;
  const urlRef = /https?:\/\/github\.com\/[\w.-]+\/[\w.-]+\/(?:issues|pull)\/\d+/;
  return shortRef.test(stripped) || urlRef.test(stripped);
}

function getMissingHeadings(body, headings) {
  if (!body) return headings;
  const bodyLines = new Set(body.split("\n").map((line) => line.trim()));
  return headings.filter((h) => !bodyLines.has(h));
}

async function isDatabricksAuthor({ github, context }) {
  const prAuthor = context.payload.pull_request.user.login;
  const { owner, repo } = context.repo;
  const prNumber = context.payload.pull_request.number;

  // Check user profile for Databricks affiliation
  const { data: user } = await github.rest.users.getByUsername({ username: prAuthor });
  if ([user.company, user.email].some((v) => /databricks/i.test(v || ""))) return true;

  // Check commit author emails for @databricks.com
  const commits = await github.paginate(github.rest.pulls.listCommits, {
    owner,
    repo,
    pull_number: prNumber,
    per_page: 100,
  });
  return commits.some((c) => /@databricks\.com$/i.test(c.commit.author.email || ""));
}

async function getCloseReason({ github, context }) {
  const association = context.payload.pull_request.author_association;
  if (["OWNER", "MEMBER", "COLLABORATOR"].includes(association)) return undefined;
  if (context.payload.pull_request.user.type === "Bot") return undefined;

  if (await isDatabricksAuthor({ github, context })) {
    const prAuthor = context.payload.pull_request.user.login;
    console.log(`PR author @${prAuthor} has Databricks affiliation. Skipping.`);
    return undefined;
  }

  const prNumber = context.payload.pull_request.number;
  const prAuthor = context.payload.pull_request.user.login;
  const { owner, repo } = context.repo;

  // Check that the PR body follows the PR template
  const templateHeadings = getTemplateHeadings();
  const prBody = context.payload.pull_request.body;
  const missingHeadings = getMissingHeadings(prBody, templateHeadings);
  const missingRatio = missingHeadings.length / templateHeadings.length;
  console.log(
    `PR #${prNumber} is missing ${missingHeadings.length}/${templateHeadings.length} template section(s).`
  );
  if (missingRatio > 0.5) {
    const missingList = missingHeadings.map((h) => `- ${h.replace(/^#+\s*/, "")}`).join("\n");
    return [
      "This PR was automatically closed because it does not follow the PR template.",
      `<details>\n<summary>Missing sections</summary>\n\n${missingList}\n</details>`,
      `Please update your PR body to include all sections from the [PR template](https://github.com/${owner}/${repo}/blob/master/${PR_TEMPLATE_PATH}) and reopen this PR.`,
    ].join("\n\n");
  }

  const response = await github.graphql(QUERY, { owner, repo, number: prNumber });
  const issues = response.repository.pullRequest.closingIssuesReferences.nodes;

  if (issues.length === 0) {
    // closingIssuesReferences only catches closing keywords (Fixes/Closes/Resolves).
    // Also accept `#123`, `owner/repo#123`, or an issue/PR URL in the PR body.
    if (hasIssueReference(prBody)) {
      console.log(`PR #${prNumber} body contains an issue reference. Skipping.`);
      return undefined;
    }

    const prCreatedAt = new Date(context.payload.pull_request.created_at);
    if (prCreatedAt < CUTOFF_DATE) {
      console.log(`PR #${prNumber} was created before ${CUTOFF_DATE.toISOString()}. Skipping.`);
      return undefined;
    }

    const { additions, deletions } = context.payload.pull_request;
    const totalChanges = additions + deletions;

    if (totalChanges <= LOC_THRESHOLD) {
      console.log(
        `PR #${prNumber} has no linked issue but only ${totalChanges} LOC changed (<= ${LOC_THRESHOLD}). Skipping.`
      );
      return undefined;
    }

    console.log(
      `PR #${prNumber} has no linked issue and ${totalChanges} LOC changed (> ${LOC_THRESHOLD}). Closing.`
    );
    return [
      "This PR was automatically closed because it does not link to an issue.",
      "Please open an issue describing the bug or feature first, wait for a maintainer to triage it, then link it from your PR description (e.g. `Fixes #123`).",
      "Please do not force-push to or delete the PR branch so this PR can be reopened.",
    ].join(" ");
  }

  if (issues.length > 1) {
    console.log(
      `Multiple issues referenced (${issues.map((i) => `#${i.number}`).join(", ")}). Skipping.`
    );
    return undefined;
  }

  const issue = issues[0];
  console.log(`PR #${prNumber} references issue #${issue.number}`);

  // Skip issues created before the cutoff date
  if (new Date(issue.createdAt) < CUTOFF_DATE) {
    console.log(
      `Issue #${issue.number} was created before ${CUTOFF_DATE.toISOString()}. Skipping.`
    );
    return undefined;
  }

  const hasReadyLabel = issue.labels.nodes.some((label) => label.name === READY_LABEL);
  if (!hasReadyLabel) {
    console.log(
      `Issue #${issue.number} is missing the "${READY_LABEL}" label. Closing PR #${prNumber}.`
    );
    return [
      `This PR was automatically closed because #${issue.number} is missing the \`${READY_LABEL}\` label.`,
      "Once a maintainer triages the issue and applies the label, feel free to reopen this PR.",
      "Please do not force-push to or delete the PR branch so this PR can be reopened.",
    ].join(" ");
  }

  const assigneeLogins = issue.assignees.nodes.map((a) => a.login);
  if (assigneeLogins.length > 0 && !assigneeLogins.includes(prAuthor)) {
    const assigneeList = assigneeLogins.map((login) => `@${login}`).join(", ");
    console.log(
      `Issue #${issue.number} is assigned to ${assigneeList} but PR author is @${prAuthor}. Closing PR #${prNumber}.`
    );
    return [
      `This PR was automatically closed because #${issue.number} is assigned to ${assigneeList}.`,
      "If you believe this was done in error, please reach out to a maintainer.",
      "Please do not force-push to or delete the PR branch so this PR can be reopened.",
    ].join(" ");
  }

  console.log(`Issue #${issue.number} has the "${READY_LABEL}" label. No action needed.`);
  return undefined;
}

async function main({ context, github }) {
  const commentBody = await getCloseReason({ github, context });
  if (commentBody !== undefined) {
    const prNumber = context.payload.pull_request.number;
    const { owner, repo } = context.repo;
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: prNumber,
      body: commentBody,
    });

    await github.rest.pulls.update({
      owner,
      repo,
      pull_number: prNumber,
      state: "closed",
    });

    console.log(`PR #${prNumber} closed.`);
  }
}

module.exports = { main, getCloseReason, isDatabricksAuthor };
