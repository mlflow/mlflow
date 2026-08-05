// Auto-close PRs based on linked-issue policy:
//   1. PRs that attempt to close an issue without the "ready" label.
//   2. PRs that don't link to any issue and change more than LOC_THRESHOLD
//      lines.
//   3. PRs that reference an already-owned issue or an issue already claimed by
//      an earlier open PR.
// Only enforces on issues/PRs created on or after 2026-03-10.

const fs = require("fs");
const path = require("path");
const runTeamReview = require("./team-review.js");

const READY_LABEL = "ready";
const TEAM_REVIEW_LABEL = "team-review";
const PR_TEMPLATE_PATH = ".github/pull_request_template.md";
const MAINTAINER_ASSOCIATIONS = new Set(["OWNER", "MEMBER", "COLLABORATOR"]);
// The date we introduced the "ready" label policy; skip older issues/PRs.
const CUTOFF_DATE = new Date("2026-03-10T00:00:00Z");
// PRs with more than this many LOC changed must link to an issue.
const LOC_THRESHOLD = 100;

// The duplicate-PR check intentionally reads the first page only. This keeps the
// workflow cheap for normal issues, while very high-traffic issues can still be
// handled manually if older cross-references fall off this page.
const QUERY = `
  query($owner: String!, $repo: String!, $number: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $number) {
        closingIssuesReferences(first: 10) {
          nodes {
            number
            state
            createdAt
            labels(first: 50) {
              nodes { name }
            }
            assignees(first: 10) {
              nodes { login }
            }
            timelineItems(first: 100, itemTypes: [CROSS_REFERENCED_EVENT]) {
              nodes {
                __typename
                ... on CrossReferencedEvent {
                  willCloseTarget
                  source {
                    __typename
                    ... on PullRequest {
                      number
                      state
                      createdAt
                      url
                      author {
                        login
                      }
                    }
                  }
                }
              }
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

function hasMaintainerAssociation(context) {
  return MAINTAINER_ASSOCIATIONS.has(context.payload.pull_request.author_association);
}

function getIssueLabels(issue) {
  return issue.labels.nodes.map((label) => label.name);
}

function getIssueAssignees(issue) {
  return issue.assignees.nodes.map((assignee) => assignee.login);
}

function getPrLabels(pr) {
  return (pr.labels ?? []).map((label) => (typeof label === "string" ? label : label.name));
}

function getEarlierOpenLinkedPr(issue, currentPr) {
  const currentCreatedAt = new Date(currentPr.created_at ?? currentPr.createdAt);
  return issue.timelineItems.nodes
    .filter((node) => node.__typename === "CrossReferencedEvent")
    .filter((node) => node.willCloseTarget)
    .map((node) => node.source)
    .filter((source) => source?.__typename === "PullRequest")
    .filter((pr) => pr.number !== currentPr.number)
    .filter((pr) => pr.state === "OPEN")
    .filter((pr) => new Date(pr.createdAt) < currentCreatedAt)
    .sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt))[0];
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

async function getPrAction({ github, context }) {
  const prNumber = context.payload.pull_request.number;
  const prAuthor = context.payload.pull_request.user.login;
  const { owner, repo } = context.repo;
  const isMaintainer = hasMaintainerAssociation(context);
  const hasTeamReviewLabel = getPrLabels(context.payload.pull_request).includes(TEAM_REVIEW_LABEL);

  if (context.payload.pull_request.user.type === "Bot") return {};

  let shouldAutoClose = !isMaintainer;
  if (shouldAutoClose && (await isDatabricksAuthor({ github, context }))) {
    console.log(`PR author @${prAuthor} has Databricks affiliation. Skipping close checks.`);
    shouldAutoClose = false;
  }

  // Check that the PR body follows the PR template
  const templateHeadings = getTemplateHeadings();
  const prBody = context.payload.pull_request.body;
  const missingHeadings = getMissingHeadings(prBody, templateHeadings);
  const missingRatio = missingHeadings.length / templateHeadings.length;
  console.log(
    `PR #${prNumber} is missing ${missingHeadings.length}/${templateHeadings.length} template section(s).`
  );
  if (shouldAutoClose && missingRatio > 0.5) {
    const missingList = missingHeadings.map((h) => `- ${h.replace(/^#+\s*/, "")}`).join("\n");
    return {
      closeReason: [
        "This PR was automatically closed because it does not follow the PR template.",
        `<details>\n<summary>Missing sections</summary>\n\n${missingList}\n</details>`,
        `Please update your PR body to include all sections from the [PR template](https://github.com/${owner}/${repo}/blob/master/${PR_TEMPLATE_PATH}) and reopen this PR.`,
      ].join("\n\n"),
    };
  }

  const response = await github.graphql(QUERY, { owner, repo, number: prNumber });
  const issues = response.repository.pullRequest.closingIssuesReferences.nodes;

  if (issues.length === 0) {
    // closingIssuesReferences only catches closing keywords (Fixes/Closes/Resolves).
    // Also accept `#123`, `owner/repo#123`, or an issue/PR URL in the PR body.
    if (!shouldAutoClose) {
      console.log(`PR #${prNumber} has no closing issue. Skipping close checks.`);
      return {};
    }
    if (hasIssueReference(prBody)) {
      console.log(`PR #${prNumber} body contains an issue reference. Skipping.`);
      return {};
    }

    const prCreatedAt = new Date(context.payload.pull_request.created_at);
    if (prCreatedAt < CUTOFF_DATE) {
      console.log(`PR #${prNumber} was created before ${CUTOFF_DATE.toISOString()}. Skipping.`);
      return {};
    }

    const { additions, deletions } = context.payload.pull_request;
    const totalChanges = additions + deletions;

    if (totalChanges <= LOC_THRESHOLD) {
      console.log(
        `PR #${prNumber} has no linked issue but only ${totalChanges} LOC changed (<= ${LOC_THRESHOLD}). Skipping.`
      );
      return {};
    }

    console.log(
      `PR #${prNumber} has no linked issue and ${totalChanges} LOC changed (> ${LOC_THRESHOLD}). Closing.`
    );
    return {
      closeReason: [
        "This PR was automatically closed because it does not link to an issue.",
        "Please open an issue describing the bug or feature first, wait for a maintainer to triage it, then link it from your PR description (e.g. `Fixes #123`).",
        "Please do not force-push to or delete the PR branch so this PR can be reopened.",
      ].join(" "),
    };
  }

  if (issues.length > 1) {
    // GitHub returns closing references in PR body order; treat the first one
    // as the author-selected primary issue.
    console.log(
      `Multiple issues referenced (${issues.map((i) => `#${i.number}`).join(", ")}); using #${
        issues[0].number
      }.`
    );
  }

  const issue = issues[0];
  console.log(`PR #${prNumber} references issue #${issue.number}`);

  // Skip issues created before the cutoff date
  if (new Date(issue.createdAt) < CUTOFF_DATE) {
    console.log(
      `Issue #${issue.number} was created before ${CUTOFF_DATE.toISOString()}. Skipping.`
    );
    return {};
  }

  if (issue.state !== "OPEN") {
    console.log(`Issue #${issue.number} is ${issue.state}.`);
    if (!shouldAutoClose) return {};
    return {
      closeReason: [
        `This PR was automatically closed because #${issue.number} is not open.`,
        "Please open or link to an open issue before submitting a PR.",
        "Please do not force-push to or delete the PR branch so this PR can be reopened.",
      ].join(" "),
    };
  }

  const hasReadyLabel = getIssueLabels(issue).includes(READY_LABEL);
  if (!hasReadyLabel) {
    console.log(`Issue #${issue.number} is missing the "${READY_LABEL}" label.`);
    if (!shouldAutoClose) return {};
    return {
      closeReason: [
        `This PR was automatically closed because #${issue.number} is missing the \`${READY_LABEL}\` label.`,
        "Please discuss the issue first. Once a maintainer triages the issue and applies the label, feel free to reopen this PR.",
        "Please do not force-push to or delete the PR branch so this PR can be reopened.",
      ].join(" "),
    };
  }

  const assigneeLogins = getIssueAssignees(issue);
  if (assigneeLogins.length > 0 && !assigneeLogins.includes(prAuthor)) {
    const assigneeList = assigneeLogins.map((login) => `@${login}`).join(", ");
    console.log(
      `Issue #${issue.number} is assigned to ${assigneeList} but PR author is @${prAuthor}.`
    );
    if (!shouldAutoClose) return {};
    return {
      closeReason: [
        `This PR was automatically closed because #${issue.number} is assigned to ${assigneeList}.`,
        "If there is a valid reason to reassign it, please comment on the issue thread and ping a maintainer.",
        "Please do not force-push to or delete the PR branch so this PR can be reopened.",
      ].join(" "),
    };
  }

  const earlierOpenPr = getEarlierOpenLinkedPr(issue, context.payload.pull_request);
  if (earlierOpenPr !== undefined) {
    if (assigneeLogins.length === 0) {
      console.log(
        `Issue #${issue.number} is already claimed by earlier open PR #${earlierOpenPr.number}.`
      );
      if (!shouldAutoClose) return {};
      return {
        closeReason: [
          `This PR was automatically closed because #${issue.number} is already linked from earlier open PR #${earlierOpenPr.number}.`,
          "Please coordinate on the issue thread and ask a maintainer for reassignment if there is a valid reason.",
          "Please do not force-push to or delete the PR branch so this PR can be reopened.",
        ].join(" "),
      };
    }
    console.log(
      `Issue #${issue.number} is assigned, so earlier open PR #${earlierOpenPr.number} does not override the assigned owner.`
    );
  }

  console.log(`PR #${prNumber} is valid for ready issue #${issue.number}.`);
  if (context.payload.pull_request.draft) {
    console.log(
      `PR #${prNumber} is draft. Deferring "${TEAM_REVIEW_LABEL}" until ready_for_review.`
    );
    return {};
  }

  if (hasTeamReviewLabel) {
    console.log(`PR #${prNumber} already has "${TEAM_REVIEW_LABEL}". Skipping reviewer routing.`);
  }

  // shouldAutoClose only controls enforcement. Ready-issue routing applies once
  // per valid non-bot PR, including maintainers and Databricks authors.
  return {
    issueToAssign: assigneeLogins.length === 0 ? issue.number : undefined,
    addTeamReview: !hasTeamReviewLabel,
  };
}

async function main({ context, github }) {
  const action = await getPrAction({ github, context });
  const prNumber = context.payload.pull_request.number;
  const prAuthor = context.payload.pull_request.user.login;
  const { owner, repo } = context.repo;

  if (action.closeReason !== undefined) {
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: prNumber,
      body: action.closeReason,
    });

    await github.rest.pulls.update({
      owner,
      repo,
      pull_number: prNumber,
      state: "closed",
    });

    console.log(`PR #${prNumber} closed.`);
    return;
  }

  if (action.issueToAssign !== undefined) {
    await github.rest.issues.addAssignees({
      owner,
      repo,
      issue_number: action.issueToAssign,
      assignees: [prAuthor],
    });
    console.log(`Assigned issue #${action.issueToAssign} to @${prAuthor}.`);
  }

  if (action.addTeamReview) {
    await github.rest.issues.addLabels({
      owner,
      repo,
      issue_number: prNumber,
      labels: [TEAM_REVIEW_LABEL],
    });
    console.log(`Added "${TEAM_REVIEW_LABEL}" label to PR #${prNumber}.`);

    // Labels added with GITHUB_TOKEN do not trigger a follow-up workflow run.
    await runTeamReview({ github, context });
  }
}

module.exports = {
  main,
  getPrAction,
  getEarlierOpenLinkedPr,
  isDatabricksAuthor,
};
