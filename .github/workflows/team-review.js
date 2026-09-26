const STATS_ISSUE_NUMBER = 19428;
const REVIEWER_BALANCE_RULES = [{ reviewers: ["mprahl", "HumairAK"], maxSelected: 1 }];

async function loadStats(github, owner, repo) {
  const issueUrl = `https://github.com/${owner}/${repo}/issues/${STATS_ISSUE_NUMBER}`;
  const issue = await github.rest.issues.get({
    owner,
    repo,
    issue_number: STATS_ISSUE_NUMBER,
  });
  const match = issue.data.body.match(/```json\n([\s\S]*?)\n```/);
  if (!match) {
    throw new Error(`No JSON block found in ${issueUrl}`);
  }

  let stats;
  try {
    stats = JSON.parse(match[1]);
  } catch (err) {
    throw new Error(`Malformed JSON block in ${issueUrl}: ${err.message}`);
  }

  const { reviewCounts } = stats;
  if (typeof reviewCounts !== "object" || reviewCounts === null || Array.isArray(reviewCounts)) {
    throw new Error(`\`reviewCounts\` is missing or not an object in ${issueUrl}`);
  }
  if (Object.keys(reviewCounts).length === 0) {
    throw new Error(`\`reviewCounts\` is empty in ${issueUrl}, so the roster has no members`);
  }

  // A rule naming someone off the roster is inert rather than broken, so warn instead of throwing.
  const staleRuleReviewers = [
    ...new Set(
      REVIEWER_BALANCE_RULES.flatMap((rule) => rule.reviewers.filter((r) => !(r in reviewCounts)))
    ),
  ];
  if (staleRuleReviewers.length > 0) {
    console.warn(
      `Warning: REVIEWER_BALANCE_RULES names ${staleRuleReviewers.join(", ")}, ` +
        `absent from ${issueUrl}. Affected rules have no effect.`
    );
  }
  return stats;
}

async function saveStats(github, owner, repo, stats) {
  const body = `This issue is the source of truth for team review membership. Add or remove a
reviewer by editing the JSON block below. Add new reviewers at roughly the current highest count,
since the lowest counts are assigned first.

\`\`\`json
${JSON.stringify(stats, null, 2)}
\`\`\`
`;

  await github.rest.issues.update({
    owner,
    repo,
    issue_number: STATS_ISSUE_NUMBER,
    body,
  });
}

function shuffle(array) {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

function violatesReviewerBalanceRule(reviewers) {
  return REVIEWER_BALANCE_RULES.some((rule) => {
    const selectedCount = rule.reviewers.filter((reviewer) => reviewers.includes(reviewer)).length;
    return selectedCount > rule.maxSelected;
  });
}

function balanceReviewerSelection(selectedReviewers, candidateReviewers) {
  if (!violatesReviewerBalanceRule(selectedReviewers)) {
    return selectedReviewers;
  }

  for (let i = selectedReviewers.length - 1; i >= 0; i--) {
    for (const candidate of candidateReviewers) {
      if (selectedReviewers.includes(candidate)) {
        continue;
      }

      const replacement = [...selectedReviewers];
      replacement[i] = candidate;
      if (!violatesReviewerBalanceRule(replacement)) {
        return replacement;
      }
    }
  }

  return selectedReviewers;
}

/**
 * Select reviewers with the lowest review counts, with random shuffling within each count tier.
 *
 * @example
 * // Counts: {A: 0, B: 0, C: 1}, need 2
 * // → Group 0: [A, B] → shuffle → [B, A] → select both → [B, A]
 *
 * @example
 * // Counts: {A: 0, B: 1, C: 1}, need 2
 * // → Group 0: [A] → select A
 * // → Group 1: [B, C] → shuffle → [C, B] → select C → [A, C]
 *
 * @example
 * // Counts: {A: 2, B: 2, C: 2}, need 2
 * // → Group 2: [A, B, C] → shuffle → [C, A, B] → select C, A → [C, A]
 */
function selectReviewers(eligibleReviewers, stats, count = 2) {
  if (eligibleReviewers.length === 0) {
    return [];
  }

  const reviewCounts = stats.reviewCounts || {};

  // Group by review count
  const groups = {};
  for (const reviewer of eligibleReviewers) {
    const c = reviewCounts[reviewer] || 0;
    if (!groups[c]) groups[c] = [];
    groups[c].push(reviewer);
  }

  // Process groups from lowest count, shuffle each, and select
  const sortedCounts = Object.keys(groups)
    .map(Number)
    .sort((a, b) => a - b);
  const candidates = [];
  for (const c of sortedCounts) {
    const shuffled = shuffle(groups[c]);
    for (const reviewer of shuffled) {
      candidates.push(reviewer);
    }
  }

  return balanceReviewerSelection(candidates.slice(0, count), candidates);
}

function updateStats(stats, selectedReviewers) {
  const reviewCounts = stats.reviewCounts || {};
  for (const reviewer of selectedReviewers) {
    reviewCounts[reviewer] = (reviewCounts[reviewer] || 0) + 1;
  }
  stats.reviewCounts = Object.fromEntries(
    Object.keys(reviewCounts)
      .sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base", numeric: true }))
      .map((k) => [k, reviewCounts[k]])
  );
  return stats;
}

async function getCopilotInitiator(github, owner, repo, pull_number) {
  const timeline = await github.rest.issues.listEventsForTimeline({
    owner,
    repo,
    issue_number: pull_number,
  });
  return timeline.data.find((e) => e.event === "copilot_work_started")?.actor?.login;
}

module.exports = async ({ github, context }) => {
  const { owner, repo } = context.repo;
  const pull_number = context.payload.pull_request.number;
  const author = context.payload.pull_request.user.login;

  const copilotInitiator = await getCopilotInitiator(github, owner, repo, pull_number);

  // Get existing reviews
  const reviews = await github.rest.pulls.listReviews({
    owner,
    repo,
    pull_number,
  });

  const approved = reviews.data.filter((r) => r.state === "APPROVED").map((r) => r.user.login);
  const requested = context.payload.pull_request.requested_reviewers.map((r) => r.login);

  const stats = await loadStats(github, owner, repo);
  const eligibleReviewers = Object.keys(stats.reviewCounts).filter(
    (m) => !approved.includes(m) && !requested.includes(m) && m !== author && m !== copilotInitiator
  );
  const selectedReviewers = selectReviewers(eligibleReviewers, stats);

  if (selectedReviewers.length > 0) {
    try {
      await github.rest.pulls.requestReviewers({
        owner,
        repo,
        pull_number,
        reviewers: selectedReviewers,
      });
      console.log(`Assigned reviewers: ${selectedReviewers.join(", ")}`);
      console.log(`Review counts before: ${JSON.stringify(stats.reviewCounts || {})}`);

      const updatedStats = updateStats(stats, selectedReviewers);
      await saveStats(github, owner, repo, updatedStats);

      console.log(`Review counts after: ${JSON.stringify(updatedStats.reviewCounts)}`);
      console.log(
        `Saved stats to https://github.com/${owner}/${repo}/issues/${STATS_ISSUE_NUMBER}`
      );
    } catch (error) {
      console.error("Failed to assign reviewers:", error);
    }
  } else {
    console.log("No eligible reviewers available");
  }
};
