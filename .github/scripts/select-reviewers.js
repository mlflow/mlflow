const STATS_ISSUE_NUMBER = 19428;
const MEMBERS = [
  "B-Step62",
  "BenWilson2",
  "daniellok-db",
  "harupy",
  "serena-ruan",
  "TomeHirata",
  "WeichenXu123",
];

async function loadStats(github, owner, repo) {
  try {
    const issue = await github.rest.issues.get({
      owner,
      repo,
      issue_number: STATS_ISSUE_NUMBER,
    });
    const match = issue.data.body.match(/```json\n([\s\S]*?)\n```/);
    if (match) {
      return JSON.parse(match[1]);
    }
  } catch (err) {
    console.warn(`Warning: Failed to load stats from issue: ${err.message}`);
  }
  return { reviewCounts: {} };
}

async function saveStats(github, owner, repo, stats) {
  const body = `This issue tracks reviewer assignment counts for fair distribution.

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
  const result = [];
  for (const c of sortedCounts) {
    const shuffled = shuffle(groups[c]);
    for (const reviewer of shuffled) {
      result.push(reviewer);
      if (result.length >= count) {
        return result;
      }
    }
  }

  return result;
}

function updateStats(stats, selectedReviewers) {
  const reviewCounts = stats.reviewCounts || {};
  for (const reviewer of selectedReviewers) {
    reviewCounts[reviewer] = (reviewCounts[reviewer] || 0) + 1;
  }
  stats.reviewCounts = reviewCounts;
  return stats;
}

module.exports = async ({ github, context }) => {
  const { owner, repo } = context.repo;
  const pull_number = context.payload.pull_request.number;
  const author = context.payload.pull_request.user.login;

  // Get existing reviews
  const reviews = await github.rest.pulls.listReviews({
    owner,
    repo,
    pull_number,
  });

  const approved = reviews.data.filter((r) => r.state === "APPROVED").map((r) => r.user.login);
  const requested = context.payload.pull_request.requested_reviewers.map((r) => r.login);

  const eligibleReviewers = MEMBERS.filter(
    (m) => !approved.includes(m) && !requested.includes(m) && m !== author
  );

  // Load stats, select reviewers, and update stats
  const stats = await loadStats(github, owner, repo);
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
