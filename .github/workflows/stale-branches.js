// Deletes stale branches: no open PR, not protected, and untouched for longer
// than the cutoff that applies to whoever authored the last commit.

const MS_PER_DAY = 24 * 60 * 60 * 1000;

// Days untouched before sweeping, by last-commit author. Bots recreate their
// branches from scratch, so a stale one is a leftover. null exempts humans,
// whose branches may hold work that lives nowhere else.
const STALE_DAYS = { bot: 30, human: null };

// Bot-pushed and never has a PR, so it passes every other filter.
const EXCLUDED = new Set(["gh-pages"]);

const QUERY = `
  query($owner: String!, $name: String!, $cursor: String) {
    rateLimit { remaining resetAt }
    repository(owner: $owner, name: $name) {
      refs(refPrefix: "refs/heads/", first: 100, after: $cursor) {
        pageInfo {
          hasNextPage
          endCursor
        }
        nodes {
          name
          target {
            ... on Commit {
              committedDate
              author {
                name
                user { login }
              }
            }
          }
          associatedPullRequests(first: 1, states: OPEN) { totalCount }
        }
      }
    }
  }
`;

// `GitActor.user` is always typed `User`, so the `__typename === "Bot"` check
// used on timeline events fails here. The `[bot]` suffix can be on the login, or
// only on the commit name (login `Claude` commits as `anthropic-code-agent[bot]`).
const isBotCommit = (commit) =>
  [commit?.author?.user?.login, commit?.author?.name].some((name) => name?.endsWith("[bot]"));

const findStaleBranches = async (github, { owner, repo }, protectedBranches) => {
  const stale = [];
  let cursor = null;
  let hasNextPage = true;

  while (hasNextPage) {
    const response = await github.graphql(QUERY, { owner, name: repo, cursor });
    const { remaining, resetAt } = response.rateLimit;
    console.log(`Rate limit: ${remaining} remaining, resets at ${resetAt}`);

    const { nodes, pageInfo } = response.repository.refs;
    hasNextPage = pageInfo.hasNextPage;
    cursor = pageInfo.endCursor;

    for (const ref of nodes) {
      if (EXCLUDED.has(ref.name) || protectedBranches.has(ref.name)) {
        continue;
      }

      if (ref.associatedPullRequests.totalCount > 0) {
        continue;
      }

      const staleDays = STALE_DAYS[isBotCommit(ref.target) ? "bot" : "human"];
      if (staleDays === null) {
        continue;
      }

      const days = Math.floor((Date.now() - new Date(ref.target.committedDate)) / MS_PER_DAY);
      if (days <= staleDays) {
        continue;
      }

      stale.push({ name: ref.name, days });
    }
  }

  return stale.sort((a, b) => b.days - a.days);
};

module.exports = async ({ context, github }) => {
  const { owner, repo } = context.repo;
  // Default to a dry run so an unset or unexpected value never deletes.
  const dryRun = process.env.DRY_RUN !== "false";

  // Protection here comes from rulesets, which GraphQL's `branchProtectionRule`
  // does not report. The REST `protected` flag does.
  const branches = await github.paginate(github.rest.repos.listBranches, {
    owner,
    repo,
    per_page: 100,
  });
  const protectedBranches = new Set(branches.filter((b) => b.protected).map((b) => b.name));

  let deleteCount = 0;

  try {
    const stale = await findStaleBranches(github, { owner, repo }, protectedBranches);
    console.log(`Found ${stale.length} stale branches.`);

    for (const { name, days } of stale) {
      if (dryRun) {
        console.log(`[dry run] Would delete ${name} (inactive for ${days} days)`);
        continue;
      }

      console.log(`Deleting ${name} (inactive for ${days} days)`);
      await github.rest.git.deleteRef({ owner, repo, ref: `heads/${name}` });
      deleteCount++;
    }

    console.log(`Deleted ${deleteCount} stale branches.`);
  } catch (error) {
    if (error.status === 429 || error.message?.includes("rate limit")) {
      console.log(`Rate limit hit after deleting ${deleteCount} branches. Exiting gracefully.`);
      return;
    }
    throw error;
  }
};
