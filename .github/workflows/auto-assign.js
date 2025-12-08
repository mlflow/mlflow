async function getMaintainers({ github, context }) {
  const collaborators = await github.paginate(github.rest.repos.listCollaborators, {
    owner: context.repo.owner,
    repo: context.repo.repo,
  });
  return collaborators
    .filter(({ role_name }) => ["admin", "maintain"].includes(role_name))
    .map(({ login }) => login)
    .sort();
}

module.exports = async ({ github, context }) => {
  const { owner, repo } = context.repo;
  const maintainers = await getMaintainers({ github, context });

  // Get current time minus 200 minutes to look for recent comments and PRs
  const lookbackTime = new Date(Date.now() - 200 * 60 * 1000);

  // Use search API to find recently updated open PRs
  const searchQuery = `repo:${owner}/${repo} is:pr is:open updated:>=${lookbackTime.toISOString()}`;
  const searchResults = await github.paginate(github.rest.search.issuesAndPullRequests, {
    q: searchQuery,
    sort: "updated",
    order: "desc",
  });

  console.log(`Scanning ${searchResults.length} recently updated PRs`);

  for (const pr of searchResults) {
    const prAuthor = pr.user.login;
    const currentAssignees = pr.assignees.map((a) => a.login);

    // Get recent comments and reviews
    const [issueComments, reviews] = await Promise.all([
      github.rest.issues.listComments({
        owner,
        repo,
        issue_number: pr.number,
        since: lookbackTime.toISOString(),
      }),
      github.rest.pulls.listReviews({
        owner,
        repo,
        pull_number: pr.number,
      }),
    ]);

    // Filter reviews by lookback time
    const recentReviews = reviews.data.filter((r) => new Date(r.submitted_at) > lookbackTime);

    // Extract and filter maintainer authors directly
    const allAuthors = [
      ...issueComments.data.map((c) => c.user.login),
      ...recentReviews.map((r) => r.user.login),
    ];
    const maintainersToAssign = [
      ...new Set(
        allAuthors.filter(
          (login) =>
            maintainers.includes(login) && login !== prAuthor && !currentAssignees.includes(login)
        )
      ),
    ];

    if (maintainersToAssign.length === 0) {
      continue;
    }

    // Assign maintainers
    await github.rest.issues.addAssignees({
      owner,
      repo,
      issue_number: pr.number,
      assignees: maintainersToAssign,
    });
    console.log(`Assigned [${maintainersToAssign.join(", ")}] to PR #${pr.number}`);
  }

  console.log("Scan completed");
};
