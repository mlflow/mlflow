async function getMaintainers({ github, context }) {
  const collaborators = await github.paginate(github.rest.repos.listCollaborators, {
    owner: context.repo.owner,
    repo: context.repo.repo,
  });
  return collaborators
    .filter(
      ({ role_name, login }) =>
        ["admin", "maintain"].includes(role_name) ||
        [
          "alkispoly-db",
          "AveshCSingh",
          "danielseong1",
          "smoorjani",
          "SomtochiUmeh",
          "xsh310",
        ].includes(login)
    )
    .map(({ login }) => login)
    .sort();
}

async function getLinkedIssues({ github, owner, repo, prNumber }) {
  const query = `
    query($owner: String!, $repo: String!, $number: Int!) {
      repository(owner: $owner, name: $repo) {
        pullRequest(number: $number) {
          createdAt
          closingIssuesReferences(first: 10) {
            nodes {
              number
              createdAt
            }
          }
        }
      }
    }
  `;

  const result = await github.graphql(query, {
    owner,
    repo,
    number: prNumber,
  });

  return {
    prCreatedAt: result.repository.pullRequest.createdAt,
    issues: result.repository.pullRequest.closingIssuesReferences.nodes,
  };
}

const SEVEN_DAYS_MS = 7 * 24 * 60 * 60 * 1000;

async function findFirstMaintainerInIssueComments({
  github,
  owner,
  repo,
  issueNumber,
  maintainers,
}) {
  for await (const response of github.paginate.iterator(github.rest.issues.listComments, {
    owner,
    repo,
    issue_number: issueNumber,
  })) {
    for (const comment of response.data) {
      if (maintainers.has(comment.user.login)) {
        return comment.user.login;
      }
    }
  }
  return null;
}

module.exports = async ({ github, context, skipAssignment = false }) => {
  const { owner, repo } = context.repo;
  const maintainers = new Set(await getMaintainers({ github, context }));

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
    // Get recent comments and reviews
    const issueComments = await github.rest.issues.listComments({
      owner,
      repo,
      issue_number: pr.number,
      since: lookbackTime.toISOString(),
    });
    const reviews = await github.rest.pulls.listReviews({
      owner,
      repo,
      pull_number: pr.number,
    });

    // Filter reviews by lookback time and extract authors
    const recentReviews = reviews.data.filter((r) => new Date(r.submitted_at) > lookbackTime);
    const commentAuthors = new Set([
      ...issueComments.data.map((c) => c.user.login),
      ...recentReviews.map((r) => r.user.login),
    ]);

    // Check for linked issues and add maintainers who commented on them
    // Skip if we already found maintainers from recent PR activity
    const hasMaintainerFromRecentActivity = [...commentAuthors].some((login) =>
      maintainers.has(login)
    );
    if (!hasMaintainerFromRecentActivity) {
      const { prCreatedAt, issues: linkedIssues } = await getLinkedIssues({
        github,
        owner,
        repo,
        prNumber: pr.number,
      });

      // Only check linked issue comments if exactly one issue is linked
      if (linkedIssues.length === 1 && prCreatedAt) {
        const linkedIssue = linkedIssues[0];
        const prCreatedDate = new Date(prCreatedAt);
        const issueCreatedDate = new Date(linkedIssue.createdAt);

        // Only assign if issue was created within 7 days before the PR
        if (prCreatedDate - issueCreatedDate <= SEVEN_DAYS_MS) {
          const maintainer = await findFirstMaintainerInIssueComments({
            github,
            owner,
            repo,
            issueNumber: linkedIssue.number,
            maintainers,
          });
          if (maintainer) {
            commentAuthors.add(maintainer);
          }
        }
      }
    }

    // Use Set operations to find maintainers to assign
    const prAuthor = pr.user.login;
    const currentAssignees = new Set(pr.assignees.map((a) => a.login));
    const excludeSet = new Set([prAuthor, ...currentAssignees]);

    const maintainersToAssign = [
      ...commentAuthors.intersection(maintainers).difference(excludeSet),
    ];

    if (maintainersToAssign.length === 0) {
      continue;
    }

    // Assign maintainers
    if (!skipAssignment) {
      await github.rest.issues.addAssignees({
        owner,
        repo,
        issue_number: pr.number,
        assignees: maintainersToAssign,
      });
    }
    console.log(
      `${skipAssignment ? "[DRY RUN] Would assign" : "Assigned"} [${maintainersToAssign.join(
        ", "
      )}] to PR #${pr.number}`
    );
  }

  console.log("Scan completed");
};
