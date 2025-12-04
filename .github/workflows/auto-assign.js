const MAINTAINERS = [
  "B-Step62",
  "BenWilson2",
  "daniellok-db",
  "harupy",
  "serena-ruan",
  "TomeHirata",
  "WeichenXu123",
];

module.exports = async ({ github, context }) => {
  const { owner, repo } = context.repo;
  const pull_number = context.issue.number;
  const commenter = context.payload.comment.user.login;

  // Check if the commenter is a maintainer
  console.log("Maintainers:", MAINTAINERS);
  console.log("Commenter:", commenter);

  if (!MAINTAINERS.includes(commenter)) {
    console.log(`${commenter} is not a maintainer, skipping assignment`);
    return;
  }

  // Get PR details to check for reviewers
  const pr = await github.rest.pulls.get({
    owner,
    repo,
    pull_number,
  });

  const requestedReviewers = pr.data.requested_reviewers || [];
  const requestedTeams = pr.data.requested_teams || [];
  console.log(
    "Requested reviewers:",
    requestedReviewers.map((r) => r.login)
  );
  console.log(
    "Requested teams:",
    requestedTeams.map((t) => t.slug)
  );

  // Only assign if there are no reviewers
  if (requestedReviewers.length > 0 || requestedTeams.length > 0) {
    console.log("PR already has reviewers, skipping assignment");
    return;
  }

  // Check if the commenter is already assigned
  const currentAssignees = pr.data.assignees || [];
  console.log(
    "Current assignees:",
    currentAssignees.map((a) => a.login)
  );

  if (currentAssignees.some((a) => a.login === commenter)) {
    console.log(`${commenter} is already assigned, skipping`);
    return;
  }

  // Add the maintainer as an assignee
  console.log(`Adding ${commenter} as assignee`);
  await github.rest.issues.addAssignees({
    owner,
    repo,
    issue_number: pull_number,
    assignees: [commenter],
  });
  console.log(`Successfully added ${commenter} as assignee`);
};
