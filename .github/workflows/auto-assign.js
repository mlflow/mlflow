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
  const issue_number = context.issue.number;
  const commenter = context.payload.comment.user.login;

  // Check if the commenter is a maintainer
  console.log("Maintainers:", MAINTAINERS);
  console.log("Commenter:", commenter);

  if (!MAINTAINERS.includes(commenter)) {
    console.log(`${commenter} is not a maintainer, skipping assignment`);
    return;
  }

  // Get issue/PR details to check current assignees
  const issue = await github.rest.issues.get({
    owner,
    repo,
    issue_number,
  });

  const currentAssignees = issue.data.assignees || [];
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
    issue_number,
    assignees: [commenter],
  });
  console.log(`Successfully added ${commenter} as assignee`);
};
