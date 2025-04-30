module.exports = async ({ github, context }) => {
  const { owner, repo } = context.repo;
  const { body, number: issue_number } = context.payload.issue || context.payload.pull_request;
  const pattern = /- \[(.*?)\]\s*`(.+?)`/g;
  // Labels extracted from the issue/PR body
  const bodyLabels = [];
  let match;
  while ((match = pattern.exec(body)) !== null) {
    bodyLabels.push({ checked: match[1].trim().toLowerCase() === "x", name: match[2].trim() });
  }
  console.log("Body labels:", bodyLabels);

  const events = await github.paginate(github.rest.issues.listEvents, {
    owner,
    repo,
    issue_number,
  });
  // Labels added or removed by a user
  const userLabels = events
    .filter(({ event, actor }) => ["labeled", "unlabeled"].includes(event) && actor.type === "User")
    .map(({ label }) => label.name);
  console.log("User labels:", userLabels);

  // Labels available in the repository
  const repoLabels = (
    await github.paginate(github.rest.issues.listLabelsForRepo, {
      owner,
      repo,
    })
  ).map(({ name }) => name);

  // Exclude labels that are not available in the repository or have been added/removed by a user
  const labels = bodyLabels.filter(
    ({ name }) => repoLabels.includes(name) && !userLabels.includes(name)
  );
  console.log("Labels to add/remove:", labels);

  const existingLabels = (
    await github.paginate(github.rest.issues.listLabelsOnIssue, {
      owner,
      repo,
      issue_number,
    })
  ).map(({ name }) => name);
  console.log("Existing labels:", existingLabels);

  const labelsToAdd = labels
    .filter(({ name, checked }) => checked && !existingLabels.includes(name))
    .map(({ name }) => name);
  console.log("Labels to add:", labelsToAdd);
  if (labelsToAdd.length > 0) {
    await github.rest.issues.addLabels({
      owner,
      repo,
      issue_number,
      labels: labelsToAdd,
    });
  }

  const labelsToRemove = labels
    .filter(({ name, checked }) => !checked && existingLabels.includes(name))
    .map(({ name }) => name);
  console.log("Labels to remove:", labelsToRemove);
  if (labelsToRemove.length > 0) {
    const results = await Promise.allSettled(
      labelsToRemove.map((name) =>
        github.rest.issues.removeLabel({
          owner,
          repo,
          issue_number,
          name,
        })
      )
    );
    for (const { status, reason } of results) {
      if (status === "rejected") {
        console.error(reason);
      }
    }
  }
};
