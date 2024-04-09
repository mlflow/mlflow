module.exports = async ({ context, github, core }) => {
  const { body, base } = context.payload.pull_request;
  const { owner, repo } = context.repo;

  if (base.ref.match(/^branch-\d+\.\d+$/)) {
    return;
  }

  const marker = "<!-- patch -->";
  if (!body.includes(marker)) {
    return;
  }

  const patchSection = body.split(marker)[1];
  const yesMatch = patchSection.match(/- \[( |x)\] yes/gi);
  const yes = yesMatch ? yesSelected[0].toLowerCase() === "x" : false;
  const noMatch = patchSection.match(/- \[( |x)\] no/gi);
  const no = noMatch ? noSelected[0].toLowerCase() === "x" : false;

  if (yes && no) {
    core.setFailed("Both yes and no are selected. Please select only one.");
  }

  if (!yes && !no) {
    core.setFailed("Please select either yes or no.");
  }

  if (no) {
    return;
  }

  const latestRelease = await github.rest.repos.getLatestRelease({
    owner,
    repo,
  });
  const version = latestRelease.data.tag_name.replace("v", "");
  const [major, minor, micro] = version.split(".");
  const label = `patch-${major}.${minor}.${parseInt(micro) + 1}`;
  await github.rest.issues.addLabels({
    owner,
    repo,
    issue_number: context.payload.pull_request.number,
    labels: [label],
  });
};
