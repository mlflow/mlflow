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
  const yesRegex = /- \[( |x)\] yes/gi;
  const yesMatch = yesRegex.exec(patchSection);
  const yes = yesMatch ? yesMatch[1].toLowerCase() === "x" : false;
  const noRegex = /- \[( |x)\] no/gi;
  const noMatch = noRegex.exec(patchSection);
  const no = noMatch ? noMatch[1].toLowerCase() === "x" : false;

  if (yes && no) {
    core.setFailed(
      "Both yes and no are selected. Please select only one in the `Should this PR be included in the next patch release?` section."
    );
  }

  if (!yes && !no) {
    core.setFailed(
      "Please fill in the `Should this PR be included in the next patch release?` section."
    );
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
