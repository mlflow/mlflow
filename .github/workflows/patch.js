// Helper function to parse semantic version for comparison
function parseSemanticVersion(version) {
  const versionStr = version.replace(/^v/, "");
  const cleanVersion = versionStr.replace(/rc\d+$/, "");
  const parts = cleanVersion.split(".").map(Number);

  return {
    major: parts[0] || 0,
    minor: parts[1] || 0,
    micro: parts[2] || 0,
    isRc: versionStr.includes("rc"),
    original: version,
  };
}

// Helper function to compare versions for sorting
function compareVersions(a, b) {
  const versionA = parseSemanticVersion(a.tag_name);
  const versionB = parseSemanticVersion(b.tag_name);

  // Compare major.minor.micro in descending order
  if (versionA.major !== versionB.major) {
    return versionB.major - versionA.major;
  }
  if (versionA.minor !== versionB.minor) {
    return versionB.minor - versionA.minor;
  }
  if (versionA.micro !== versionB.micro) {
    return versionB.micro - versionA.micro;
  }

  // If versions are equal, prefer non-rc over rc
  if (versionA.isRc !== versionB.isRc) {
    return versionA.isRc ? 1 : -1;
  }

  return 0;
}

module.exports = async ({ context, github, core }) => {
  const { owner, repo } = context.repo;
  const { base, number: pull_number } = context.payload.pull_request;
  if (base.ref.match(/^branch-\d+\.\d+$/)) {
    return;
  }

  const pr = await github.rest.pulls.get({
    owner,
    repo,
    pull_number,
  });
  const { body } = pr.data;

  // Skip running this check if PR is filed by a bot
  if (pr.data.user?.type?.toLowerCase() === "bot") {
    core.info(
      `Skipping processing because the PR is filed by a bot: ${pr.data.user?.login || "unknown"}`
    );
    return;
  }

  // Skip running this check on CD automation PRs
  if (!body || body.trim() === "") {
    core.info("Skipping processing because the PR has no body.");
    return;
  }

  const yesRegex = /- \[( |x)\] yes \(this PR will be/gi;
  const yesMatches = [...body.matchAll(yesRegex)];
  const yesMatch = yesMatches.length > 0 ? yesMatches[yesMatches.length - 1] : null;
  const yes = yesMatch ? yesMatch[1].toLowerCase() === "x" : false;
  const noRegex = /- \[( |x)\] no \(this PR will be/gi;
  const noMatches = [...body.matchAll(noRegex)];
  const noMatch = noMatches.length > 0 ? noMatches[noMatches.length - 1] : null;
  const no = noMatch ? noMatch[1].toLowerCase() === "x" : false;

  if (yes && no) {
    core.setFailed(
      "Both yes and no are selected. Please select only one in the `Should this PR be included in the next patch release?` section."
    );
    return;
  }

  if (!yes && !no) {
    core.setFailed(
      "Please fill in the `Should this PR be included in the next patch release?` section."
    );
    return;
  }

  if (no) {
    return;
  }

  // Check if a version label already exists
  const existingLabels = await github.rest.issues.listLabelsOnIssue({
    owner,
    repo,
    issue_number: context.payload.pull_request.number,
  });

  const versionLabelPattern = /^v\d+\.\d+\.\d+$/;
  const existingVersionLabel = existingLabels.data.find((label) =>
    versionLabelPattern.test(label.name)
  );

  if (existingVersionLabel) {
    core.info(
      `Version label ${existingVersionLabel.name} already exists on this PR. Skipping label addition.`
    );
    return;
  }

  const releases = await github.rest.repos.listReleases({
    owner,
    repo,
  });

  // Filter version tags that start with 'v', sort by semantic version, and select the latest
  const versionReleases = releases.data.filter(({ tag_name }) => tag_name.startsWith("v"));
  const sortedReleases = versionReleases.sort(compareVersions);
  const latest = sortedReleases[0];
  const version = latest.tag_name.replace("v", "");
  const [major, minor, micro] = version.replace(/rc\d+$/, "").split(".");
  const nextMicro = version.includes("rc") ? micro : (parseInt(micro) + 1).toString();
  const label = `v${major}.${minor}.${nextMicro}`;
  await github.rest.issues.addLabels({
    owner,
    repo,
    issue_number: context.payload.pull_request.number,
    labels: [label],
  });
};
