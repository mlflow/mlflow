module.exports = async ({ core, context, github }) => {
  const { user, html_url: pr_url } = context.payload.pull_request;
  const { owner, repo } = context.repo;
  const { number: issue_number } = context.issue;

  // Skip validation on pull requests created by the automation bot
  if (user.login === "mlflow-automation") {
    console.log(
      "Skipping validation since this pull request was created by the automation bot"
    );
    return;
  }

  // Fetch release note category labels
  const labelsForRepoResp = await github.issues.listLabelsForRepo({
    owner,
    repo,
    per_page: 100, // the default value is 30, which is too small to fetch all labels
  });
  const releaseNoteLabels = labelsForRepoResp.data
    .map(({ name }) => name)
    .filter(name => name.startsWith("rn/"));
  console.log("Available release note category labels:");
  console.log(releaseNoteLabels);

  // Fetch release note category labels applied to this PR
  const listLabelsOnIssueResp = await github.issues.listLabelsOnIssue({
    owner,
    repo,
    issue_number,
  });
  const appliedLabels = listLabelsOnIssueResp.data
    .map(({ name }) => name)
    .filter(name => releaseNoteLabels.includes(name));

  console.log("Release note category labels applied to this PR:");
  console.log(appliedLabels);

  // If no release note category label is applied to this PR, set the action status to "failed"
  if (appliedLabels.length === 0) {
    // Make sure '.github/pull_request_template.md' contains an HTML anchor with this name
    const anchorName = "release-note-category";

    // Fragmented URL to jump to the release note category section in the PR description
    const anchorUrl = `${pr_url}#user-content-${anchorName}`;
    const message = [
      "No release note category label is applied to this PR. ",
      `Please select a checkbox in the release note category section: ${anchorUrl} `,
      "or manually apply a release note category label (e.g. 'rn/bug-fix') ",
      "if you're a maintainer of this repository. ",
      "If this job failed when a release note category label is already applied, ",
      "please re-run it.",
    ].join("");
    core.setFailed(message);
  }
};
