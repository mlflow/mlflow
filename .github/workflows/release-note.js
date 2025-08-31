async function fetchRepoLabels({ github, owner, repo }) {
  const { data } = await github.rest.issues.listLabelsForRepo({
    owner,
    repo,
    per_page: 100, // the default value is 30, which is too small to fetch all labels
  });
  return data.map(({ name }) => name);
}

async function fetchPrLabels({ github, owner, repo, issue_number }) {
  const { data } = await github.rest.issues.listLabelsOnIssue({
    owner,
    repo,
    issue_number,
  });
  return data.map(({ name }) => name);
}

function isReleaseNoteLabel(name) {
  return name.startsWith("rn/");
}

async function validateLabeled({ core, context, github }) {
  const { user, html_url: pr_url } = context.payload.pull_request;
  const { owner, repo } = context.repo;
  const { number: issue_number } = context.issue;

  // Skip validation on pull requests created by the automation bot
  if (user.login === "mlflow-app[bot]") {
    console.log("This pull request was created by the automation bot, skipping");
    return;
  }

  const repoLabels = await fetchRepoLabels({ github, owner, repo });
  const releaseNoteLabels = repoLabels.filter(isReleaseNoteLabel);

  // Fetch the release-note category labels applied on this PR
  const fetchAppliedLabels = async () => {
    const backoffs = [0, 1, 2, 4, 8, 16];
    for (const [index, backoff] of backoffs.entries()) {
      console.log(`Attempt ${index + 1}/${backoffs.length}`);
      await new Promise((r) => setTimeout(r, backoff * 1000));
      const prLabels = await fetchPrLabels({
        github,
        owner,
        repo,
        issue_number,
      });
      const prReleaseNoteLabels = prLabels.filter((name) => releaseNoteLabels.includes(name));

      if (prReleaseNoteLabels.length > 0) {
        return prReleaseNoteLabels;
      }
    }
    return [];
  };

  const prReleaseNoteLabels = await fetchAppliedLabels();

  // If no release note category label is applied to this PR, set the action status to "failed"
  if (prReleaseNoteLabels.length === 0) {
    // Make sure '.github/pull_request_template.md' contains an HTML anchor with this name
    const anchorName = "release-note-category";

    // Fragmented URL to jump to the release note category section in the PR description
    const anchorUrl = `${pr_url}#user-content-${anchorName}`;
    const message = [
      "No release-note label is applied to this PR. ",
      `Please select a checkbox in the release note category section: ${anchorUrl} `,
      "or manually apply a release note category label (e.g. 'rn/bug-fix') ",
      "if you're a maintainer of this repository. ",
      "If this job failed when a release note category label is already applied, ",
      "please re-run it.",
    ].join("");
    core.setFailed(message);
  }
}

async function postMerge({ context, github }) {
  const { user } = context.payload.pull_request;
  const { owner, repo } = context.repo;
  const { number: issue_number } = context.issue;

  if (user.login === "mlflow-app[bot]") {
    console.log("This PR was created by the automation bot, skipping");
    return;
  }

  const repoLabels = await fetchRepoLabels({ github, owner, repo });
  const releaseNoteLabels = repoLabels.filter(isReleaseNoteLabel);
  const prLabels = await fetchPrLabels({
    github,
    owner,
    repo,
    issue_number,
  });
  const prReleaseNoteLabels = prLabels.filter((name) => releaseNoteLabels.includes(name));

  if (prReleaseNoteLabels.length === 0) {
    const pull = await github.rest.pulls.get({
      owner,
      repo,
      pull_number: issue_number,
    });
    const { login: mergedBy } = pull.data.merged_by;
    const noneLabel = "rn/none";
    const body = [
      `@${mergedBy} This PR is missing a release-note label, adding \`${noneLabel}\`. `,
      "If this label is incorrect, please replace it with the correct label.",
    ].join("");
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number,
      body,
    });
    await github.rest.issues.addLabels({
      owner,
      repo,
      issue_number,
      labels: [noneLabel],
    });
  }
}

module.exports = {
  validateLabeled,
  postMerge,
};
