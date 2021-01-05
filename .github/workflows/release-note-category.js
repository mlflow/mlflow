module.exports = ({ core, context }) => {
  const { body, user, html_url } = context.payload.pull_request;

  // Skip validation on pull requests created by the automation bot
  if (user.login === "mlflow-automation") {
    console.log(
      "Skipping validation since this pull request was created by the automation bot"
    );
    return;
  }

  const categories = [
    "rn/feature",
    "rn/breaking-change",
    "rn/bug-fix",
    "rn/documentation",
    "rn/none",
  ];

  // Extract selected release-note categories from the PR description
  const regexp = /^- \[(.*?)\] ?`(.+?)`/gm;
  const selectedCategories = [];
  let match = regexp.exec(body);
  while (match != null) {
    const selected = match[1].trim().toLocaleLowerCase() === "x";
    const name = match[2].trim();

    if (categories.includes(name) && selected) {
      selectedCategories.push(name);
    }
    match = regexp.exec(body);
  }

  console.log("Selected release-note categories:");
  console.log(selectedCategories);

  // ".github/pull_request_template.md" must contain an HTML anchor with this name
  const anchorName = "release-note-category";

  // GitHub prefixes anchor names in markdown with "user-content-"
  const anchorUrl = `${html_url}#user-content-${anchorName}`;

  // If the number of selected release-note categories is not equal to 1,
  // set the action status to "failed"
  const numSelected = selectedCategories.length;
  if (numSelected === 0) {
    core.setFailed(
      "At least one release-note category must be selected: " + anchorUrl
    );
  }
};
